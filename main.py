import warnings
warnings.filterwarnings("ignore")
from torch import multiprocessing

from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing, nn

from torchrl.collectors import Collector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ToTensorImage,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
import gymnasium as gym

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
torch.set_default_device(device)
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

frames_per_batch = 1000
# For a complete training, bring the number of frames up to 1M
total_frames = 500000

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

base_env = GymEnv("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=True, continuous=False)


env = TransformedEnv(
    base_env,
    Compose(
        # normalize observations
        ToTensorImage(in_keys=["pixels"]), #use to tensor image instead bc its an image, not some data abt the position
        StepCounter(),
    ),
)

check_env_specs(env)

# print("action_spec (as defined by input_spec):", env.action_spec.shape[-1])

actor_net = nn.Sequential(
            nn.LazyConv2d(16, kernel_size=12, stride=4), 
            nn.Tanh(),
            nn.LazyConv2d(32, kernel_size=4, stride=2), 
            nn.Tanh(),
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.Tanh(),
            nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
            NormalParamExtractor()
        )

policy_module = TensorDictModule(
    actor_net, in_keys=["pixels"], out_keys=["loc", "scale"]
)

policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    # distribution_kwargs={
    #     "low": env.action_spec_unbatched.space.low,
    #     "high": env.action_spec_unbatched.space.high,
    # },
    return_log_prob=True,
)

value_net = nn.Sequential(
            nn.LazyConv2d(16, kernel_size=12, stride=4), 
            nn.Tanh(),
            nn.LazyConv2d(32, kernel_size=4, stride=2), 
            nn.Tanh(),
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.Tanh(),
            nn.LazyLinear(1, device=device)
        )

value_module = ValueOperator(
    module=value_net,
    in_keys=["pixels"],
)

print("Running policy:", policy_module(env.reset()))
print("Running value:", value_module(env.reset()))

collector = Collector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coeff=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coeff=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)