import warnings
warnings.filterwarnings("ignore")
from torch import multiprocessing


from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
import os
import gymnasium as gym
import numpy as np

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
torch.set_default_device(device=device)
print(f"Using {device} device")

Alpha = 1
Horizon = 128
Adam_stepsize = 2.5e-4 * Alpha
Num_epochs = 3
Minibatch_size = 32 * 8
Discount = 0.99
GAE_parameter = 0.95
Number_of_actors = 8
Clipping_parameter = 0.1 * Alpha

state = torch.zeros((1,4,96,96), dtype=torch.float32, device=device)



def preprocess(obs, prev_state):
    #use ITU-R 601-2 luma formula to convert to grayscale
    obs = np.dot(obs[..., :3], [0.299, 0.587, 0.114])


    # obs = obs[..., np.newaxis]
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    state = torch.cat([prev_state, obs], dim=1)
    if state.size(1) > 4:
        state = state[:, 1:, ...]

    return state


        

env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=False, max_episode_steps=1000)


frames_per_batch = 1000
total_frames = 500000

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4


    # running_reward.append(total_reward)
    # ep_reward.append(sum(running_reward)/len(running_reward))
    # plt.plot(range(len(ep_reward)), ep_reward)
    # plt.title("Episode Rewards")
    # plt.xlabel("Time")
    # plt.ylabel("Rewards")
    # plt.savefig('.plots/reward_plot.png')
    # plt.close()  

    # plt.plot(range(len(train_loss)), train_loss)
    # plt.title("Training Loss")
    # plt.xlabel("Time")
    # plt.ylabel("Loss") 
    # plt.savefig('.plots/train_loss.png')
    # plt.close() 

torch.save(policy_model, "policymodel.pth")