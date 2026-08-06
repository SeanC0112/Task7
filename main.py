import numpy as np
import pandas as pd
import gymnasium as gym
import torch
from torch import nn
from collections import namedtuple, deque
import math, random
import matplotlib.pyplot as plt
from itertools import count
import os

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
torch.set_default_device(device=device)
print(f"Using {device} device")

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 150000
TAU = 0.005
LR = 3e-4

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

state = torch.zeros((1,4,96,96), dtype=torch.float32, device=device)

class Q_Value_Function(nn.Module): 
    def __init__(self, number_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=12, stride=4), # different from the paper convolutional size because 8x8 stride 4 would give a 23x23 output, which would be awkward
            nn.ReLU(),
            nn.LazyConv2d(32, kernel_size=4, stride=2), # same size as paper because this [produces a nice 10x10x32 output
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(number_actions)
        )

    def forward(self, x):
        return self.model(x)



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

policy_model = Q_Value_Function(number_actions=env.action_space.n)
target_model = Q_Value_Function(number_actions=env.action_space.n)
target_model.load_state_dict(policy_model.state_dict())

memory = ReplayMemory(10000)
optimizer = torch.optim.RMSprop(policy_model.parameters(), lr=LR, alpha=0.95)

num_episodes = 500
train_loss = []
running_loss = deque(maxlen=250)
for _ in range (250):
    running_loss.append(0)
running_reward = deque(maxlen=15)
# for _ in range (15):
#     running_reward.append(0)
ep_reward = []


steps = 0
episodes = 0

def select_action(state, steps):
    sample = random.random()
    threshold = max(EPS_END, EPS_START * math.exp(-1. * steps / EPS_DECAY))
    if sample < threshold:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    else:
        with torch.no_grad():
            return policy_model(state).max(1).indices.view(1, 1)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_model(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_model.parameters(), 100)
    optimizer.step()
    running_loss.append(loss.item())
    train_loss.append(sum(running_loss)/len(running_loss))

for episodes in range(num_episodes):
    # Initialize the environment and get its state
    state = torch.zeros((1,4,96,96), dtype=torch.float32, device=device)
    obs, info = env.reset()
    state = preprocess(obs, state)
    with torch.no_grad():
        policy_model.forward(state)
        target_model.forward(state)

    total_reward = 0
    for t in count():
        steps += 1
        action = select_action(state, steps)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = preprocess(observation, state)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_model.state_dict()
        policy_net_state_dict = policy_model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_model.load_state_dict(target_net_state_dict)

        if done:
            break

    for filename in os.listdir('.plots'):
        os.remove(os.path.join('.plots', filename))

    running_reward.append(total_reward)
    ep_reward.append(sum(running_reward)/len(running_reward))
    plt.plot(range(len(ep_reward)), ep_reward)
    plt.title("Episode Rewards")
    plt.xlabel("Time")
    plt.ylabel("Rewards")
    plt.savefig('.plots/reward_plot.png')
    plt.close()  

    plt.plot(range(len(train_loss)), train_loss)
    plt.title("Training Loss")
    plt.xlabel("Time")
    plt.ylabel("Loss") 
    plt.savefig('.plots/train_loss.png')
    plt.close() 

torch.save(policy_model, "policymodel.pth")