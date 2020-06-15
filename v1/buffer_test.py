from unityagents import UnityEnvironment
import envs
# from buffer import ReplayBuffer
from maddpg import MADDPG
import torch
import numpy as np
import os
import progressbar as pb
from utilities import transpose_list, transpose_to_tensor
from collections import deque, namedtuple
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)


def pre_process(entity, batchsize):
    processed_entity = []
    for j in range(3):
        list = []
        for i in range(batchsize):
            b = entity[i][j]
            list.append(b)
        c = torch.Tensor(list)
        processed_entity.append(c)
    return processed_entity


env = UnityEnvironment(file_name="Tennis_Linux_NoVis/Tennis.x86_64")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# Examine Rewards
print('The rewards looks like: ', env_info.rewards)


# MAIN
batchsize = 1
scores = np.zeros(num_agents)

# amplitude of OU noise
# this slowly decreases to 0
noise = 2
noise_reduction = 0.9999

# keep 1e6 episodes worth of replay
# buffer = ReplayBuffer(int(1))

# Replay memory
BUFFER_SIZE = 1
BATCH_SIZE = 1
buffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, 42)

reward_this_episode = np.zeros((1, 2))

env_info = env.reset(train_mode=True)[brain_name]     # reset the environment
obs = env_info.vector_observations

# initialize policy and critic
maddpg = MADDPG()

actions = maddpg.act(transpose_to_tensor(np.transpose(obs)), noise=noise)

noise *= noise_reduction
actions_array = torch.stack(actions).detach().numpy()

actions_for_env = np.rollaxis(actions_array, 1)

# step forward one frame
# next_obs, next_obs_full, rewards, dones, info = env.step(actions_for_env)
env_info = env.step(actions_for_env)[brain_name]  # send all actions to tne environment
next_obs = env_info.vector_observations  # get next state (for each agent)
rewards = env_info.rewards  # get reward (for each agent)
dones = env_info.local_done  # see if episode finished
scores += env_info.rewards  # update the score (for each agent)

# Rewards
reward_this_episode += rewards

# add data to buffer
transition = (obs, actions_for_env, rewards, next_obs, dones)

# buffer.push(transition)
buffer.add(obs, actions_for_env, np.transpose(rewards), next_obs, np.transpose(dones))

samples = buffer.sample()
maddpg.update(samples, 0)


