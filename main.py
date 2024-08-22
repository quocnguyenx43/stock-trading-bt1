import gym
import numpy as np
import pandas as pd
import torch
import random
from tqdm import tqdm
from torch.autograd import Variable
from collections import deque
from ddpg import DDPGAgent
from environment import StockTradingEnv
from utils import ReplayBuffer


df = pd.read_csv('./data/ssi.csv')
df['time'] = pd.to_datetime(df['time'], format='%m/%d/%Y')
df_train = df[df['time'] < pd.to_datetime('31/12/2019')]

train_env = StockTradingEnv(df)
obs_shape = train_env.observation_space.shape
act_shape = train_env.action_space.shape
act_range = {'high': train_env.action_space.high, 'low': train_env.action_space.low}


n_episodes = 10
batch_size = 64
epsilon = {'start': 0.9, 'decay': 0.05, 'min': 0.001}
max_timesteps = 200
reward_hist = []
noise_scale = 0.1

# agent and memory
agent = DDPGAgent(obs_shape, act_shape, act_range, epsilon=epsilon['start'])
replay_buffer = ReplayBuffer(obs_shape, act_shape)

for episode in range(n_episodes):
    state = train_env.reset()
    # state = torch.rand(obs_shape)
    replay_buffer.clear()
    ep_reward = 0
    done = False

    print(episode)
    while not done:
        # chose action
        action = agent.act(state, noise_scale)

        # interact with env and store experiences
        next_state, reward, done, _ = train_env.step(action)
        # next_state = torch.rand(obs_shape)
        # reward = torch.rand(1) * 20 - 10
        replay_buffer.store(state, action, reward, next_state, done)

        # model learning
        if replay_buffer.size >= batch_size:
            batch = replay_buffer.sample_batch(batch_size)
            agent.learn(batch)

        # move to the next state
        state = next_state
        ep_reward += reward
    
    if episode >= 1:
        agent.epsilon = -np.inf

    agent.save_model()
    
    print('Reward: ', ep_reward)
    reward_hist.append(ep_reward)
