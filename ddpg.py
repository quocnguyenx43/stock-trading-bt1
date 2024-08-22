import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from networks import Actor, Critic
from utils import OUNoise, ReplayBuffer


class DDPGAgent:
    def __init__(self, obs_shape, act_shape, act_range, epsilon):
        self.gamma = 0.99
        self.tau = 0.005
        self.lr = 1e-4
        self.epsilon = epsilon

        self.act_range = act_range
        obs_dim = obs_shape[0] * obs_shape[1]
        act_dim = act_shape[0]

        # actor, critic networks
        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim, act_dim)
        self.actor_target = Actor(obs_dim, act_dim)
        self.critic_target = Critic(obs_dim, act_dim)

        # copy params to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        # noise to add to action
        self.noise = OUNoise(act_dim)

    def act(self, state, noise_scale):
        state = state.view(-1).to(torch.float32).unsqueeze(0)
        if np.random.rand() < self.epsilon:
            action = np.random.uniform(self.act_range['low'], self.act_range['high'])
        else:
            action = self.actor(state).detach().numpy()[0]
            action += noise_scale * self.noise.sample()
        return np.clip(action, self.act_range['low'], self.act_range['high'])


    def learn(self, batch):
        # extract
        state = torch.FloatTensor(batch['obs'])
        action = torch.FloatTensor(batch['act'])
        reward = torch.FloatTensor(batch['rew']).unsqueeze(1)
        next_state = torch.FloatTensor(batch['obs2'])
        done = torch.FloatTensor(batch['done']).unsqueeze(1)

        # flatten
        state = state.view(state.shape[0], -1)
        action = action.view(action.shape[0], -1)
        reward = reward.view(reward.shape[0], -1)
        next_state = next_state.view(next_state.shape[0], -1)
        done = done.view(done.shape[0], -1)

        # critic loss
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            q_target_next = self.critic_target(next_state, next_action)
            q_target = reward + (1 - done) * self.gamma * q_target_next

        q_val = self.critic(state, action)
        critic_loss = nn.MSELoss()(q_val, q_target)

        # actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # update networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # soft update target networks
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save_model(self):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, 'models/ddpg.pth')
    
    def load_model(self, path):
        model_state = torch.load(path)
        self.actor.load_state_dict(model_state['actor'])
        self.critic.load_state_dict(model_state['critic'])
        self.actor_optimizer.load_state_dict(model_state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(model_state['critic_optimizer'])