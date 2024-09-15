import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

class PPO:
    """
    Proximal Policy Optimization (PPO) algorithm implementation.

    Parameters:
    - policy_class (object): Policy class for the actor and critic networks.
    - env (object): Environment for training the agent.
    - lr (float): Learning rate for the optimizer.
    - gamma (float): Discount factor for future rewards.
    - clip (float): Clip parameter for PPO.
    - n_updates (int): Number of updates per episode for PPO.
    """
    def __init__(self, policy_class, env, lr, gamma, clip, ent_coef, critic_factor, max_grad_norm, gae_lambda, n_updates):
        self.lr = lr 
        self.gamma = gamma
        self.clip = clip
        self.n_updates = n_updates

        self.env = env
        self.s_dim = env.observation_space.shape[0]
        self.a_dim = env.action_space.shape[0]

        self.actor = policy_class(self.s_dim, self.a_dim)
        self.critic = policy_class(self.s_dim, 1)

        self.cov_var = nn.Parameter(torch.full(size=(self.a_dim,), fill_value=1.0))
        self.cov_mat = torch.diag(self.cov_var)

        self.actor_optim = Adam(list(self.actor.parameters()) + [self.cov_var], lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters() , lr=self.lr)

    def select_action(self, s):
        mean = self.actor(s)
        dist = MultivariateNormal(mean, self.cov_mat)
        a = dist.sample()
        log_prob = dist.log_prob(a)
        return a.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_s, batch_a):
        """
        Evaluate the policy and value function.

        Parameters:
        - batch_s (Tensor): Batch of states.
        - batch_a (Tensor): Batch of actions.

        Returns:
        - V (Tensor): Value function estimates.
        - log_prob (Tensor): Log probabilities of the actions.
        - entropy (Tensor): Entropy of the action distribution.
        """
        V = self.critic(batch_s).squeeze()
        mean = self.actor(batch_s)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_prob = dist.log_prob(batch_a)
        return V, log_prob

    def compute_G(self, batch_r, batch_terminal, V):
        """
        Compute the episodic returns.

        Parameters:
        - batch_r (Tensor): Batch of rewards.

        Returns:
        - G (Tensor): Episodic returns.
        - A (Tensor): Advantage estimates.
        """
        G = 0
        batch_G = []
        for r in reversed(batch_r):
            G = r + self.gamma*G
            batch_G.insert(0, G)
        batch_G = torch.tensor(batch_G, dtype=torch.float)
        return batch_G

    def update(self, batch_r, batch_s, batch_a, batch_terminal):
        """
        Perform PPO update step.

        Parameters:
        - batch_r (Tensor): Batch of rewards.
        - batch_s (Tensor): Batch of states.
        - batch_a (Tensor): Batch of actions.
        - batch_terminal (Tensor): Batch of terminal flags.
        """
        V, old_log_prob = self.evaluate(batch_s, batch_a)
        old_log_prob = old_log_prob.detach()
        batch_G = self.compute_G(batch_r, batch_terminal, V)
        A = batch_G - V.detach()
        A = (A - A.mean())/(A.std() + 1e-10)
        for i in range(self.n_updates):
            V, log_prob = self.evaluate(batch_s, batch_a)
            ratios = torch.exp(log_prob - old_log_prob)

            term1 = ratios*A
            term2 = torch.clamp(ratios, 1-self.clip, 1+self.clip)*A

            actor_loss = (-torch.min(term1, term2)).mean()
            critic_loss = nn.MSELoss()(V, batch_G)

            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            
    def save(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'cov_var': self.cov_var
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.cov_var = checkpoint['cov_var']
        
class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.ln1 = nn.LayerNorm(64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

        nn.init.normal_(self.layer1.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.layer2.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.layer3.weight, mean=0.0, std=0.1)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        activation1 = F.relu(self.ln1(self.layer1(obs)))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        return output