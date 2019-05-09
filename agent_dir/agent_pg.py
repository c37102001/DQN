import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_dir.agent import Agent
from environment import Environment
from torch.distributions import Categorical
import ipdb
from matplotlib import pyplot as plt


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_prob = F.softmax(x, dim=1)
        return action_prob


class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        self.model = PolicyNet(state_dim=self.env.observation_space.shape[0],
                               action_num=self.env.action_space.n,
                               hidden_dim=128)
        if args.test_pg:
            self.load('pg.cpt')
        
        # discounted reward
        self.gamma = 0.99 
        
        # training hyperparameters
        self.num_episodes = 10000  # total training episodes (actually too large...)
        self.display_freq = 10  # frequency to display training progress
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        # saved rewards and actions
        self.rewards, self.action_log_probs = [], []
    
    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.action_log_probs = [], []

    def make_action(self, state, test=False):
        # TODO:
        # Use your model to output distribution over actions and sample from it.
        # HINT: google torch.distributions.Categorical

        # state: ndarray array([0.0051,  1.40,  0.519, -0.4636, -0.005, -0.117, 0, 0], dtype=float32)
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = self.model.forward(state)    # action_probs: tensor([[0.2631, 0.2631, 0.2166, 0.2573]])
        action_distributions = Categorical(action_probs)  # Categorical(action_probs: torch.Size([1, 4]))
        action = action_distributions.sample()      # tensor([2])
        self.action_log_probs.append(action_distributions.log_prob(action))
        return action.item()

    def update(self):
        reward, loss, rewards = 0, [], []
        for r in self.rewards[::-1]:  # policy.rewards:[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ...] for 78
            reward = r + self.gamma * reward
            rewards.append(reward)
        rewards = rewards[::-1]
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / rewards.std()
        for log_prob, reward in zip(self.action_log_probs, rewards):
            loss.append(-log_prob * reward)
        loss = torch.cat(loss).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        avg_reward = None  # moving average of reward
        rewards_in_ten = np.array([])
        plot_epoch = []
        plot_avg_reward = []
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while not done:
                action = self.make_action(state)
                state, reward, done, _ = self.env.step(action)
                # (array([0.0154, 1.382, 0.531, -0.3916, -0.0173, -0.1057, 0, 0]), 0.50118 , False, {})

                self.rewards.append(reward)

            # for logging 
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1
            
            # update model
            self.update()

            # for plotting
            plot_epoch.append(epoch)
            rewards_in_ten = np.append(rewards_in_ten, last_reward)
            if len(rewards_in_ten) > 10:
                rewards_in_ten = np.delete(rewards_in_ten, 0)
            plot_avg_reward.append(rewards_in_ten.mean())

            if epoch % self.display_freq == 0:
                print('Epochs: %d/%d | Avg reward: %f ' % (epoch, self.num_episodes, avg_reward))
            
            if avg_reward > 50:  # to pass baseline, avg. reward > 50 is enough.
                self.save('pg.cpt')
                break

        return plot_epoch, plot_avg_reward





