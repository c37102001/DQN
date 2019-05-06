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
        self.rewards, self.saved_actions, self.saved_log_probs = [], [], []
    
    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.saved_actions, self.saved_log_probs = [], [], []

    def make_action(self, state, test=False):
        # TODO:
        # Use your model to output distribution over actions and sample from it.
        # HINT: google torch.distributions.Categorical

        # state: ndarray array([0.0051,  1.40,  0.519, -0.4636, -0.005, -0.117, 0, 0], dtype=float32)
        state = torch.from_numpy(state).float().unsqueeze(0)
        # state: tensor([[ 0.0051,  1.4005,  0.5199, -0.4637, -0.0059, -0.1178,  0.0000,  0.0000]])
        probs = self.model.forward(state)
        # probs: tensor([[0.2631, 0.2631, 0.2166, 0.2573]], grad_fn=<SoftmaxBackward>)
        m = Categorical(probs)  # Categorical(probs: torch.Size([1, 4]))
        action = m.sample()     # tensor([2])
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def update(self):
        R = 0
        policy_loss = []
        returns = []
        eps = np.finfo(np.float32).eps.item()
        for r in self.rewards[::-1]:  # policy.rewards:[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ...] for 78
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
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

                self.saved_actions.append(action)
                self.rewards.append(reward)

            # for logging 
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1
            
            # update model
            self.update()

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
        plt.title('learning curve - pg')
        plt.xlabel('Epochs')
        plt.ylabel('Avg rewards in last 10 epochs')
        plt.plot(plot_epoch, plot_avg_reward)
        plt.savefig('pg_learning_curve.png')




