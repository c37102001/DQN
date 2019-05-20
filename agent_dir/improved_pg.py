import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.utils as utils

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


class Agent_Improved_PG(Agent):
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
        self.rewards, self.log_probs = [], []
        self.actions, self.states = [], []

        self.trajs = []
        self.result = []
        self.num_trajs = 10

    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.log_probs = [], []
        self.actions, self.states = [], []

    def make_action(self, state, test=False):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.model.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action

    def cal_log_prob(self, state, action):

        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.model.forward(state)
        m = Categorical(probs)
        log_prob = m.log_prob(action)

        return log_prob

    def update(self):

        t_states, t_actions, t_rewards, t_log_probs = cvt_axis(self.trajs)
        t_Rs = reward_to_value(t_rewards, self.gamma)

        Z = 0
        b = 0
        losses = []
        Z_s = []

        for (states, actions, Rs, log_probs) in zip(t_states, t_actions, t_Rs, t_log_probs):
            p_log_prob = 0
            q_log_prob = 0
            for t in range(len(Rs)):
                p_log_prob += (self.cal_log_prob(states[t], actions[t])).data.numpy()
                q_log_prob += log_probs[t].data.numpy()
            Z_ = math.exp(p_log_prob) / math.exp(q_log_prob) if not math.exp(q_log_prob) == 0 else 1
            Z += Z_
            Z_s.append(Z_)
            b += Z_ * sum(Rs) / len(Rs)
        b = b / Z

        for (states, actions, Rs, log_probs) in zip(t_states, t_actions, t_Rs, t_log_probs):
            loss = []

            for t in range(len(Rs)):
                loss.append(-log_probs[t] * (Rs[t]-b))
            loss = torch.cat(loss).sum()

            Z_ = Z_s.pop(0)
            loss = loss / Z_
            losses.append(loss)

        loss = sum(losses) / Z

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        utils.clip_grad_norm(self.model.parameters(), 40)
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
                self.states.append(state)
                action = self.make_action(state)
                self.actions.append(action)
                state, reward, done, _ = self.env.step(action.item())
                # (array([0.0154, 1.382, 0.531, -0.3916, -0.0173, -0.1057, 0, 0]), 0.50118 , False, {})

                self.rewards.append(reward)

            if len(self.trajs) > self.num_trajs:
                self.trajs.pop(0)

            self.trajs.append((self.states, self.actions, self.rewards, self.log_probs))


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


def cvt_axis(trajs):
    t_states = []
    t_actions = []
    t_rewards = []
    t_log_probs = []

    for traj in trajs:
        t_states.append(traj[0])
        t_actions.append(traj[1])
        t_rewards.append(traj[2])
        t_log_probs.append(traj[3])

    return t_states, t_actions, t_rewards, t_log_probs


def reward_to_value(t_rewards, gamma):
    t_Rs = []

    for rewards in t_rewards:
        Rs = []
        R = torch.zeros(1, 1)

        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            Rs.insert(0, R)
        # Rs = torch.tensor(Rs)
        # Rs = (Rs - Rs.mean()) / Rs.std()
        t_Rs.append(Rs)

    return t_Rs


