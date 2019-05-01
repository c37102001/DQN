import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import ipdb


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v1')
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):               # state array([ 0.00260174,  0.02689104, -0.01369596, -0.02003628])
    state = torch.from_numpy(state).float().unsqueeze(0)    # tensor([[ 0.0026,  0.0269, -0.0137, -0.0200]])
    probs = policy(state)   # tensor([[0.4648, 0.5352]])
    m = Categorical(probs)  # Categorical(probs: torch.Size([1, 2]))
    action = m.sample()     # tensor([1])
    policy.saved_log_probs.append(m.log_prob(action))
    # m.log_prob([0]): tensor([-0.7661]), m.log_prob([1]): tensor([-0.6252])
    return action.item()    # 1(int)


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:      # policy.rewards:[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ...] for 78
        R = r + args.gamma * R
        returns.insert(0, R)            # [..., 4.90099501, 3.9403989999999998, 2.9701, 1.99, 1.0]
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    # returns:
    # tensor([1.4914e+00, 1.4619e+00, 1.4321e+00, 1.4020e+00, 1.3716e+00,
    #         1.3409e+00, 1.3098e+00, 1.2785e+00, 1.2468e+00, 1.2148e+00,
    #         1.1825e+00, 1.1499e+00, 1.1169e+00, 1.0836e+00, 1.0500e+00,
    #         1.0160e+00, 9.8171e-01, 9.4705e-01, 9.1204e-01, 8.7667e-01,
    #         8.4095e-01, 8.0486e-01, 7.6842e-01, 7.3160e-01, 6.9441e-01,
    #         6.5685e-01, 6.1890e-01, 5.8058e-01, 5.4187e-01, 5.0276e-01,
    #         4.6326e-01, 4.2336e-01, 3.8306e-01, 3.4235e-01, 3.0123e-01,
    #         2.5970e-01, 2.1774e-01, 1.7536e-01, 1.3256e-01, 8.9319e-02,
    #         4.5643e-02, 1.5266e-03, -4.3036e-02, -8.8049e-02, -1.3352e-01,
    #         -1.7944e-01, -2.2583e-01, -2.7269e-01, -3.2002e-01, -3.6783e-01,
    #         -4.1613e-01, -4.6491e-01, -5.1418e-01, -5.6395e-01, -6.1423e-01,
    #         -6.6501e-01, -7.1631e-01, -7.6812e-01, -8.2046e-01, -8.7332e-01,
    #         -9.2672e-01, -9.8066e-01, -1.0351e+00, -1.0902e+00, -1.1458e+00,
    #         -1.2019e+00, -1.2586e+00, -1.3159e+00, -1.3738e+00, -1.4323e+00,
    #         -1.4913e+00, -1.5509e+00, -1.6112e+00, -1.6720e+00, -1.7335e+00,
    #         -1.7956e+00, -1.8583e+00, -1.9217e+00])

    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()  # policy_loss:list of tensor -> tensor(of list) ->tensor(0.6655)
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0   # state array([ 0.00260174,  0.02689104, -0.01369596, -0.02003628])
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)   # action = 1 (int)
            state, reward, done, _ = env.step(action)
            # (array([ 0.00313956,  0.2222067 , -0.01409669, -0.31700878]), 1.0, False, {})
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()