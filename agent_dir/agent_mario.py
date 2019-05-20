import torch
from torch.distributions import Categorical
from torch.optim import RMSprop
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
import torch.nn.functional as F
from matplotlib import pyplot as plt

from a2c.environment_a2c import make_vec_envs
from a2c.storage import RolloutStorage
from a2c.actor_critic import ActorCritic

from collections import deque
import os
import ipdb

use_cuda = torch.cuda.is_available()

class AgentMario:
    def __init__(self, env, args):

        # Hyperparameters
        self.lr = 7e-4
        self.gamma = 0.9
        self.hidden_size = 512
        self.update_freq = 5
        self.n_processes = 16
        self.seed = 7122
        self.max_steps = 1e4
        self.grad_norm = 0.5
        self.entropy_weight = 0.05

        #######################    NOTE: You need to implement
        self.recurrent = True # <- ActorCritic._forward_rnn()
        #######################    Please check a2c/actor_critic.py
        
        self.display_freq = 4000
        self.save_freq = 100000
        self.save_dir = './checkpoints/'

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        self.envs = env
        if self.envs == None:
            self.envs = make_vec_envs('SuperMarioBros-v0', self.seed,
                    self.n_processes)
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        self.obs_shape = self.envs.observation_space.shape
        self.act_shape = self.envs.action_space.n

        self.rollouts = RolloutStorage(self.update_freq, self.n_processes,
                                       self.obs_shape, self.act_shape, self.hidden_size)
        self.model = ActorCritic(self.obs_shape, self.act_shape, self.hidden_size, self.recurrent).to(self.device)
        self.optimizer = RMSprop(self.model.parameters(), lr=self.lr, eps=1e-5)

        self.hidden = None
        self.init_game_setting()
   
    def _update(self):
        # TODO: Compute returns
        # R_t = reward_t + gamma * R_{t+1}

        batch_returns = []
        for i in range(self.rollouts.rewards.size(0)):      # rollouts.rewards: (5, 16, 1)
            R, policy_loss, returns = 0, [], []
            for r in self.rollouts.rewards[i].view(-1).tolist()[::-1]:
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / returns.std()
            batch_returns.append(returns)   # (5, 16)

        # TODO:
        # Compute actor critic loss (value_loss, action_loss)
        # OPTIONAL: You can also maxmize entropy to encourage exploration
        # loss = value_loss + action_loss (- entropy_weight * entropy)

        action_losses = []
        value_loss = []
        for i in range(self.rollouts.obs.size(0) - 1):
            loss = []

            values, action_probs, hiddens = self.model(
                self.rollouts.obs[i], self.rollouts.hiddens[i], self.rollouts.masks[i])
            # action_probs: (16, 12)
            # values: (16, 1)
            # self.rollouts.value_preds[0]: (16, 1)
            # self.rollouts.rewards[0]: (16, 1)

            values2, action_probs2, hiddens2 = self.model(
                self.rollouts.obs[i+1], self.rollouts.hiddens[i+1], self.rollouts.masks[i+1])

            values = values.view(1, -1)
            next_state_values = (values2 * self.gamma + self.rollouts.rewards[i]).view(1, -1)

            value_loss.append(F.smooth_l1_loss(values, next_state_values).unsqueeze(0))
            # loss.append(value_loss.unsqueeze(0))

            action_log_probs = action_probs.gather(1, self.rollouts.actions[i])

            for log_prob, reward in zip(action_log_probs, batch_returns[i]):
                loss.append(-log_prob * reward)
            loss = torch.cat(loss).sum()
            action_losses.append(loss.unsqueeze(0))
        entropy = Categorical(probs=torch.cat(action_losses)).entropy()     # tensor(-4.2006)
        # ipdb.set_trace()
        loss = action_losses + value_loss
        loss = torch.cat(loss).sum() - self.entropy_weight * entropy
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        
        # TODO:
        # Clear rollouts after update (RolloutStorage.reset())
        self.rollouts.reset()

        return loss.item()

    def _step(self, obs, hiddens, masks):
        # obs: [16, 4, 84, 84],  hiddens: [16, 512], masks: [16, 1]
        with torch.no_grad():

            values, action_probs, hiddens = self.model(obs, hiddens, masks)
            # TODO:
            # Sample actions from the output distributions
            # HINT: you can use torch.distributions.Categorical
            # values: (16, 1),  action_probs: (16, 12),  hiddens: (16, 512)

            action_distributions = Categorical(action_probs)    # (16, 12)
            actions = action_distributions.sample()             # (16)
        # ipdb.set_trace()
        obs, rewards, dones, infos = self.envs.step(actions.cpu().numpy())
        # TODO:
        #                    np   tensor   tensor   tensor  np       tensor
        # Store transitions (obs, hiddens, actions, values, rewards, masks)
        # You need to convert arrays to tensors first
        # HINT: masks = (1 - dones)

        obs = torch.from_numpy(obs)
        rewards = torch.from_numpy(rewards).unsqueeze(1)
        masks = torch.from_numpy(1 - dones).unsqueeze(1)
        self.rollouts.insert(obs, hiddens, actions.unsqueeze(1), values, rewards, masks)
        self.rollouts.to(self.device)
        
    def train(self):

        plot_steps = []
        plot_rewards = []

        print('Start training')
        running_reward = deque(maxlen=10)
        episode_rewards = torch.zeros(self.n_processes, 1).to(self.device)
        total_steps = 0
        
        # Store first observation
        obs = torch.from_numpy(self.envs.reset()).to(self.device)
        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device)
        
        while True:
            # Update once every n-steps
            for step in range(self.update_freq):
                self._step(
                    self.rollouts.obs[step],
                    self.rollouts.hiddens[step],
                    self.rollouts.masks[step])

                # Calculate episode rewards
                episode_rewards += self.rollouts.rewards[step]
                for r, m in zip(episode_rewards, self.rollouts.masks[step + 1]):
                    if m == 0:
                        running_reward.append(r.item())
                episode_rewards *= self.rollouts.masks[step + 1]

            loss = self._update()
            total_steps += self.update_freq * self.n_processes

            # Log & save model
            if len(running_reward) == 0:
                avg_reward = 0
            else:
                avg_reward = sum(running_reward) / len(running_reward)

            if total_steps % self.display_freq == 0:
                print('Steps: %d/%d | Avg reward: %f'%
                        (total_steps, self.max_steps, avg_reward))
                plot_steps.append(total_steps)
                plot_rewards.append(avg_reward)
            
            if total_steps % self.save_freq == 0:
                self.save_model('model.pt')
            
            if total_steps >= self.max_steps:
                break

        plt.title('learning curve - a2c')
        plt.xlabel('Timesteps')
        plt.ylabel('Avg rewards')
        plt.plot(plot_steps, plot_rewards)
        plt.savefig('a2c_learning_curve.png')


    def save_model(self, filename):
        torch.save(self.model, os.path.join(self.save_dir, filename))

    def load_model(self, path):
        self.model = torch.load(path)

    def init_game_setting(self):
        if self.recurrent:
            self.hidden = torch.zeros(1, self.hidden_size).to(self.device)

    def make_action(self, observation, test=False):
        # TODO: Use you model to choose an action
        return action
