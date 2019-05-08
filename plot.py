"""

### NOTICE ###
You DO NOT need to upload this file

"""
import argparse
from test import test
from environment import Environment
from matplotlib import pyplot as plt


def parse():
    parser = argparse.ArgumentParser(description="MLDS&ADL HW3")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_pg', action='store_true', help='whether train policy gradient')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--train_dqn_exp', action='store_true', help='whether train DQN EXP')
    parser.add_argument('--dqn_exp', default=0, type=int, help='whether train DQN EXP')
    parser.add_argument('--train_mario', action='store_true', help='whether train mario')
    parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--test_mario', action='store_true', help='whether test mario')
    parser.add_argument('--video_dir', default=None, help='output video directory')
    parser.add_argument('--do_render', action='store_true', help='whether render environment')

    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def run(args):
    if args.train_pg:
        env_name = args.env_name or 'LunarLander-v2'
        env = Environment(env_name, args, atari_wrapper=False)
        from agent_dir.agent_pg import AgentPG
        agent = AgentPG(env, args)
        plot_epoch, plot_avg_reward = agent.train()

        plt.title('learning curve - pg')
        plt.xlabel('Epochs')
        plt.ylabel('Avg rewards in last 10 epochs')
        plt.plot(plot_epoch, plot_avg_reward)
        plt.savefig('pg_learning_curve.png')

    if args.train_dqn:
        env_name = args.env_name or 'AssaultNoFrameskip-v0'
        env = Environment(env_name, args, atari_wrapper=True)
        from agent_dir.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        plot_timesteps, plot_avg_reward = agent.train()

        plt.title('learning curve - dqn')
        plt.xlabel('timesteps')
        plt.ylabel('Avg rewards in last 10 episodes')
        plt.plot(plot_timesteps, plot_avg_reward)
        plt.savefig('dqn_learning_curve.png')

    if args.train_dqn_exp:
        env_name = args.env_name or 'AssaultNoFrameskip-v0'
        env = Environment(env_name, args, atari_wrapper=True, test=True)
        from agent_dir.agent_dqn import AgentDQN

        args.dqn_exp = 0
        agent = AgentDQN(env, args)
        plot_timesteps0, plot_avg_reward0 = agent.train()
        args.dqn_exp = 1
        agent = AgentDQN(env, args)
        plot_timesteps1, plot_avg_reward1 = agent.train()
        args.dqn_exp = 2
        agent = AgentDQN(env, args)
        plot_timesteps2, plot_avg_reward2 = agent.train()
        args.dqn_exp = 3
        agent = AgentDQN(env, args)
        plot_timesteps3, plot_avg_reward3 = agent.train()

        plt.figure(figsize=(10, 10))
        plt.suptitle('learning curve - dqn')
        plt.xlabel('timesteps')
        plt.ylabel('Avg rewards in last 10 episodes')

        plot1 = plt.subplot(2, 2, 1)
        plot1.set_title('threshold 0')
        plot1.plot(plot_timesteps0, plot_avg_reward0)

        plot2 = plt.subplot(2, 2, 2)
        plot2.set_title('threshold 0.05')
        plot2.plot(plot_timesteps1, plot_avg_reward1)
        #
        plot3 = plt.subplot(2, 2, 3)
        plot3.set_title('threshold 0.05 to 0')
        plot3.plot(plot_timesteps2, plot_avg_reward2)

        plot4 = plt.subplot(2, 2, 4)
        plot4.set_title('threshold 0.9 to 0.05')
        plot4.plot(plot_timesteps3, plot_avg_reward3)

        plt.savefig('dqn_experiment.png')

    if args.train_mario:
        from agent_dir.agent_mario import AgentMario
        agent = AgentMario(None, args)
        agent.train()


if __name__ == '__main__':
    args = parse()
    run(args)
