import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
from gym import wrappers
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="SpaceInvaders-v0",
        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=10000000,
        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will be enabled by default')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='weather to capture videos of the agent performances (check out `videos` folder)')

    # Algorithm specific arguments
    parser.add_argument('--num-envs', type=int, default=8,
        help='the number of parallel game environments')
    parser.add_argument('--num-steps', type=int, default=128,
        help='the number of steps to run in each environment per policy rollout')
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Use GAE for advantage computation')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
        help='the lambda for the general advantage estimation')
    parser.add_argument('--num-minibatches', type=int, default=4,
        help='the number of mini-batches')
    parser.add_argument('--update-epochs', type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help="Toggles advantages normalization")
    parser.add_argument('--clip-coef', type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')
    parser.add_argument('--ent-coef', type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
        help='the maximum norm for the gradient clipping')
    parser.add_argument('--target-kl', type=float, default=None,
        help='the target KL divergence threshold')
    
    parser.add_argument('--filename', type=str, default=None,
        help="path to expert policy weights")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        #env = NoopResetEnv(env, noop_max=30)
        #env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        #env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def create_buffer(args, agent, replay_buffer):
    """ """
    run_name = f"{args.gym_id}__{args.exp_name}__{0}__{int(steps)}"
    env = make_env(args.gym_id, 0 , 0, False, run_name)()
    old_buffer_size = 0
    while True:
        obs = torch.Tensor(env.reset()).to(device).unsqueeze(0)
        rewards = 0
        steps = 0
        while True:
            steps += 1
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs)
            action = action.cpu().numpy()[0]
            obs, reward, done, info = env.step(action)
            replay_buffer.add(obs, action, reward, done)
            obs = torch.Tensor(obs).to(device).unsqueeze(0)
            rewards += reward
            if replay_buffer.idx >= args.buffer_size:
                print("Save Buffer with size of {}".format(replay_buffer.idx))
                replay_buffer.
            if done:
                if rewards > 1000:
                    print("Add traj to buffer with a size of {} ".format(replay_buffer.idx))
                else:
                    print("to low reward do not add")
                    old_buffer_size = replay_buffer.idx
            average_return.append(rewards)
                print("Reward {} steps {}".format(rewards, steps))
                break



def eval_policy(args, agent, steps):
    runs = 10
    average_return = []
    run_name = f"{args.gym_id}__{args.exp_name}__{0}__{int(steps)}"
    env = make_env(args.gym_id, 0 , 0, True, run_name)()
    for i in range(runs):
        obs = torch.Tensor(env.reset()).to(device).unsqueeze(0)
        rewards = 0
        steps = 0
        while True:
            steps += 1
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs)
            action = action.cpu().numpy()[0]
            obs, reward, done, info = env.step(action)
            obs = torch.Tensor(obs).to(device).unsqueeze(0)
            rewards += reward
            if done:
                average_return.append(rewards)
                print("Reward {} steps {}".format(rewards, steps))
                break
    print("Average return {} over {} runs".format(np.mean(average_return), runs))

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    replay_buffer = ReplayBuffer((size, size, 3), args.buffer_size + 1, device)
    envs = gym.vector.SyncVectorEnv(
            [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
            )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    agent = Agent(envs).to(device)
    agent.critic.load_state_dict(torch.load(args.filename + "_critic"))
    agent.actor.load_state_dict(torch.load(args.filename + "_actor"))
    agent.network.load_state_dict(torch.load(args.filename + "_network"))
    create_buffer(args, agent, replay_buffer) 
