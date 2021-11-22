import gym
import argparse
import sys
import random
import time
import cv2
from replayBuffer import ReplayBuffer

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--gym-id', type=str, default="SpaceInvaders-v0",
        help='the id of the gym environment')
    parser.add_argument('--filename', type=str, default="train-buffer",
        help='file name of replaybuffer ')
    parser.add_argument('--buffer_size', type=int, default=10000,
        help='size of saved buffer')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    size = 256
    #env = make_env(args.gym_id, args.seed , 0, True, run_name)()
    replay_buffer = ReplayBuffer((size, size, 3), args.buffer_size + 1, "cuda") 
    env = gym.make(args.gym_id)
    while True:
        episode_steps = 0
        episode_reward = 0
        state = env.reset()
        replay_buffer.add_state(cv2.resize(state, (256, 256)))
        while True:
            episode_steps += 1
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            replay_buffer.add_state(cv2.resize(state, (256, 256)))
            episode_reward += reward
            if replay_buffer.idx >= args.buffer_size:
                print("Save bufffer {} of size {}".format(args.filename, replay_buffer.idx))
                replay_buffer.save_memory(args.filename)
                sys.exit()
            if done:
                print("Buffer size {}".format(replay_buffer.idx))
                print("End episode with reward {} and steps {}".format(episode_reward, episode_steps))
                break
