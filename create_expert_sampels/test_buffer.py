import argparse
from replayBuffer import ReplayBuffer



def main(args):
    replay_buffer = ReplayBuffer((4, args.size, args.size), args.action_shape, args.buffer_size + 1, args.device)
    replay_buffer.load_memory(args.bufferpath)
    obses, next_obses, actions, rewards, dones = replay_buffer.sample(32)
    print("state shape ", obses.shape)
    print("actions shape ", actions.shape)
    print("done shape ", dones.shape)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bufferpath', type=str, default="", help='the name of buffer path')
    parser.add_argument('--device', type=str, default="cuda", help='device name')
    parser.add_argument('--size', type=int, default=84, help='image witdh and hight')
    parser.add_argument('--action_shape', type=int, default=1, help='image witdh and hight')
    parser.add_argument('--buffer_size', type=int, default=20000, help='amount of samples buffer can store')
    args = parser.parse_args()
    main(args)


