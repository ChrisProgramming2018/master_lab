from taskonomy_network import TaskonomyEncoder, TaskonomyDecoder, TaskonomyNetwork,  TASKS_TO_CHANNELS
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from replayBuffer import ReplayBuffer
import cv2
import numpy as np
import argparse



def main(args):
    """ """
    size = 256
    memory_small = ReplayBuffer((size,size,3), 201, "cuda")
    memory_small.load_memory("valid-buffer")
    model = TaskonomyNetwork()
    model.load_model(args.model_path) 
    print("model loaded")
    model.to(args.device)
    RGB_state = memory_small.obses[42]
    cv2.imshow("RGB_image", RGB_state)
    cv2.waitKey(0)
    RGB_state = TF.to_tensor(RGB_state).to(args.device)
    RGB_state = RGB_state.unsqueeze(0)
    print("shape", RGB_state.shape)
    outPut = model(RGB_state).cpu().numpy()
    print(outPut.shape)
    outPut = np.array(outPut.squeeze(0))
    print(outPut.shape)
    print(outPut)
    outPut = outPut.transpose(1,2,0)
    print(outPut.shape)
    cv2.imshow("RGB_image", outPut * 255)
    cv2.waitKey(0)

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="",
        help='path to trained predictor')
    parser.add_argument('--buffer_path', type=str, default="train-buffer",
        help='file name of replaybuffer ')
    parser.add_argument('--buffer_size', type=int, default=1000,
        help='size of saved buffer')
    parser.add_argument('--device', type=str, default="cuda",
        help='select used device cpu or cuda')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
