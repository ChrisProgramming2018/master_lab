import os
import cv2
import sys
import torch
import time
import argparse
import numpy as np
import torch.optim as optim
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from collections import deque
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from taskonomy_network import TaskonomyEncoder, TaskonomyDecoder, TaskonomyNetwork,  TASKS_TO_CHANNELS
#from models import VisualPrior, VisualPriorRepresentation
from PIL import Image
from replayBuffer import ReplayBuffer

def time_format(sec):
    """
    Args:
    param1():
    """
    hours = sec // 3600
    rem = sec - hours * 3600
    mins = rem // 60
    secs = rem - mins * 60
    return hours, mins, round(secs,2)

def main(args):
    """ """
    path_img = args.path
    t0 = time.time()
    TASKONOMY_PRETRAINED_WEIGHT_FILES= ["autoencoding_decoder-a4a006b5a8b314b9b0ae815c12cf80e4c5f2e6c703abdf65a64a020d3fef7941.pth", "autoencoding_encoder-e35146c09253720e97c0a7f8ee4e896ac931f5faa1449df003d81e6089ac6307.pth"]
    path_de = 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/' + str(TASKONOMY_PRETRAINED_WEIGHT_FILES[0])
    path_en = 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/' + str(TASKONOMY_PRETRAINED_WEIGHT_FILES[1])
    model = TaskonomyNetwork(load_encoder_path=path_en, load_decoder_path=path_de)
    model_path = "trained_model"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save_model(model_path + "/model-{}".format(0))
    model.encoder.eval_only = False
    model.decoder.eval_only = False
    
    for param in model.parameters():
        param.requires_grad = True
        
    model.to(args.device)
    model.train()
    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    now = datetime.now()    
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    pathname = "var_model"
    pathname += dt_string
    tensorboard_name = 'runs/' + pathname
    writer = SummaryWriter(tensorboard_name)
    size = 256
    memory = ReplayBuffer((size, size, 3), 15001, "cuda")
    memory_valid = ReplayBuffer((size, size, 3), 2001, "cuda")
    path = "train-buffer"
    print("Load buffer ...")
    memory.load_memory(path)
    print("... buffer size {} loaded".format(memory.idx))
    
    path = "valid-buffer"
    print("Load valid buffer ...")
    memory_valid.load_memory(path)
    print("... valid buffer {} loaded".format(memory_valid.idx))
    
    print("buffer size train ", memory.idx)
    print("buffer size valid ", memory_valid.idx)
    
    torch.cuda.empty_cache()
    batch_size = 48
    scores_window = deque(maxlen=100) 
    epochs = int(100e4)
    for epoch in range(epochs):
        print('\rEpisode {}'.format(epoch), end="")
        rgb_batch = memory.sample(batch_size)
        x_recon = model(rgb_batch.to(args.device))
        
        optimizer.zero_grad()
        loss = F.mse_loss(x_recon, rgb_batch.to(args.device))
        loss.backward()
        optimizer.step()
        scores_window.append(loss.item())
        mean_loss = np.mean(scores_window)
        writer.add_scalar('loss', loss.item(), epoch)  
        writer.add_scalar('mean_loss', mean_loss, epoch)  
    
        if epoch % 75 == 0:
            model.eval()
            eval_loss = 0
            evaL_size = 25
            for i in range(evaL_size):
                rgb_batch = memory_valid.sample(batch_size)
                x_recon = model(rgb_batch.to(args.device)).detach()
                loss = F.mse_loss(x_recon, rgb_batch.to(args.device))
                eval_loss +=loss

            model.train()
            text = "Eval model {} eval loss {:10f} time {}  \r".format(epoch, eval_loss, time_format(time.time() - t0))
            writer.add_scalar('eval_loss', eval_loss, epoch)  
            print("  ")
            print(text)
        if epoch % 5 == 0:
            text = "Epochs {}  loss {:.5f}  ave loss {:.5f}  time {}  \r".format(epoch, loss, mean_loss, time_format(time.time() - t0))
            print("  ")
            print(text)
            img = memory_valid.obses[0]
            # import pdb; pdb.set_trace()
            im = Image.fromarray(img)
            if not os.path.exists(path_img):
                os.makedirs(path_img)
            im.save(path_img + "/orginal-{}.jpeg".format(epoch))
            print("im ", img.shape)
            obs = torch.as_tensor(img, device=args.device).float().unsqueeze(0)
            obs = torch.reshape(obs,(1, 3, 256, 256))
            #import pdb; pdb.set_trace()
            x_recon_s = model(obs.to(args.device)).detach().squeeze(0) * 255
            x_recon_s = x_recon_s.type(torch.int).cpu().numpy()
            #import pdb; pdb.set_trace()
            x_recon_s = np.moveaxis(x_recon_s, 0, -1)
            x_recon_s = x_recon_s.astype(np.uint8)
            # import pdb; pdb.set_trace()
            re = Image.fromarray(x_recon_s)
            re.save(path_img + "/reconst-{}.jpeg".format(epoch))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="test_buffer",
            help='file name of replaybuffer ')
    parser.add_argument('--buffer_size', type=int, default=1000,
            help='size of saved buffer')
    parser.add_argument('--device', type=str, default="cpu",
            help='select device')
    parser.add_argument('--path', type=str, default="test",
            help='path to save experiment data')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
