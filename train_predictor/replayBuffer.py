import os
import cv2
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, capacity, device):
        self.capacity = capacity
        self.device = device
        self.obs_shape = obs_shape
        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs):
        np.copyto(self.obses[self.idx], obs.astype(np.uint8))
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)    
        obses = self.obses[idxs]
        obses = torch.as_tensor(obses, device=self.device).float()
        obses = torch.reshape(obses,(batch_size, 3, 256, 256))
        return obses
    
    def save_memory(self, filename):
        """
        Use numpy save function to store the data in a given file
        """
        if not os.path.exists(filename):
            os.makedirs(filename)
        
        with open(filename + '/obses.npy', 'wb') as f:
            np.save(f, self.obses) 
        
        with open(filename + '/index.txt', 'w') as f:
            f.write("{}".format(self.idx))
        
        print("save buffer to {}".format(filename))
    
    def load_memory(self, filename):
        """
        Use numpy load function to store the data in a given file
        """
        
        with open(filename + '/obses.npy', 'rb') as f:
            self.obses = np.load(f)
  
        with open(filename + '/index.txt', 'r') as f:
            self.idx = int(f.read())
