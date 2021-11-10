import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from replayBuffer import ReplayBuffer
import argparse
import numpy as np
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self, capacity, latent_dims):
        super(Encoder, self).__init__()
        c = capacity
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1) # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: c x 7 x 7
        self.fc_mu = nn.Linear(in_features=c*2*21*21, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=c*2*21*21, out_features=latent_dims)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self, capacity, latent_dims):
        super(Decoder, self).__init__()
        c = capacity
        self.capacity = capacity
        self.fc = nn.Linear(in_features=latent_dims, out_features=c*2*21*21)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)
            
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.capacity*2, 21, 21) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, capacity, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(capacity, latent_dims)
        self.decoder = Decoder(capacity, latent_dims)
    
    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    def save_var(self, filename):
        """ """
        torch.save(self.encoder.state_dict(), filename + "_encoder")
        torch.save(self.decoder.state_dict(), filename + "_decoder")

    def load_var(self, filename):
        self.encoder.load_state_dict(torch.load(filename + "_encoder"))
        self.decoder.load_state_dict(torch.load(filename + "_decoder"))
        print("vae load weights from {}".format(filename))


def vae_loss(recon_x, x, mu, logvar, variational_beta=1):
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # Averaging or not averaging the binary cross-entropy over all pixels here
    # is a subtle detail with big effect on training, since it changes the weight
    # we need to pick for the other loss term by several orders of magnitude.
    # Not averaging is the direct implementation of the negative log likelihood,
    # but averaging makes the weight of the other loss term independent of the image resolution.
    recon_loss = F.mse_loss(recon_x, x) 
    # KL-divergence between the prior distribution over latent vectors
    # (the one we are going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image.
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + variational_beta * kldivergence
    
def main(args):
    latent_dims = 2
    num_epochs = args.updates
    batch_size = 32
    capacity = 64
    learning_rate = 1e-3
    variational_beta = 1
    vae = VariationalAutoencoder(capacity, latent_dims).to(args.device)
    replay_buffer = ReplayBuffer((4, args.size, args.size), (args.action_shape, ), args.buffer_size + 1, args.device)
    replay_buffer.load_memory(args.bufferpath)
    vae.eval()
    vae.load_var(args.filename)
    images = replay_buffer.obses[0]
    img = images[0]
    print(img.shape)
    #plt.imshow((img))
    #plt.show()
    img = torch.as_tensor(img, device=args.device).float()
    img = img.unsqueeze(0)
    img = img.unsqueeze(0)
    print(img.shape)
    image_recon, latent_mu, latent_logvar = vae(img)
    plt.imshow((image_recon.detach().cpu().squeeze(0).squeeze(0)))
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bufferpath', type=str, default="", help='the name of buffer path')
    parser.add_argument('--filename', type=str, default="", help='the name of vae weights path')
    parser.add_argument('--device', type=str, default="cuda", help='device name')
    parser.add_argument('--size', type=int, default=84, help='image witdh and hight')
    parser.add_argument('--action_shape', type=int, default=1, help='image witdh and hight')
    parser.add_argument('--buffer_size', type=int, default=20000, help='amount of samples buffer can store')
    parser.add_argument('--updates', type=int, default=20000, help='amount of samples buffer can store')
    args = parser.parse_args()
    main(args)
