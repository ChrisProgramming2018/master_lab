import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from replayBuffer import ReplayBuffer
from omegaconf import OmegaConf
import hydra
import numpy as np
from vae_model import VariationalAutoencoder, vae_loss
import datetime
import wandb



@hydra.main(config_path="config", config_name="stable_train")    
def main(cfg):
    bufferpath_train = "/home/leiningc/master_lab/vae/expert_data_space_invadars_average_reward_1086.25_size_10000"
    bufferpath_eval = "/home/leiningc/master_lab/vae/expert_data_space_invadars_average_reward_nan_size_2000"
    buffer_size = 10000
    buffer_size_eval = 2000
    device = "cuda"
    seed = 0
    size = 84
    action_shape = 4
    run_name = cfg["run_name"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    latent_dims = int(cfg["latent_dims"])
    num_epochs = int(cfg["updates"])
    batch_size = int(cfg["batch_size"])
    capacity = int(cfg["capacity"])
    learning_rate = float(cfg["lr"])
    variational_beta = 1
    if "wandb" in cfg:
        wandb.init(
                project="master_lab",
                sync_tensorboard=True,
                config=OmegaConf.to_container(cfg, resolve=True),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
    vae = VariationalAutoencoder(capacity, latent_dims).to(device)
    replay_buffer = ReplayBuffer((4, size, size), (action_shape, ), buffer_size + 1, device)
    replay_buffer.load_memory(bufferpath_train)
    
    replay_buffer_eval = ReplayBuffer((4, size, size), (action_shape, ), buffer_size_eval + 1, device)
    replay_buffer_eval.load_memory(bufferpath_eval)
    
    optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=1e-5)
    train_loss_avg = []
    dt = datetime.datetime.today().strftime("%Y-%m-%d")
    now = datetime.datetime.now()
    time = str(now.hour) +"-" + str(now.minute) + "-" + str(now.second)
    print(time)
    experiment_path = os.path.join(os.path.expanduser('~'), "exeriments/vae", run_name, dt, time)
    print("create experiment path {}".format(experiment_path))
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    else:
        print("folder {} already exists".format(experiment_path))
    print('Training ...')
    for epoch in range(1, num_epochs + 1):
        obses, next_obses, actions, rewards, dones = replay_buffer.sample(batch_size)
        img = []
        for i in range(batch_size):
            for j in range(4):
                img.append(obses[i][j].unsqueeze(0))
        
        image_batch = torch.stack(img)

        image_batch_recon, latent_mu, latent_logvar = vae(image_batch)
        # reconstruction error
        loss = vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()
        loss_train = loss.item()
        train_loss_avg.append(loss_train)
        if epoch % 100 == 0:
            train_loss_mean = np.mean(train_loss_avg)
            print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, num_epochs, loss_train))
            #filename = experiment_path + "/model_at_{}-loss{:.2f}".format(epoch, loss_train)
            #vae.save_var(filename)
            if "wandb" in cfg:
                wandb.log({"train loss": loss_train, "train loss mean" : train_loss_mean })
        if epoch % 250 == 0 or False:
            vae.eval()
            obses_eval, next_obses, actions, rewards, dones = replay_buffer_eval.sample(batch_size)
            img_eval = []
            for i in range(batch_size):
                for j in range(4):
                    img_eval.append(obses_eval[i][j].unsqueeze(0))
            image_eval = torch.stack(img_eval)
            image_eval_recon, latent_mu_eval, latent_logvar_eval = vae(image_eval.detach())
            vae.train()
            loss_eval = vae_loss(image_eval_recon, image_eval, latent_mu_eval, latent_logvar_eval)
            eval_loss = loss_eval.item()
            if  "wandb" in cfg:
                wandb.log({"eval loss": eval_loss})
                log_image(wandb, replay_buffer_eval.obses[0], vae, epoch)
            print('Epoch [%d / %d] eval reconstruction error: %f' % (epoch+1, num_epochs, eval_loss ))


def log_image(w, images, vae, steps):
    device = "cuda"
    img = images[0]
    img = torch.as_tensor(img, device=device).float()
    img = img.unsqueeze(0)
    img = img.unsqueeze(0)
    vae.eval()
    image_recon, latent_mu, latent_logvar = vae(img.detach())
    vae.train()
    recon_image = image_recon.detach().cpu().squeeze(0).squeeze(0).numpy() * 255
    img = img.detach().cpu().squeeze(0).squeeze(0).numpy()
    verti = np.concatenate((img, recon_image), axis=0).astype(int)
    # verti = np.expand_dims(verti,0)
    # print(verti.shape)
    # print(verti.dtype)
    #cv2.imshow('VERTICAL', verti)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    path = "images"
    name= os.path.join(path, "image_step-{}.jpg".format(steps))
    cv2.imwrite(name, verti)
    w.log({"recon image": wandb.Image(verti)})
    return vae

if __name__ == "__main__":
    main()
