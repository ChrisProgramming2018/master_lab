import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from replayBuffer import ReplayBuffer
import argparse
import numpy as np
import datetime
import wandb
import torch
from vit_pytorch import ViT, MAE

def main(args, wandb):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    latent_dims = args.latent_dims
    num_epochs = args.updates
    batch_size = args.batch_size
    capacity = args.capacity
    learning_rate = args.lr
    variational_beta = 1
    use_gpu = True
    """
    replay_buffer = ReplayBuffer((4, args.size, args.size), (args.action_shape, ), args.buffer_size + 1, args.device)
    replay_buffer.load_memory(args.bufferpath_train)
    
    replay_buffer_eval = ReplayBuffer((4, args.size, args.size), (args.action_shape, ), args.buffer_size_eval + 1, args.device)
    replay_buffer_eval.load_memory(args.bufferpath_eval)
    """
    from transformers import ViTFeatureExtractor, ViTForImageClassification
    from PIL import Image
    import requests
    from PIL import Image
    from torchvision.transforms import ToTensor

    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    
    img_path="/home/programmer/master_1/lab/transformer_hugging_face/coco_images/test2017/*"
    from os import listdir
    from os.path import isfile, join
    import glob
    names = []
    for idx, files in enumerate(glob.glob(img_path)):
        #print(files)
        names.append(files)
        if idx > 32:
            break



    batch = []
    for n in names:
        #print(n)
        image = Image.open(n)
        #import pdb; pdb.set_trace()
        #print(image)
        if image.mode == "RGB":
            inputs = feature_extractor(images=image, return_tensors="pt")
            data = inputs["pixel_values"].to("cuda")
            #print(data.shape)
            batch.append(data.squeeze(0))
    print(len(batch))
    batch = torch.stack(batch)
    print(batch.shape)
    v = ViT(
            image_size = 224,
            patch_size = 32,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 8,
            mlp_dim = 2048
            )
    mae = MAE(
            encoder = v,
            masking_ratio = 0.75,   # the paper recommended 75% masked patches
            decoder_dim = 512,      # paper showed good results with just 512
            decoder_depth = 6       # anywhere from 1 to 8
            ).to("cuda")
    train_loss_avg = []
    for progress in range(10000):
        #sys.stdout.write("Download progress: %d%%   \r" % (progress) )
        #sys.stdout.flush()
        # import pdb; pdb.set_trace()
        loss = mae(batch)
        loss.backward()
        train_loss_avg.append(loss.cpu().detach().numpy())
        if progress % 100 == 0:
            print("at {}av loss {} ".format(progress, np.mean(train_loss_avg)))
    
    sys.exit()
    torch.save(v.state_dict(), './trained-vit.pt')

    train_loss_avg = []
    dt = datetime.datetime.today().strftime("%Y-%m-%d")
    now = datetime.datetime.now()
    time = str(now.hour) +"-" + str(now.minute) + "-" + str(now.second)
    print(time)
    experiment_path = os.path.join(os.path.expanduser('~'), "exeriments/vae", args.run_name, dt, time)
    print("create experiment path {}".format(experiment_path))
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    else:
        print("folder {} already exists".format(experiment_path))
    print('Training ...')
    for epoch in range(num_epochs):
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
            # log_image(wandb, replay_buffer_eval.obses[0], vae, epoch)
            if args.track:
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
            if args.track:
                wandb.log({"eval loss": eval_loss})
                log_image(wandb, replay_buffer_eval.obses[0], vae, epoch)
            print('Epoch [%d / %d] eval reconstruction error: %f' % (epoch+1, num_epochs, eval_loss ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='image witdh and hight')
    parser.add_argument('--capacity', type=int, default=64, help='image witdh and hight')
    parser.add_argument('--latent_dims', type=int, default=2, help='image witdh and hight')
    parser.add_argument('--lr', type=float, default=1e-3, help='image witdh and hight')
    parser.add_argument('--bufferpath_train', type=str, default="", help='the name of buffer path')
    parser.add_argument('--bufferpath_eval', type=str, default="", help='the name of buffer path')
    parser.add_argument('--device', type=str, default="cuda", help='device name')
    parser.add_argument('--size', type=int, default=84, help='image witdh and hight')
    parser.add_argument('--action_shape', type=int, default=1, help='image witdh and hight')
    parser.add_argument('--buffer_size', type=int, default=20000, help='amount of samples buffer can store')
    parser.add_argument('--buffer_size_eval', type=int, default=2000, help='amount of samples buffer can store')
    parser.add_argument('--updates', type=int, default=20000, help='amount of samples buffer can store')
    parser.add_argument('--track', type=bool, default=False, help='activate wandb')
    parser.add_argument('--run_name', type=str, default="test", help='run name')
    parser.add_argument('--seed', type=int, default=1, help='seed ')
    args = parser.parse_args()
    if args.track:
        wandb.init(
                project="master_lab",
                sync_tensorboard=True,
                config=vars(args),
                name=args.run_name,
                monitor_gym=True,
                save_code=True,
                )
    main(args, wandb)
