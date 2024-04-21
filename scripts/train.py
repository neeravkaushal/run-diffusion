import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from datasets import MnistDataset
from torch.utils.data import DataLoader
from model import Unet
from noise import NoiseDealer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    #--Read params from the config file
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    #--Instantiate noise dealer
    scheduler = NoiseDealer(beta_start=diffusion_config['beta_start'],
                            beta_end=diffusion_config['beta_end'],
                            num_timesteps=diffusion_config['num_timesteps'])
    
    #--Create dataset
    mnist = MnistDataset(split='train',
                         im_path=dataset_config['im_path'],
                         num_train_samples=dataset_config['num_train_samples'])
    mnist_loader = DataLoader(mnist,
                              batch_size=train_config['batch_size'],
                              shuffle=True,
                              num_workers=4)
    
    #--Instantiate model and move to device
    model = Unet(model_config).to(device)
    model.train()
    
    #--Create output dirs
    if not os.path.exists(train_config['task_name']): os.mkdir(train_config['task_name'])
    
    #--Load checkpoint if found
    if os.path.exists(os.path.join(train_config['task_name'],train_config['ckpt_name'])):
        print('Loading checkpoint as found one')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['ckpt_name']),
                                                      map_location=device))
        
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()
    
    for epoch_idx in range(train_config['num_epochs']):
        losses = []

        for im in tqdm(mnist_loader):
            optimizer.zero_grad()
            
            im    = im.float().to(device) #--Read data
            noise = torch.randn_like(im).to(device) #--Sample noise
            t     = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device) #--choose random timestep
            noisy_im   = scheduler.add_noise(im, noise, t) #--Add noise for that timestep to image
            noise_pred = model(noisy_im, t) #--Get model prediction
            loss       = criterion(noise_pred, noise) #--Calculate loss

            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        print('Finished epoch:{} | Loss : {:.4f}'.format(epoch_idx + 1, np.mean(losses),))
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['ckpt_name']))
    print('Training completed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config',
                        dest='config_path',
                        default='model_config.yaml',
                        type=str)
    args = parser.parse_args()
    train(args)