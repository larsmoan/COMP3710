import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from utils import get_dataset
from models import VAE, vae_loss

wandb.init(project="OSASIS_VAE_rangpur", name="VAE")
wandb.config.update({"architecture": "VAE", "dataset": "OASIS_mri", "epochs": 20, 
                     "batch_size": 32, "weight_decay": 5e-4, "max_lr": 0.1, "grad_clip": 1.5})

# ---------- Device configuration ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)



# Load and preprocess the dataset
dataset = get_dataset('vae_mri')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the VAE model
latent_dim = 128
vae = VAE(latent_dim)
vae.to(device)

# Define optimizer
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# Training loop
vae.train()
for epoch in range(wandb.config.epochs):
    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()
        data = data[0].to(device)
        recon_batch, mu, logvar = vae(data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()
        learning_rate = optimizer.param_groups[0]['lr']
        wandb.log({"Epoch": epoch, "Learning_rate": learning_rate, "Loss": loss.item(), "Bathc": {batch_idx+1}})  

