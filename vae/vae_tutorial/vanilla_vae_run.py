from vanilla_vae import VanillaVAE
from utils import get_dataset
import wandb
from torch.utils.data import DataLoader
import torch

#---------- WANDB CONFIG ----------
wandb.init(project="OSASIS_VAE_rangpur", name="VAE")
wandb.config.update({"architecture": "VAE", "dataset": "OASIS_mri", "epochs": 20, 
                     "batch_size": 32, "weight_decay": 5e-4, "max_lr": 0.1, "grad_clip": 1.5})


""" # ---------- Device configuration ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device) """


dataset = get_dataset('vae_mri')
dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)



model = VanillaVAE(in_channels=3, latent_dim=128)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(wandb.config.epochs):
    for batch_idx, data in enumerate(dataloader):
        print(data[0].shape, data[1].shape)
        print(data[1])


        [z, input, mu, log_var] = model.forward(data[0])
        