from model import VAE
from utils import get_dataset, plot_input_output
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
from utils import plot_generated_image
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = 2
model = VAE(latent_dim)

model.load_state_dict(torch.load('vae_model.pth'))
model.eval()
model.to(device)

dataset = get_dataset('vae_mri')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


