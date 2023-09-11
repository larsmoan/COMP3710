import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import gdown
import zipfile
import os


def get_dataset(folder_name: str, file_id: str='16dS_-kOPkBjNgfVViU9VVnDX9h0ZLfHl')
    if not os.path.exists(folder_name):
        output = 'dataset.zip'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output, quiet=False)
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall()
        os.remove(output)
    else:
        print('Dataset is already downloaded')
        
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=folder_name, transform=transform)
    return dataset


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(512*16*16, latent_dim)
        self.fc_logvar = nn.Linear(512*16*16, latent_dim)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512*16*16),
            nn.ReLU()
        )
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        z = self.decoder(z)
        z = z.view(z.size(0), 512, 16, 16)  # Adjust the shape for convolutional layers
        z = self.decoder_conv(z)
        return z
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Define the loss function
def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Load and preprocess the dataset
dataset = get_dataset('vae_mri')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the VAE model
latent_dim = 128
vae = VAE(latent_dim)

# Define optimizer
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# Training loop
for epoch in range(50):
    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data[0])
        loss = vae_loss(recon_batch, data[0], mu, logvar)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item()}')

# Visualize the latent space using UMAP (similar to previous code)
# Generate samples from the latent space (similar to previous code)
# Evaluate the model (optional, similar to previous code)
