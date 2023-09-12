from model import VAE
import torch
import os
import matplotlib.pyplot as plt
from utils import plot_input_output, plot_generated_image

latent_dim = 128 
model = VAE(latent_dim)
model.load_state_dict(torch.load('vae_model.pth'))  
model.eval() 

num_samples = 10  # Number of images you want to generate
random_latent_vectors = torch.randn(num_samples, latent_dim)

with torch.no_grad():
    generated_images = model.decode(random_latent_vectors)
    
# Create a folder to save the generated images
os.makedirs("generated_images", exist_ok=True)

# Assuming `generated_images` is a torch tensor of generated images
for i in range(num_samples):
    plt = plot_generated_image(generated_images[i])
    plt.savefig(f"generated_images/generated_image_{i}.png", bbox_inches='tight', pad_inches=0)
    plt.close()