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
model.to(device)

dataset = get_dataset('vae_mri')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



learning_rate = 0.001
batch_size = 32
num_epochs = 20

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define loss function (negation of ELBO, which is equivalent to maximizing the ELBO)
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        data = batch[0]
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f'Batch loss: {loss.item()}')
      
    print(f'Epoch {epoch + 1} Loss: {total_loss / len(dataloader.dataset)}')

def generation(model):
    model.eval()
    num_samples = 10
    random_latent_vectors = torch.randn(num_samples, latent_dim).to(device)
    with torch.no_grad():
        generated_images = model.decode(random_latent_vectors)

        # Create a folder to save the generated images
        os.makedirs("generated_images", exist_ok=True)

        # Assuming `generated_images` is a torch tensor of generated images
        for i in range(num_samples):
            plt = plot_generated_image(generated_images[i])
            plt.savefig(f"generated_images/generated_image_{i}.png", bbox_inches='tight', pad_inches=0)
            plt.close()

generation(model)
# Save the trained model
torch.save(model.state_dict(), 'vae_model.pth')
