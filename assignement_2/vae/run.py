from model import VAE
from utils import get_dataset, plot_input_output
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
from utils import plot_generated_image
import matplotlib.pyplot as plt
import argparse
import datetime
import wandb
import datetime


# Define loss function (negation of ELBO, which is equivalent to maximizing the ELBO)
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(model, optimizer, dataloader, num_epochs=5):
    model.train()
    try:
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
                wandb.log({"batch_loss": loss.item()})

            
            print(f'Epoch {epoch + 1} Loss: {total_loss / len(dataloader.dataset)}')
            wandb.log({"epoch_loss": total_loss / len(dataloader.dataset)})
    except KeyboardInterrupt:
        print("Canceled training, saving current weights")
        now = datetime.datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H")
        torch.save(model.state_dict(), f'vae_model_{dt_string.replace("/", "_")}.pth')
    #Save the run
    torch.save(model.state_dict(), f'vae_model_{datetime.datetime.now()}.pth')

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


if __name__ == "__main__":
    #Get arguments from cli
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 2
    dataset = get_dataset('vae_mri')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    if args.action == 'train':
        print("Training")
        now = datetime.datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H")
        wandb.init(project="VAE", name=f'vae_{dt_string.replace("/", "_")}')
        
        model = VAE(latent_dim)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        wandb.config.latent_dim = model.latent_dim
        wandb.config.batch_size = 32
        wandb.config.num_epochs = 20
        wandb.config.learning_rate = 0.001

        train(model, optimizer, dataloader, num_epochs=wandb.config.num_epochs)
    else:
        print("Running image generation")
        model = VAE(latent_dim)
        model.load_state_dict(torch.load('vae_model.pth'))
        model.to(device)
        generation(model)
