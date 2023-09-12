from model import VAE
from utils import get_dataset, plot_input_output
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F

model = VAE(128)

dataset = get_dataset('vae_mri')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


learning_rate = 0.001
batch_size = 32
num_epochs = 15

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
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f'Batch loss: {loss.item()}', end='\r')
      
    print(f'Epoch {epoch + 1} Loss: {total_loss / len(dataloader.dataset)}')

# Save the trained model
torch.save(model.state_dict(), 'vae_model.pth')
