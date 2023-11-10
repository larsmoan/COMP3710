import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the Autoencoder Architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load and Preprocess the MNIST Dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Initialize the Autoencoder and Define Loss Function
autoencoder = Autoencoder()

criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Training Loop
num_epochs = 50

for epoch in range(num_epochs):
    for batch in train_loader:
        inputs, _ = batch
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

#Save the model
torch.save(autoencoder.state_dict(), 'autoencoder.pth')

# Upscale an Example Image
import matplotlib.pyplot as plt

# Load a test image from MNIST
test_image, _ = train_dataset[0]
test_image = test_image.unsqueeze(0)  # Add batch dimension

# Upscale the image using the autoencoder
upscaled_image = autoencoder(test_image)

# Display the original and upscaled images
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(test_image.squeeze().numpy(), cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Upscaled Image')
plt.imshow(upscaled_image.squeeze().detach().numpy(), cmap='gray')
plt.show()