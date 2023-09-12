import matplotlib.pyplot as plt
import numpy as np
import os
import gdown
import zipfile
import torch
from torchvision import datasets, transforms


def get_dataset(folder_name: str, file_id: str='16dS_-kOPkBjNgfVViU9VVnDX9h0ZLfHl'):
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

def plot_input_output(random_image, reconstructed_image):
    # Assuming 'reconstructed_image' is in range [0, 1]
    reconstructed_image = reconstructed_image.squeeze().permute(1, 2, 0).cpu().numpy()  # Adjust dimensions for plotting

    # Display the original and reconstructed images
    plt.figure(figsize=(8, 4))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(random_image.squeeze().permute(1, 2, 0).detach().numpy())  # Assuming random_image is in [0, 1]
    plt.axis('off')

    # Reconstructed Image
    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Image")
    plt.imshow(reconstructed_image)
    plt.axis('off')

    plt.tight_layout()
    #Return the plot
    return plt

def plot_generated_image(generated_image):
    # Assuming 'generated_image' is in range [0, 1]
    generated_image = generated_image.squeeze().permute(1, 2, 0).detach().numpy()  # Adjust dimensions for plotting

    # Display the original and reconstructed images
    plt.figure(figsize=(8, 4))

    # Generated Image
    plt.subplot(1, 1, 1)
    plt.title("Generated Image")
    plt.imshow(generated_image)
    plt.axis('off')

    plt.tight_layout()
    #Return the plot
    return plt


