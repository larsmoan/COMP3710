import matplotlib.pyplot as plt
import numpy as np
import torch

print("PyTorch Version:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for computation")


def gaussian(variance: int = 4):
    # grid for computing image, subdivide the space
    size: int = 4
    resolution: float = 0.01
    X, Y = np.mgrid[-size:size:resolution, -size:size:resolution]

    x = torch.tensor(X)
    y = torch.tensor(Y)
    # transfer to GPU device, dont know what happens when I dont have a gpu though. Perhaps a waste
    x = x.to(device)
    y = y.to(device)

    z = 1 / (2 * np.pi * variance) * torch.exp(-(x**2 + y**2) / (2 * variance))

    return z


# Change the Gaussian function into a 2D sine function
def sine(a=1, b=0):
    X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]
    x = torch.tensor(X)
    y = torch.tensor(Y)
    # z = torch.sin(a*torch.sqrt(x**2 + y**2)+b)
    z = torch.sin(x + y)
    return z


def visualize(result):
    plt.imshow(result.cpu().numpy())  # Updated!
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Part 1
    # visualize(sine(a=2,b=0))
    # Multiplying the two dimensional sine function with the gaussian, makes sense that this "dampens" out the waves.
    visualize(sine())
    visualize(gaussian())
    visualize(gaussian() * sine())

    # Question: Is the 2D sine function correctly implemented? Doesnt exactly look like the Gabor filter
