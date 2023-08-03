import torch
import numpy as np
import matplotlib.pyplot as plt

print("PyTorch Version:", torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} for computation")

def gaussian_blur():
    #grid for computing image, subdivide the space
    X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]
    x = torch.tensor(X)
    y = torch.tensor(Y)
    #transfer to GPU device, dont know what happens when I dont have a gpu though. Perhaps a waste
    x = x.to(device)
    y = y.to(device)
    #Compute the gaussian "blur?"
    z = torch.exp(-(x**2+y**2)/2.0)
    return z

#Change the Gaussian function into a 2D sine function

def sine():
    X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]
    x = torch.tensor(X)
    y = torch.tensor(Y)
    sine = torch.sin(x*y)
    return sine

def visualize(result):
    plt.imshow(result.numpy())
    plt.tight_layout()
    plt.show()

def sine_gauss_product():
    s = sine()
    gauss = gaussian_blur()
    result = s*gauss
    visualize(result)

sine_gauss_product()


#Part two
