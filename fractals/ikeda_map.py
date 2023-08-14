import torch
import matplotlib.pyplot as plt
import numpy as np

print("PyTorch Version:", torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} for computation")


def compute_ikeda_trajectory(u:float, x:float, y:float, N:int):
    #What we store the result in
    X = torch.zeros((N, 2))

    for i in range(N):
        X[i] = torch.tensor((x, y))

        t = 0.4 - 6 / (1 + x ** 2 + y ** 2)
        x1 = 1 + u * (x * torch.cos(t) - y * torch.sin(t))
        y1 = u * (x * torch.sin(t) + y * torch.cos(t))

        #Update step - > x = x1, y = y1
        x = x1
        y = y1

    return X

def plot_ikeda_trajectory(X, linewidth:float=0.1):
    plt.plot(X[:, 0], X[:, 1], 'k', linewidth=linewidth)

def main(u:float= 0.918, points:int = 100):
    #Creating 100x1 tensors with random values with mean 0 and std 10
    x = 10 * torch.randn(points, 1)
    y = 10 * torch.randn(points, 1)

    for n in range(points):
        X = compute_ikeda_trajectory(u, x[n], y[n], 1000)
        plot_ikeda_trajectory(X)

    return plt
if __name__ == "__main__":
    main().show()
