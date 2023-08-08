#This is just an example program, created using chatgpt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function to generate Julia set fractal
def julia_set(width, height, max_iterations, c_real, c_imag):
    x = np.linspace(-2, 2, width)
    y = np.linspace(-2, 2, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    fractal = np.zeros((height, width), dtype=np.int32)

    for i in range(max_iterations):
        mask = np.abs(Z) < 10
        fractal += mask
        Z[mask] = Z[mask]**2 + complex(c_real, c_imag)

    return fractal

# Function to update animation frame
def update(frame):
    plt.clf()
    c_real = np.cos(frame * 0.02)
    c_imag = np.sin(frame * 0.02)
    fractal = julia_set(width, height, max_iterations, c_real, c_imag)
    plt.imshow(fractal, cmap='hot', extent=(-2, 2, -2, 2))
    plt.title(f'Julia Set Frame {frame}')
    plt.axis('off')

# Video parameters
width = 400
height = 400
max_iterations = 50

# Create animation
fig = plt.figure(figsize=(6, 6))
animation = FuncAnimation(fig, update, frames=np.arange(0, 200), interval=50)
plt.show()
