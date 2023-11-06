import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml

# Step 1: Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Step 2: Preprocess the data
X = mnist.data / 255.0  # Normalize pixel values between 0 and 1

# Step 3: Flatten the images (28x28 -> 784)
X_flattened = X.reshape((-1, 784))

# Step 4: Compute the Covariance Matrix (implicitly done in PCA)

# Step 5: Perform SVD (implicitly done in PCA)
pca = PCA(n_components=50)  # Number of principal components to retain
pca.fit(X_flattened)

# Step 6: Transform Data
X_compressed = pca.transform(X_flattened)
X_reconstructed = pca.inverse_transform(X_compressed)

# Step 8: Display Original and Compressed Images
n_samples = 5  # Number of samples to display

plt.figure(figsize=(10, 4))

for i in range(n_samples):
    # Original Image
    plt.subplot(2, n_samples, i + 1)
    plt.imshow(X_flattened[i].reshape(28, 28), cmap='gray')
    plt.title('Original')
    plt.axis('off')

    # Compressed Image
    plt.subplot(2, n_samples, n_samples + i + 1)
    plt.imshow(X_reconstructed[i].reshape(28, 28), cmap='gray')
    plt.title('Compressed')
    plt.axis('off')

plt.tight_layout()
plt.show()
