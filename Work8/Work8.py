import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import blurred image
blurred_image = cv2.imread('heart.jpg', cv2.IMREAD_GRAYSCALE)

# Create Gaussian kernel
kernel_size = 25
sigma = 2.5
gaussian_kernel = cv2.getGaussianKernel(kernel_size, sigma)
kernel = gaussian_kernel * gaussian_kernel.T

# Kernel use Fourier Transformer
H_uv = np.fft.fft2(kernel, s=blurred_image.shape)

# Blurred image use Fourier Transformer
G_uv = np.fft.fft2(blurred_image)

# Use Wiener filter
K = 0.001
F_uv = (G_uv / (H_uv + 1e-10)) * (np.abs(H_uv)**2 / (np.abs(H_uv)**2 + K))
F_uv = np.fft.ifftshift(F_uv)
restored_image = np.abs(np.fft.ifft2(F_uv))

# Rescale the restored image to the range 0-255
restored_image = np.clip(restored_image, 0, 255)
restored_image = restored_image.astype(np.uint8)

# Show image
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.title('Blurred Image')
plt.imshow(blurred_image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Restored Image')
plt.imshow(restored_image, cmap='gray')
plt.show()