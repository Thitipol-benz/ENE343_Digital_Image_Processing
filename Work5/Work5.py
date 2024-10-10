import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import image
image = cv2.imread('Lion.jpg', cv2.IMREAD_GRAYSCALE)

plt.imshow(image, cmap='gray')
plt.title('Gray Image')
plt.axis('on')
plt.show()

# Get image size
rows, cols = image.shape
crow, ccol = rows // 2 , cols // 2  # Center

# Fourier Transform and shift zero frequency to center
dft = np.fft.fft2(image)
dft_shifted = np.fft.fftshift(dft)

# 1) Low Pass Filter
# Create Gaussian Low-Pass Filter
D0 = 5  # Cut-off frequency
u = np.arange(rows)
v = np.arange(cols)
U, V = np.meshgrid(u - crow, v - ccol)
D = np.sqrt(U**2 + V**2)  

# Gaussian filter formula
H = np.exp(-(D**2) / (2 * (D0**2)))

# Apply filter in the frequency domain
filtered_dft = dft_shifted * H

# Inverse FFT to get the image back
inverse_shift = np.fft.ifftshift(filtered_dft)
img_back1 = np.fft.ifft2(inverse_shift)
img_back1 = np.abs(img_back1)

plt.imshow(img_back1, cmap='gray')
plt.title('Low Pass Filter')
plt.axis('on')
plt.show()

D0 = 30  # Cut-off frequency
u = np.arange(rows)
v = np.arange(cols)
U, V = np.meshgrid(u - crow, v - ccol)
D = np.sqrt(U**2 + V**2)  

# Gaussian filter formula
H = np.exp(-(D**2) / (2 * (D0**2)))

# Apply filter in the frequency domain
filtered_dft = dft_shifted * H

inverse_shift = np.fft.ifftshift(filtered_dft)
img_back2 = np.fft.ifft2(inverse_shift)
img_back2 = np.abs(img_back2) 

plt.imshow(img_back2, cmap='gray')
plt.title('Low Pass Filter')
plt.axis('on')
plt.show()

# 2) High Pass Filter
# Create a Gaussian High-Pass Filter
D0 = 30  # Cut-off frequency
u = np.arange(rows)
v = np.arange(cols)
U, V = np.meshgrid(u - crow, v - ccol)
D = np.sqrt(U**2 + V**2)  

# Gaussian high-pass filter formula 
H = 1 - np.exp(-(D**2) / (2 * (D0**2)))

# Apply the Gaussian high-pass filter in  frequency domain
filtered_dft = dft_shifted * H

# Invers to  the spatial domain
inverse_shift = np.fft.ifftshift(filtered_dft)
img_back3 = np.fft.ifft2(inverse_shift)
img_back3 = np.abs(img_back3)

plt.imshow(img_back3, cmap='gray')
plt.title('High Pass Filter')  
plt.axis('on')
plt.show()

D0 = 100  # Cut-off frequency
u = np.arange(rows)
v = np.arange(cols)
U, V = np.meshgrid(u - crow, v - ccol)
D = np.sqrt(U**2 + V**2)  

# Gaussian high-pass filter formula 
H = 1 - np.exp(-(D**2) / (2 * (D0**2)))

# Apply the Gaussian high-pass filter in  frequency domain
filtered_dft = dft_shifted * H

# Invers to  the spatial domain
inverse_shift = np.fft.ifftshift(filtered_dft)
img_back4 = np.fft.ifft2(inverse_shift)
img_back4 = np.abs(img_back4)

plt.imshow(img_back4, cmap='gray')
plt.title('High Pass Filter')  
plt.axis('on')
plt.show()

cv2.waitKey(0)  
cv2.destroyAllWindows()