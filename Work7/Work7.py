import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# import photo
image = cv2.imread('CORTEF20.jpg')

# change to gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binaryImage = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Sobel operators
sobelOperatorX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobelOperatorY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Calculate gradient in X and Y directions
gradientX = convolve(binaryImage.astype(float), sobelOperatorX)
gradientY = convolve(binaryImage.astype(float), sobelOperatorY)

# Calculate magnitude of the gradient
gradientMagnitude = np.sqrt(gradientX**2 + gradientY**2)

# Display images of gradients and magnitude
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.imshow(binaryImage, cmap='gray'); plt.title('Original Binary Image')
plt.subplot(1, 3, 2); plt.imshow(gradientX, cmap='gray'); plt.title('Gradient X')
plt.subplot(1, 3, 3); plt.imshow(gradientY, cmap='gray'); plt.title('Gradient Y')
plt.show()

plt.figure()
plt.imshow(gradientMagnitude, cmap='gray')
plt.title('Gradient Magnitude')
plt.show()

# Calculate gradient direction
gradientDirection = np.degrees(np.arctan2(gradientY, gradientX))

# Convert negative angles to positive
gradientDirection[gradientDirection < 0] += 360

# Display gradient direction image
plt.figure()
plt.imshow(gradientDirection, cmap='gray')
plt.title('Gradient Direction')
plt.show()

# Create histogram of gradient direction
plt.figure()
plt.hist(gradientDirection.ravel(), bins=np.arange(0, 370, 10), density=True)
plt.title('Edge Direction Histogram')
plt.xlabel('Edge Direction (degrees)')
plt.ylabel('Probability')
plt.show()

# Create Laplacian filter (4-connectivity)
laplacianFilter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

# Calculate Laplacian of the binary image
laplacianImage = convolve(binaryImage.astype(float), laplacianFilter)

# Display Laplacian image
plt.figure()
plt.imshow(laplacianImage, cmap='gray')
plt.title('Laplacian Image')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()