import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import image
image = cv2.imread('Lion.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show color image
plt.figure()
plt.imshow(image_rgb)
plt.title('Color Image')
plt.axis('on')
plt.show()

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Show grayscale image
plt.figure()
plt.imshow(gray_image, cmap='gray')
plt.title('Gray Image')
plt.axis('on')
plt.show()

# 1. Low Pass Filter (Mean filter)
mask3x3 = np.ones((3, 3)) / 9
mask9x9 = np.ones((9, 9)) / 81
rows, cols = gray_image.shape

# Create an empty image
Filteredgray_image = np.zeros_like(gray_image)

# Apply mean filter mask 3x3
for x in range(1, rows-1):
    for y in range(1, cols-1):
        # Extract the 3x3 neighborhood
        neighborhood = gray_image[x-1:x+2, y-1:y+2]
        # Apply the filter (mean)
        Filteredgray_image[x, y] = np.sum(neighborhood * mask3x3)

plt.figure()
plt.imshow(Filteredgray_image, cmap='gray')
plt.title('Image Mean Filter 3x3')
plt.axis('on')
plt.show()

# Apply mean filter mask 9x9
for x in range(4, rows-4):
    for y in range(4, cols-4):
        # Extract the 9x9 neighborhood
        neighborhood = gray_image[x-4:x+5, y-4:y+5]
        # Apply the filter (mean)
        Filteredgray_image[x, y] = np.sum(neighborhood * mask9x9)

plt.figure()
plt.imshow(Filteredgray_image, cmap='gray')
plt.title('Image Mean Filter 9x9')
plt.axis('on')
plt.show()

# 2.Highpass Filter (Laplacian Filter)
# Define the Laplacian filter (3x3)
mask_laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

# Create an empty image to store the filtered result
filtered_gray = np.zeros_like(gray_image)

# Get image dimensions
rows, cols = gray_image.shape

# Apply the Laplacian filter
for x in range(1, rows-1):
    for y in range(1, cols-1):
        summ = 0
        # Apply the 3x3 mask
        for X in range(-1, 2):
            for Y in range(-1, 2):
                summ += gray_image[x+X, y+Y] * mask_laplacian[1+X, 1+Y]
        # Clip the value to the range [0, 255]
        filtered_gray[x, y] = max(0, min(255, summ))

# Display the filtered image
plt.figure()
plt.imshow(filtered_gray, cmap='gray')
plt.title('Image Laplacian Filter 3x3')
plt.axis('on')
plt.show()

cv2.waitkey(0)
cv2.destroyAllWindows()