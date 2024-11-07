import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Import the image
image = cv2.imread('ObjectSegProb.jpg', cv2.IMREAD_GRAYSCALE)

# Display the image
plt.figure('Original Image')
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.show()

# Announce the values ​​specified in the problem
mean_obj = 170
mean_bg = 60

# Set the standard deviation of the noise
std_noise = 10

# Calculate the threshold value for segmentation
threshold = (mean_obj + mean_bg) / 2 + 3 * std_noise

# Calculate the x-axis values for the Gaussian curve
x = np.linspace(mean_obj - 3 * std_noise, mean_obj + 3 * std_noise, 1000)
pdf_values = norm.pdf(x, mean_obj, std_noise)

# Find the range that gives at least 90% of the area under the Gaussian curve
cumulative_sum = np.cumsum(pdf_values) / np.sum(pdf_values)
threshold_index = np.where(cumulative_sum >= 0.9)[0][0]

# Extract the desired range from the Gaussian curve
threshold_range = x[:threshold_index]

# Apply thresholding to segment the image
binary_img = image > threshold

# Display the original and segmented images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(binary_img, cmap='gray')
plt.title('Segmented Image')
plt.show()

# Calculate the area of the object
obj_area = np.sum(binary_img)

# Calculate the total area
total_area = binary_img.size

# Calculate segmentation accuracy
accuracy = (obj_area / total_area) * 100
print(f'Segmentation accuracy: {accuracy:.2f}%')