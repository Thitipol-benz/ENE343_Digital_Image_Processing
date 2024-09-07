# Histogram Equalization
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1) Use color image
# Read color input image
image = cv2.imread('Butterfly.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display

# Show color image
plt.figure()
plt.imshow(image_rgb)
plt.title('Color Image')
plt.axis('on')
plt.show()

# Histrogram Equalization
# 1. Find histrogram
channels = cv2.split(image_rgb)
equalized_channels = []
for channel in channels:
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
plt.bar(range(256), hist.flatten(), width=1, edgecolor='black')
plt.title('Color Image Histogram')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')
plt.show()

# 2. Fins Probability Density Function 
pdf = hist / np.sum(hist)

# 3. Find Commulative Distributiion Function
cdf = np.zeros(256)
cdf[0] = pdf[0]
for i in range(1, 256):
    cdf[i] = cdf[i - 1] + pdf[i]

# 4. Change the gray level
cdf_normalized = (255 * cdf).astype(np.uint8)
equalized_channel = cdf_normalized[channel]
equalized_channels.append(equalized_channel)
equalized_image = cv2.merge(equalized_channels)

# 5. Show image after equalization
plt.figure()
plt.imshow(equalized_image,cmap='gray')
plt.title('Color Image Equalization')
plt.axis('on')
plt.show()

# Plot histrogram after equalization
hist = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])
plt.bar(range(256), hist.flatten(), width=1, edgecolor='black')
plt.title('Color Image Equalization Histogram')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')
plt.show()

# -------------------------------------------------------------------------------------------------------------------------------------------------
# 2) Use grayscale image
# Convert to grayscale image
grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Show gray-scale image
plt.figure()
plt.imshow(grayimage, cmap='gray')
plt.title('Grayscale Image')
plt.axis('on')
plt.show()

# Histogram Equalization
# 1. Find histogram
hist2 = cv2.calcHist([grayimage], [0], None, [256], [0, 256])
plt.bar(range(256), hist2.flatten(), width=1, edgecolor='black')
plt.title('Grayscale Image Histogram')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')
plt.show()

# 2. Find Probability Density Function 
pdf = hist2 / np.sum(hist2)

# 3. Find Cumulative Distribution Function 
cdf = np.zeros(256)
cdf[0] = pdf[0]
for i in range(1, 256):
    cdf[i] = cdf[i - 1] + pdf[i]

# 4. Change the gray level
cdf_normalized = (cdf * 255).astype(np.uint8)

# Show image after equalization
equalized_image = cdf_normalized[grayimage]

plt.figure()
plt.imshow(equalized_image, cmap='gray')
plt.title('Grayscale Image Equalization')
plt.axis('on')
plt.show()

# Plot histrogram after equalization
hist = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])
plt.bar(range(256), hist.flatten(), width=1, edgecolor='black')
plt.title('Grayscale Image Equalization Histogram')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()