import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read color input image
image = cv2.imread('Butterfly.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display

# Show color image
plt.figure()
plt.imshow(image_rgb)
plt.title('Color Image')
plt.axis('on')
plt.show()

# Convert to grayscale image
grayimage = 0.3 * image[:, :, 0] + 0.4 * image[:, :, 1] + 0.3 * image[:, :, 2]
grayimage = grayimage.astype(np.uint8)

# Show gray-scale image
plt.figure()
plt.imshow(grayimage, cmap='gray')
plt.title('Grayscale Image')
plt.axis('on')
plt.show()

# Plot histogram
y = np.zeros(256)

# Calculate histogram using a for loop
rows, cols = grayimage.shape
for i in range(rows):
    for j in range(cols):
        intensity = grayimage[i, j]
        y[intensity] += 1

# Plot the histogram
x = np.arange(256)

# Show histrogram
plt.figure()
plt.bar(x, y, color='gray')
plt.title('Histogram of Grayscale Image')
plt.xlabel('Intensity Value')
plt.ylabel('Pixel Number')
plt.grid(True)
plt.show()

# Convert the grayscale image to binary image with threshold 130
threshold = 130
binaryimage = np.zeros_like(grayimage)

for i in range(rows):
    for j in range(cols):
        if grayimage[i, j] < threshold:
            binaryimage[i, j] = 1
        else:
            binaryimage[i, j] = 0

# Show binary image
plt.figure()
plt.imshow(binaryimage, cmap='gray')
plt.title('Binary Image')
plt.axis('on')
plt.show()

