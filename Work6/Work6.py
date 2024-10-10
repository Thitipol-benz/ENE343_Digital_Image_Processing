import cv2
import numpy as np
from skimage.morphology import rectangle, erosion, dilation, reconstruction
import matplotlib.pyplot as plt

# 1. import and change to binary image
image = cv2.imread('UTK.jpg', cv2.IMREAD_GRAYSCALE)
_, image_binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# 2. Create Morphological
se = rectangle(4, 32)
se2 = rectangle(10, 10)

# 3. Use Erosion 
eroded_image = erosion(image_binary, se)

# 4. Use Morphological Reconstruction with Dilation
marker = image_binary
result = eroded_image

for i in range(10):
    result = dilation(result, se2)
    result = np.bitwise_and(result, marker)

# Show Result 
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(image_binary, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(eroded_image, cmap='gray')
plt.title('Eroded Image')

plt.subplot(1, 3, 3)
plt.imshow(result, cmap='gray')
plt.title('Reconstruction Result')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()