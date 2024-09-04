import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read color input image
image = cv2.imread('Butterfly.jpg')

# Convert to grayscale image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate histogram using a for loop
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# Declare function | random Th = 150, repeat = 10
def iterative_threshold(gray_image, initial_threshold = 150, repeat = 10):
    Th = initial_threshold
    iteration = 0
    
    while iteration < repeat:
      
        object_group = gray_image[gray_image >= Th]
        background_group = gray_image[gray_image < Th]
        
        # Find the average of G1 & G2
        if len(object_group) > 0:
            G1 = np.mean(object_group)
        else:
            G1 = 0

        if len(background_group) > 0:
            G2 = np.mean(background_group)
        else:
            G2 = 0
        
        W1 = len(object_group)
        W2 = len(background_group)
        
        # Update the threshold by weighted averaging
        if (W1 + W2) > 0:
            Th = (W1 * G1 + W2 * G2) / (W1 + W2)
        
        iteration += 1

    return Th

new_threshold = iterative_threshold(gray_image)

# Convert to Binary Image
_, thresholded_image = cv2.threshold(gray_image, new_threshold, 255, cv2.THRESH_BINARY)

# Plot the histogram
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Intensity")
plt.ylabel("Pixel Number")
plt.xlim([0, 256])
plt.bar(range(256), hist.flatten(), width=1, edgecolor='black')

# Add vertical line for threshold
plt.axvline(x=new_threshold, color='r', linestyle='--', label=f'Threshold = {new_threshold:.2f}')
plt.legend()
plt.show()

# Show Binary Image
plt.imshow(thresholded_image, cmap='gray')
plt.title('Binary Image using Average Intensities Method')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()