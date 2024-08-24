import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Read color input image
image = cv2.imread('Butterfly.jpg')

# Convert to grayscale image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate histogram
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
hist = hist.flatten()  

# Plot the histogram
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Intensity")
plt.ylabel("Pixel Number")
plt.xlim([0, 256])
plt.bar(range(256), hist, width=1, edgecolor='black')
plt.plot(hist)
plt.show()

# Find peaks
peaks, _ = find_peaks(hist)
pks = hist[peaks]
pks_sorted = np.sort(pks)[::-1]

if len(pks_sorted) < 2:
    raise ValueError("Not enough peaks")

# First and second highest peaks
val1 = pks_sorted[0]
val2 = pks_sorted[1]

# Find the positions of these peaks
position1 = np.where(hist == val1)[0][0]
position2 = np.where(hist == val2)[0][0]

# Calculate slope and intercept
slope = (val1 - val2) / (position1 - position2)
c = val1 - slope * position1

# Calculate the distance
diff = np.zeros_like(hist, dtype=float)
for i in range(position1, position2 + 1):
    upper = abs(slope * i - hist[i] + c)
    diff[i] = upper / np.sqrt(slope**2 - 1)

# Find the maximum difference
M = np.max(diff)
I = np.argmax(diff)

# Threshold for binarization
threshold_pos = position1 + I

# Create binary image
binarize = np.where(gray_image > threshold_pos, 1, 0)

# Show binary image
plt.imshow(binarize, cmap='gray')
plt.title('Binary Image using Maximum Normal Line Method')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()