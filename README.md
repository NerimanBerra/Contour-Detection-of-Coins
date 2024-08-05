import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Preprocessing: (Load image and Convert to grayscale)
image= cv2.imread(r'C:\Users\berra\Desktop\RKSoft\8\coin.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray');

# 2. Smoothing: Apply Gaussian smoothing to the grayscale image to reduce noise ( Use the cv2.GaussianBlur() function for smoothing)
blur = cv2.GaussianBlur(gray, (7,7), 0)
plt.imshow(blur, cmap='gray')
plt.title('1. Smoothed Grayscale Image')
plt.axis('off')  
plt.show()

# 3. Thresholding: Apply binary thresholding to the smoothed image to create a binary image (Use the cv2.threshold() function for thresholding)



# 4. Edge Detection: Apply Canny edge detection to the thresholded image (Use the cv2.Canny() function for edge detection.)
canny = cv2.Canny(blur, 50, 250)
plt.imshow(canny, cmap='gray')
plt.title('2. Edge-Detected Image')
plt.axis('off')  
plt.show()

dilated = cv2.dilate(canny, (1,1), iterations = 2)
plt.imshow(dilated, cmap='gray')


# 5. Contour Detection: Find contours in the edge-detected image (using cv2.findContours() function)
contours, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Create a copy of the original image to draw contours on
contour_image = image.copy()
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.title('3. Image with Contours')
plt.axis('off') 
plt.show()

# 6. Coin Counting: 
(cnt, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (0,255,0), 2)

plt.imshow(rgb)


plt.show()
