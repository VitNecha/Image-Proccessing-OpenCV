
# Imports
import numpy as np
import cv2

# Main picture (with green background)
img = cv2.imread("front1.jpg")

# Convert image: BGR -> HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# HSV representation of green
green = np.uint8([[[0, 255, 0]]])
hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)

# Green color boundaries
light_green = (40, 70, 70)
dark_green = (80, 255, 255)

# Mask creation (for green area)
mask = cv2.inRange(hsv_img, light_green, dark_green)

# Open background and correct it's size
background = cv2.imread("bg1.png")
background = cv2.resize(background, (mask.shape[1], mask.shape[0]))

# Attach mask to the background
background = cv2.bitwise_and(background, background, mask=mask)

# Mask negative (opposite)
maskNeg = (255 - mask)

# Attach negative mask to original image
img = cv2.bitwise_and(img, img, mask=maskNeg)

# Combine image and the background
output = cv2.bitwise_or(background, img)

# Output image
cv2.imwrite("output.jpg", output)
