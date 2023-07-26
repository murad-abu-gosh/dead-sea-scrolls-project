import cv2
import numpy as np


def get_blank_image(image):
    return np.zeros((image.shape[0], image.shape[1], 3), np.uint8)


# Let's load a simple image with 3 black squares
image = cv2.imread('img.png')
cv2.waitKey(0)
blank_image = get_blank_image(image)
# cv2.imshow('blank', blank_image)

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find Canny edges
edged = cv2.Canny(gray, 30, 50)
cv2.waitKey(0)

# Finding Contours
# Use a copy of the image e.g. edged.copy()
# since findContours alters the image
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
good_contours = []
for con in contours:
    area = cv2.contourArea(con)
    if area > 10:
        good_contours.append(con)
        cv2.drawContours(blank_image, [con], 0, (255, 255, 255), thickness=cv2.FILLED)
        # cv2.fillPoly(blank_image, pts=[con], color=(255, 255, 255))
        # cv2.imshow('blank', blank_image)
        # cv2.waitKey(0)
        # good_contours.remove(con)
# cv2.imshow('Canny Edges After Contouring', edged)
cv2.waitKey(0)
cv2.imshow('blank', blank_image)

print("Number of Contours found = " + str(len(good_contours)))

# Draw all contours
# -1 signifies drawing all contours
cv2.drawContours(image, good_contours, -1, (0, 255, 0), 3)

cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
