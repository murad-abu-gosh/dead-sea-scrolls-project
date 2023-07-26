import cv2
import numpy as np
from matplotlib import pyplot as plt

#load Image rise
image = cv2.imread("../Oroginal Data/M40979-1-E.jpg", 0).astype("uint8") 
image = cv2.resize(image, (1280,720))
#dilate the image and erode to sub between them to find the contor of the rise
kernel = np.ones((5,5), dtype=np.uint8)
image_after_dilate = cv2.dilate(image, kernel)

kernel = np.ones((3,3), dtype=np.uint8)
image_after_erode = cv2.erode(image, kernel)

contor = image_after_dilate - image_after_erode                                 

# Threshold for contor image
contor[contor > 30] = 255
contor[contor <= 30] = 0

#invert the contor image to fill by rigon fill the image inversed it mean to fill all the output of the rise
invert = np.invert(contor)                             
filled = np.zeros(contor.shape, dtype=np.uint8)
step_back = filled.copy()
filled[0,0] = 255     

kernel = np.ones((3,3), dtype=np.uint8)

#rigon filling
while not np.array_equal(step_back,filled):
    step_back = np.copy(filled)
    filled = cv2.dilate(filled, kernel)
    filled = filled & invert

#invert again the image to get the result
kernel = np.ones((5,5), dtype=np.uint8)
filled = cv2.dilate(filled, kernel)
filled = np.invert(filled)

plt.subplot(1,3,1)
plt.imshow(image, cmap='gray')
plt.title('Image'),plt.xticks([]),plt.yticks([])
plt.subplot(1,3,2)
plt.imshow(contor, cmap='gray')
plt.title('Contor threshold'),plt.xticks([]),plt.yticks([])
plt.subplot(1,3,3)
plt.imshow(filled, cmap='gray')
plt.title('reign filled'),plt.xticks([]),plt.yticks([])
plt.show()

# cv2.imwrite('Masks/MaskSet_1280x720/mask_M43368-3-C.jpg', image)