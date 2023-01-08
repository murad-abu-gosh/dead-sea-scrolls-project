import cv2
import os

k = 0
for i in os.listdir('DataSet/handwrite'):
    if k == 5: break
    k += 1
    imgName = 'DataSet/handwrite/' + i
    img = cv2.imread(imgName, cv2.COLOR_BGR2GRAY)
    if (img.shape[0] < img.shape[1]):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.resize(img, (1280, 720))
    cv2.imwrite('DataSet/DataSet_1280x720/handwrite/' + i, img)
k = 0
for i in os.listdir('DataSet/printed'):
    if k == 5: break
    k += 1
    imgName = 'DataSet/printed/' + i
    img = cv2.imread(imgName, cv2.COLOR_BGR2GRAY)
    if (img.shape[0] < img.shape[1]):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.resize(img, (1280, 720))
    cv2.imwrite('DataSet/DataSet_1280x720/printed/' + i, img)
