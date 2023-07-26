import cv2
import os

for i in os.listdir('DataSet_1280x720/Fixed_Images'):
    img = cv2.imread('DataSet_1280x720/Data/Fixed_Images/' + i)
    if(img.shape[0] > img.shape[1]):
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    img = cv2.resize(img, (640, 352))
    cv2.imwrite('DataSet_640x360/Fixed_Images/'+ i, img)