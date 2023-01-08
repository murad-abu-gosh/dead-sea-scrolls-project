import cv2

import os
from os import listdir


def extract_pieces(img_name):
    image_original = cv2.imread('images/' + img_name)
    image_mask = cv2.imread('masks/mask_' + img_name)
    copy = image_mask.copy()
    gray = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    ROI_number = 1
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h < 10 or w < 10:
            continue
        ROI = image_original[y:y + h, x:x + w]  # work on original image
        cv2.imwrite('pieces/{}_ROI_{}.png'.format(img_name, ROI_number), ROI)
        cv2.rectangle(copy, (x, y), (x + w, y + h), (36, 255, 12), 2)
        ROI_number += 1


folder_dir = "images"
for img in os.listdir(folder_dir):

    if img.endswith(".jpg"):
        extract_pieces(img)

# cv2.imshow('thresh', thresh)
# cv2.imshow('copy', copy)
# cv2.waitKey()
