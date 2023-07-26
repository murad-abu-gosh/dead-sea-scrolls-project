import numpy as np
import cv2
from patchify import patchify, unpatchify
import image_slicer

x = image_slicer.slice('mask_M42280-1-E.jpg', 4)
print(type(x[0]))