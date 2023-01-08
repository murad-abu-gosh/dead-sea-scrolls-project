import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.color
import skimage.filters
import skimage.measure
from skimage.measure import label, regionprops, regionprops_table
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

orig_filename = 'M42035-1-C.jpg'
mask_filename = 'mask_M42035-1-C.jpg'
img = cv2.imread(mask_filename)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
filtro = cv2.pyrMeanShiftFiltering(img, 20, 40)
gray = cv2.cvtColor(filtro, cv2.COLOR_BGR2GRAY)
gray = np.invert(gray)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

contornos, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
buracos = []
for con in contornos:
    area = cv2.contourArea(con)
    if area < 100:
        buracos.append(con)
cv2.drawContours(thresh, buracos, -1, 255, -1)

dist = ndi.distance_transform_edt(thresh)
dist_visual = dist.copy()

local_max = peak_local_max(dist, indices=False, min_distance=20, labels=thresh)
markers = ndi.label(local_max, structure=np.ones((3, 3)))[0]
labels = watershed(-dist, markers, mask=thresh)
###################################################
props = regionprops_table(labels, properties=('centroid', 'axis_major_length', 'axis_minor_length'))
print((props))
coord_tuples = []

for x, y in zip(props['centroid-0'], props['centroid-1']):
    coord_tuples.append((y, x))


def draw(img_name, tuples_list):
    img = Image.open(img_name)
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    # font = ImageFont.truetype("sans-serif.ttf", 16)
    # draw.text((x, y),"Sample Text",(r,g,b))
    for i in range(len(tuples_list)):
        draw.text(tuples_list[i], str(i + 1), (255, 0, 0))
    img.save('with_watershed_result.jpg')


draw(orig_filename, coord_tuples)
###################################################

######################################### PLOT
titulos = ['Original image', 'Binary Image', 'Distance Transform', 'Watershed']
imagens = [img, thresh, dist_visual, labels]
fig = plt.gcf()
fig.set_size_inches(16, 12)
for i in range(4):
    plt.subplot(2, 2, i + 1)
    if (i == 3):
        cmap = "jet"
    else:
        cmap = "gray"
    plt.imshow(imagens[i], cmap)
    plt.title(titulos[i])
    plt.xticks([]), plt.yticks([])
plt.show()
######################################
