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

def connected_components(filename, sigma=1.0, t=0.5, connectivity=2):
    # load the image
    image = skimage.io.imread(filename)
    # invert image
    image = np.invert(image)
    # convert the image to grayscale
    gray_image = skimage.color.rgb2gray(image)
    # denoise the image with a Gaussian filter
    blurred_image = skimage.filters.gaussian(gray_image, sigma=sigma)
    # mask the image according to threshold
    binary_mask = blurred_image < t

    # perform connected component analysis
    labeled_image, count = skimage.measure.label(binary_mask, connectivity=connectivity, return_num=True)
    props = regionprops_table(labeled_image, properties=('centroid', 'axis_major_length', 'axis_minor_length'))
    print((props))
    coord_tuples = []

    for x, y in zip(props['centroid-0'], props['centroid-1']):
        coord_tuples.append((y, x))

    return labeled_image, count, coord_tuples





def draw(img_name, tuples_list):
    img = Image.open(img_name)
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    # font = ImageFont.truetype("sans-serif.ttf", 16)
    # draw.text((x, y),"Sample Text",(r,g,b))
    for i in range(len(tuples_list)):
        draw.text(tuples_list[i], str(i + 1), (255, 0, 0))
    img.save('connected_components_result.jpg')


filename = "mask_M42035-1-C.jpg"
labeled_image, count, coords_tuples = connected_components(filename=filename, sigma=2.0, t=0.9,
                                                           connectivity=2)
draw(filename, coords_tuples)
fig, ax = plt.subplots()
plt.imshow(labeled_image)
plt.axis("off")
plt.show()
print(labeled_image, count)
