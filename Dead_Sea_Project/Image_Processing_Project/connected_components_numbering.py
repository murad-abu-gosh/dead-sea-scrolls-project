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
    """
    Performs connected component analysis on a binary mask image to find connected regions (components).

    Parameters:
        filename (str): The filename of the binary mask image to be analyzed.
        sigma (float): The standard deviation for the Gaussian filter used to denoise the image.
            Default value is 1.0.
        t (float): The threshold value for masking the image.
            Pixels with values less than 't' will be considered part of the connected components.
            Default value is 0.5.
        connectivity (int): The connectivity level used in labeling connected regions.
            Default value is 2, which considers only the nearest 8 neighbors.

    Returns:
        tuple: A tuple containing the labeled image, the count of connected components found,
            and a list of centroid coordinates of each connected component.
    """
    # Load the image
    image = skimage.io.imread(filename)
    # Invert the image
    image = np.invert(image)
    # Convert the image to grayscale
    gray_image = skimage.color.rgb2gray(image)
    # Denoise the image with a Gaussian filter
    blurred_image = skimage.filters.gaussian(gray_image, sigma=sigma)
    # Mask the image according to the threshold
    binary_mask = blurred_image < t

    # Perform connected component analysis
    labeled_image, count = skimage.measure.label(binary_mask, connectivity=connectivity, return_num=True)
    props = regionprops_table(labeled_image, properties=('centroid', 'axis_major_length', 'axis_minor_length'))
    coord_tuples = [(y, x) for x, y in zip(props['centroid-0'], props['centroid-1'])]

    return labeled_image, count, coord_tuples

def draw(img_name, tuples_list):
    """
    Draw text labels on the image based on a list of centroid coordinates.

    Parameters:
        img_name (str): The filename of the image on which the text labels will be drawn.
        tuples_list (list): A list of tuples representing the centroid coordinates of connected components.

    Returns:
        None
    """
    img = Image.open(img_name)
    draw = ImageDraw.Draw(img)
    for i, coord in enumerate(tuples_list):
        draw.text(coord, str(i + 1), (255, 0, 0))  # Draw text label with red color
    img.save('connected_components_result.jpg')

# Example usage
original_image_filename = "M40975-1-E.jpg"
mask_filename = "mask_M40975-1-E.jpg"
labeled_image, count, coords_tuples = connected_components(filename=mask_filename, sigma=2.0, t=0.9,
                                                           connectivity=2)
draw(original_image_filename, coords_tuples)

# Display the labeled image using matplotlib
fig, ax = plt.subplots()
plt.imshow(labeled_image)
plt.imsave("connected_components_result_colored.jpg", labeled_image)
plt.axis("off")
plt.show()

# Print the labeled image and the count of connected components
print(labeled_image, count)
