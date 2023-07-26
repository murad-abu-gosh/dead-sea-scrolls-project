import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
import skimage.io
import skimage.color
import skimage.filters
import skimage.measure
from skimage.measure import label, regionprops, regionprops_table
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

def apply_watershed_segmentation(original_filename, mask_filename):
    """
    Apply watershed segmentation to an image based on a binary mask.

    Parameters:
        original_filename (str): The filename of the original image (with its extension).
        mask_filename (str): The filename of the binary mask image (with its extension).

    Returns:
        None
    """
    # Read and preprocess the image
    og_img = cv2.imread(original_filename)
    img = cv2.imread(mask_filename)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    filtro = cv2.pyrMeanShiftFiltering(img, 20, 40)
    gray = cv2.cvtColor(filtro, cv2.COLOR_BGR2GRAY)
    gray = np.invert(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find and process contours
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    buracos = []
    for con in contornos:
        area = cv2.contourArea(con)
        if area > 30:
            buracos.append(con)
    gray = np.invert(gray)
    cv2.drawContours(thresh, buracos, -1, 255, -1)

    # Perform distance transform and watershed segmentation
    dist = ndi.distance_transform_edt(thresh)
    dist_visual = dist.copy()

    local_max = peak_local_max(dist, indices=False, min_distance=30, labels=thresh)
    markers = ndi.label(local_max, structure=np.ones((3, 3)))[0]
    labels = watershed(-dist, markers, mask=thresh)

    # Get centroid coordinates for the regions
    props = regionprops_table(labels, properties=('centroid', 'axis_major_length', 'axis_minor_length'))
    coord_tuples = [(y, x) for x, y in zip(props['centroid-0'], props['centroid-1'])]

    # Draw text labels on the original image and save the result
    draw(original_filename, coord_tuples)

    # Plot the segmented images
    plot_segmented_images([og_img, thresh, dist_visual, labels], ["Original image", "Contours", "Distance Transform", "Watershed"])

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
    img.save('with_watershed_result.jpg')

def plot_segmented_images(images_list, titles_list):
    """
    Plot multiple images with corresponding titles in a 2x2 grid.

    Parameters:
        images_list (list): A list of images to be plotted.
        titles_list (list): A list of titles for each image.

    Returns:
        None
    """
    fig = plt.gcf()
    fig.set_size_inches(16, 12)
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        if (i == 3):
            cmap = "jet"
        else:
            cmap = "gray"
        plt.imshow(images_list[i], cmap=cmap)
        plt.title(titles_list[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == '__main__':
    # Example usage
    original_image_filename = "M40975-1-E.jpg"
    mask_filename = "mask_M40975-1-E.jpg"
    apply_watershed_segmentation(original_image_filename, mask_filename)
