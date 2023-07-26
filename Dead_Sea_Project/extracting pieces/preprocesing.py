from PIL import Image
import cv2
import os

def get_max_dimensions(path):
    """
    Get the maximum dimensions (width and height) of images in the specified folder.

    Parameters:
        path (str): The path to the folder containing the images.

    Returns:
        tuple: A tuple containing the maximum width and height of the images.
    """
    return max(Image.open(os.path.join(path, f), 'r').size for f in os.listdir(path))


def resize_and_rotate_images(input_folder, output_folder, target_width, target_height):
    """
    Resize and rotate images in the input folder to fit the specified target dimensions.
    Images with portrait orientation will be rotated to landscape mode.

    Parameters:
        input_folder (str): The path to the folder containing the original images to be resized.
        output_folder (str): The path to the folder where the resized images will be saved.
        target_width (int): The desired width of the resized images.
        target_height (int): The desired height of the resized images.

    Returns:
        None
    """
    new_size = (target_width, target_height)
    for piece in os.listdir(input_folder):
        img_path = os.path.join(input_folder, piece)
        img = cv2.imread(img_path)

        if img is not None:
            # Check if the image height is greater than the width and rotate it if necessary
            if img.shape[0] > img.shape[1]:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            old_im = Image.open(img_path)
            old_size = old_im.size
            new_im = Image.new("RGB", new_size)
            box = tuple((n - o) // 2 for n, o in zip(new_size, old_size))
            new_im.paste(old_im, box)

            # Save the resized and rotated image to the output folder
            output_path = os.path.join(output_folder, piece)
            new_im.save(output_path)

if __name__ == '__main__':
    input_folder = 'pieces'
    output_folder = 'pieces_fixed'
    target_width = 1024
    target_height = 576

    # Get the maximum dimensions of the images in the input folder
    # Note: If you prefer to use a specific target size, uncomment the following line
    # instead of calling get_max_dimensions(path).
    # new_size = (target_width, target_height)

    # Resize and rotate images to fit the specified target dimensions
    resize_and_rotate_images(input_folder, output_folder, target_width, target_height)
