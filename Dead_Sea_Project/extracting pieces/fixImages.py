import cv2
import os

def resize_images(input_folder, output_folder, target_width, target_height):
    """
    Resize images in the input folder and save them to the output folder with the specified dimensions.

    Parameters:
        input_folder (str): The path to the folder containing the original images to be resized.
        output_folder (str): The path to the folder where the resized images will be saved.
        target_width (int): The desired width of the resized images.
        target_height (int): The desired height of the resized images.

    Returns:
        None
    """
    for i in os.listdir(input_folder):
        img_path = os.path.join(input_folder, i)
        img = cv2.imread(img_path)

        if img is not None:
            # Check if the image height is greater than the width and rotate it if necessary
            if img.shape[0] > img.shape[1]:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            # Resize the image to the target dimensions
            img = cv2.resize(img, (target_width, target_height))

            # Save the resized image to the output folder
            output_path = os.path.join(output_folder, i)
            cv2.imwrite(output_path, img)

if __name__ == '__main__':
    input_folder = 'DataSet_1280x720/Fixed_Images'
    output_folder = 'DataSet_640x360/Fixed_Images'
    target_width = 640
    target_height = 352
    resize_images(input_folder, output_folder, target_width, target_height)
