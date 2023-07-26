from PIL import Image

def transparent_background(img_name, mask_name):
    """
    Apply transparency to an image based on a grayscale mask.

    Parameters:
        img_name (str): The filename of the original image (with its extension) to be processed.
        mask_name (str): The filename of the grayscale mask image (with its extension).

    Returns:
        Image: The processed image with a transparent background.
    """
    # Load images
    img_org = Image.open('./' + img_name)
    img_mask = Image.open('./' + mask_name)

    # Convert images
    # img_org  = img_org.convert('RGB') # or 'RGBA' if needed
    img_mask = img_mask.convert('L')  # Convert the mask image to grayscale

    # Ensure both images are the same size if needed (uncomment and specify dimensions)
    # img_org  = img_org.resize((400, 400))
    # img_mask = img_mask.resize((400, 400))

    # Add alpha channel to the original image using the mask
    img_org.putalpha(img_mask)

    # Save the processed image as PNG format to preserve the alpha channel
    img_org.save(mask_name + '_transparent.png')
    return img_org

if __name__ == '__main__':
    # Example usage
    transparent_background('M40975-1-E.jpg', 'mask_M40975-1-E.jpg')
