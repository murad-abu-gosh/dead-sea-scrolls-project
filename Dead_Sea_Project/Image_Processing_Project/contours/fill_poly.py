import numpy as np
import cv2


def display_filled_polygon(contours):
    """
    Display a filled polygon on a black image using OpenCV.

    Parameters:
        contours (numpy.ndarray): A numpy array representing the vertices of the polygon.
            It should be a 2D array with shape (N, 2), where N is the number of vertices.

    Returns:
        None
    """
    img = np.zeros((200, 200))  # create a single-channel 200x200 pixel black image
    cv2.fillPoly(img, pts=[contours], color=(255, 255, 255))  # Fill the polygon with white color
    cv2.imshow("Filled Polygon", img)  # Display the image with the filled polygon
    cv2.waitKey(0)  # Wait until a key is pressed to close the image window


if __name__ == "__main__":
    # Example polygon coordinates: a square
    contours = np.array([[50, 50], [50, 150], [150, 150], [150, 50]])

    # Display the filled polygon
    display_filled_polygon(contours)
