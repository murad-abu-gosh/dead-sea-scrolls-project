
# Import necessary libraries
import numpy as np  # For data manipulation and calculation
import matplotlib.pyplot as plt  # For plotting the histograms
from scipy.optimize import minimize  # For finding the separating value
import numpy as np
import os
import shutil





def get_keypoints_array(path):
    keypoints = []
    for file in os.listdir(path):
        if file.endswith(".npz"):
            npz = np.load(os.path.join(path, file))
            matches_count = np.sum(npz['matches'] > -1)
            keypoints.append(matches_count)
    return keypoints

# Function to divide the array and find separator
def find_separator(numbers):
    # Sorting the list of numbers
    numbers.sort()
    # Finding the middle index
    middle_index = len(numbers) // 2
    # Dividing the list into two halves
    if len(numbers) % 2 == 0:  # even number of elements
        half1 = numbers[:middle_index]
        half2 = numbers[middle_index:]
    else:  # odd number of elements
        half1 = numbers[:middle_index]  # the first half is one element shorter
        half2 = numbers[middle_index+1:]
    # Calculating the median of each half
    median1 = np.median(half1)
    median2 = np.median(half2)
    # The separator is the midpoint between the two medians
    separator = (median1 + median2) / 2
    return half1, half2, separator

# Function to plot histograms
def plot_histograms(half1, half2):
    plt.figure(figsize=(12, 6))
    plt.hist(half1, bins=3, alpha=0.5, label='No Match')  # Reduced bin size to 10
    plt.hist(half2, bins=3, alpha=0.5, label='Match')  # Reduced bin size to 10
    plt.legend(loc='upper right')
    plt.savefig('recall_result.png')
    plt.show()





def separate_images_by_separator(path,separator):

    for file in os.listdir(path):
        if file.endswith(".npz"):
            npz = np.load(os.path.join(path, file))
            matches_count = np.sum(npz['matches'] > -1)
            image_filename = file.split('.')[0] + '.png'
            image_file_path = os.path.join(path, image_filename)
            if matches_count < separator:
                shutil.copyfile(image_file_path, os.path.join('no_matches', image_filename))
            else:
                shutil.copyfile(image_file_path, os.path.join('matches', image_filename))


if __name__ == '__main__':
    top_count = 300
    path = 'top_pics'
    print((get_keypoints_array(path)))

    # Create an example data set with a normal distribution (mean=0, std dev=1) of size 100000
    data = np.array(get_keypoints_array(path))

    # Call the function with the created data and store the separating value
    # Your list of numbers
    numbers = data

    # Find separator and divide list
    half1, half2, separator = find_separator(numbers)

    # Display the separator
    print("The separator is:", separator)

    # Plot histograms
    plot_histograms(half1, half2)

    print(np.sum(data<separator))
    print(np.sum(data>=separator))

    separate_images_by_separator(path,separator)
    # Print the separating value
    # print(f"The number that separates the two histograms is: {separating_value}")