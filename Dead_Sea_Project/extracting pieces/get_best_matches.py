import numpy as np
import os
import shutil


def replace_min_file(files_dict, file, matches_count):
    """
    Updates the files_dict by replacing the file with the minimum matches count, if the dictionary size exceeds 100.
    If the dictionary size is less than 100, it simply adds the file and matches_count to the dictionary.

    Parameters:
        files_dict (dict): A dictionary containing filenames as keys and their corresponding matches count as values.
        file (str): The filename to be added or replaced in the dictionary.
        matches_count (int): The number of matches for the given file.

    Returns:
        dict: The updated files_dict with the new file and matches_count added or the minimum file replaced.
    """
    image_filename = file.split('.')[0]
    image_filename += '.png'
    if len(files_dict) > 0:
        if len(files_dict) >= 100:
            files_dict.pop(min(files_dict, key=files_dict.get), None)
            files_dict[image_filename] = matches_count
        else:
            files_dict[image_filename] = matches_count
    else:
        files_dict[image_filename] = matches_count
    return files_dict


def get_100_best_matches(path):
    """
    Finds the 100 files with the best matches from the given path and copies them to a folder named 'top_100'.

    Parameters:
        path (str): The path of the folder containing the .npz files.

    Returns:
        None
    """
    files_dict = {}
    for file in os.listdir(path):
        if file.endswith(".npz"):
            npz = np.load(os.path.join(path, file))
            matches_count = np.sum(npz['matches'] > -1)
            files_dict = replace_min_file(files_dict, file, matches_count)

    # Create a new folder 'top_100' to store the top 100 files
    os.makedirs('top_100', exist_ok=True)

    # Copy the top 100 files to the 'top_100' folder
    for file in files_dict:
        file_path = os.path.join(path, file)
        shutil.copyfile(file_path, os.path.join('top_100', file))


if __name__ == '__main__':
    path = r'output'
    get_100_best_matches(path)
