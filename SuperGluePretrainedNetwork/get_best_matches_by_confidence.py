import numpy as np
import os
import shutil


def replace_min_file(files_dict, file, matches_count):
    image_filename = file.split('.')[0]
    image_filename += '.png'
    #image_filename += '.npz'
    if len(files_dict)>0:
        if len(files_dict) >= top_count:
            files_dict.pop(min(files_dict, key=files_dict.get), None)
            files_dict[image_filename] = matches_count
        else:
            files_dict[image_filename] = matches_count
    else:
        files_dict[image_filename] = matches_count
    return files_dict

def get_100_best_matches(path):
    files_dict = {}
    for file in os.listdir(path):
        if file.endswith(".npz"):
            npz = np.load(os.path.join(path, file))
            matches_count = np.sum(npz['match_confidence'] > 0.8)
            if matches_count > 0:
                files_dict = replace_min_file(files_dict, file, matches_count)

    for file in files_dict:
        file_path = os.path.join(path, file)
        shutil.copyfile(file_path, os.path.join(output_path, file))

if __name__ == '__main__':
    top_count = 300
    path = r'data4_results'
    output_path = 'top_pics_by_confidence'
    get_100_best_matches(path)