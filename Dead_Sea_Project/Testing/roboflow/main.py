import random
from pathlib import Path

images_folder_dir = "DeadSeaScrollsFragmentsDetection.v2i.yolov8/train/images"
label_folder_dir = "DeadSeaScrollsFragmentsDetection.v2i.yolov8/train/labels"
output_folder_dir = "DeadSeaScrollsFragmentsDetection.v2i.yolov8/train/output"

import argparse
import math
import imagesize
import os
import cv2
import numpy as np

classes = {
    0: "fragment",
    1: "scroll_name",
    2: "fragment_number"
}


def get_images_info_and_annotations(path):
    path_img = Path(path / Path("images"))
    annotations = {}
    color_list = np.random.randint(low=0, high=256, size=(len(classes), 3)).tolist()
    if path_img.is_dir():
        file_paths = sorted(path_img.rglob("*.jpg"))
        file_paths += sorted(path_img.rglob("*.jpeg"))
        file_paths += sorted(path_img.rglob("*.png"))
    else:
        with open(path_img, "r") as fp:
            read_lines = fp.readlines()
        file_paths = [Path(line.replace("\n", "")) for line in read_lines]

    for file_path in file_paths:
        actual_img_id = 0
        print("Image Path : ", file_path)
        # read image file
        img_file = cv2.imread(str(file_path))
        img_file = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)
        # Build image annotation, known the image's width and height
        w, h = imagesize.get(str(file_path))

        label_file_name = f"{file_path.stem}.txt"
        annotations_path = Path(path) / Path("labels") / label_file_name
        output_path = Path(path) / Path("output") / Path("all") / file_path.stem
        os.makedirs(output_path, exist_ok=True)
        if not annotations_path.exists() or not output_path.exists():
            continue  # The image may not have any applicable annotation txt file.

        with open(str(annotations_path), "r") as label_file:
            label_read_line = label_file.readlines()
        Image_Id = 0
        annotations = {}
        for line1 in label_read_line:
            label_line = line1.split()
            _class = int(label_line[0])
            x_center = float(label_line[1])
            y_center = float(label_line[2])
            width = float(label_line[3])
            height = float(label_line[4])

            float_x_center = w * x_center
            float_y_center = h * y_center
            float_width = w * width
            float_height = h * height

            min_x = int(float_x_center - float_width / 2)
            min_y = int(float_y_center - float_height / 2)
            width = int(float_width)
            height = int(float_height)
            max_x = min_x + width
            max_y = min_y + height
            output_path_extracted_image = output_path / Path(f"{random.randint(0,700)}_{Image_Id}.png")
            annotation = {
                "name": output_path_extracted_image.stem,
                "class": classes[_class],
                "min_x": min_x,
                "min_y": min_y,
                "max_x": max_x,
                "max_y": max_y,
                "width": width,
                "height": height,
                "x_center": int(float_x_center),
                "y_center": int(float_y_center)
            }
            if classes[_class] in annotations:
                annotations[classes[_class]].append(annotation)
            else:
                annotations[classes[_class]] = [annotation]
            extracted_image = img_file[min_y:max_y + 1, min_x:max_x + 1]
            cv2.imwrite(str(output_path_extracted_image), extracted_image)
            Image_Id += 1
            # Draw bounding box
        #     cv2.rectangle(
        #         img_file,
        #         (min_x, min_y),
        #         (max_x, max_y),
        #         color_list[_class],
        #         3,
        #     )
        # cv2.imshow("file_path", img_file)
        # cv2.waitKey()
        if "fragment_number" not in annotations:
            continue
        for fragment_number in annotations["fragment_number"]:
            save_best = img_file.shape[0] * img_file.shape[1]
            best_annotations = None
            for fragment in annotations["fragment"]:
                if fragment['min_x'] <= fragment_number['min_x'] and fragment['min_y'] <= fragment_number['min_y'] and \
                        fragment['max_x'] >= fragment_number['max_x'] and fragment['max_y'] >= fragment_number['max_y']:
                    best_annotations = fragment['name']
                    break
                elif fragment_number['min_y'] >= fragment['max_y']:
                    p = [fragment['y_center'], fragment['x_center']]
                    q = [fragment_number['y_center'], fragment_number['x_center']]
                    distance = math.dist(p, q)
                    if distance < save_best:
                        best_annotations = fragment['name']
                else:
                    print(fragment)
                    print(fragment_number)
            if best_annotations:
                output_path_best = Path(path) / Path("output") / Path("best") / file_path.stem
                os.makedirs(output_path_best, exist_ok=True)
                if not output_path_best.exists():
                    continue  # The image may not have any applicable annotation txt file.
                output_path_fragment = output_path_best / Path(f"{best_annotations}.png")
                output_path_fragment_number = output_path_best / Path(
                    f"{fragment_number['name']}.png")
                img_fragment = cv2.imread(str(output_path / Path(f"{best_annotations}.png")))
                img_fragment_number = cv2.imread(str(output_path / Path(f"{fragment_number['name']}.png")))
                cv2.imwrite(str(output_path_fragment), img_fragment)
                cv2.imwrite(str(output_path_fragment_number), img_fragment_number)
    return annotations


def main(path):
    print("Start!")
    annotations = get_images_info_and_annotations(path)
    print(annotations)
    print("Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Yolo format annotations to COCO dataset format")
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="Absolute path for 'train.txt' or 'test.txt', or the root dir for images.",
    )
    args = parser.parse_args()
    path = "DeadSeaScrollsFragmentsDetection.v2i.yolov8/test"
    if args.path:
        path = args.path
    main(path)
