import argparse
import os
import os.path as path
import pandas as pd
import cv2
import progressbar

from annotations import ImageAnnotation, SUPPORTED_CLASSES
from keypoints_detection.KeypointDetector import KeypointDetector
from keypoints_detection.factory import create_keypoint_detector

allowed_extensions = ['.jpg']


class InvalidDirectoryStructureError(Exception):
    pass


def parse_args():
    ap = argparse.ArgumentParser(description='Processes images placed in images_directory with keypoint detector and '
                                             'creates annotation file for whole dataset in csv format. Images in '
                                             'images_directory must be organized in subfolders named by image '
                                             'class names')
    ap.add_argument("-i", "--images_directory", required=True,
                    help="Path to directory with images")
    ap.add_argument("-m", "--model_path", required=True,
                    help="Path to directory with images")
    return ap.parse_args()


def get_images_to_process(directory_path):
    image_path_to_class_mapping = {}

    for subdirectory_name in list_directories(directory_path):
        subdirectory_path = os.path.join(directory_path, subdirectory_name)
        class_name = subdirectory_name

        if not os.path.isdir(subdirectory_path):
            raise InvalidDirectoryStructureError(
                "Incorrect directory structure. images_directory should contain "
                "images in subdirectories named by class name")

        if class_name in SUPPORTED_CLASSES:
            for file_path in list_files_with_extension(subdirectory_path, allowed_extensions):
                if image_path_to_class_mapping.get(file_path) is not None:
                    print(f'Duplicate file {file_path}')
                    continue

                image_path_to_class_mapping[file_path] = class_name

    return image_path_to_class_mapping


def process_images(images_directory, keypoint_detector: KeypointDetector):
    annotations_df = pd.DataFrame()
    images_to_class_mapping = get_images_to_process(images_directory)
    progress_bar = progressbar.ProgressBar(max_value=len(images_to_class_mapping))

    for i, image_info in enumerate(images_to_class_mapping.items()):
        image_path, class_name = image_info
        keypoints = process_image(image_path, keypoint_detector)
        annotations_df = annotations_df.append(
            get_image_annotation(image_path, class_name, keypoints), ignore_index=True)
        progress_bar.update(i)

    return annotations_df


def process_image(image_path, keypoint_detector: KeypointDetector):
    keypoints = keypoint_detector.detect(image_path)

    if len(keypoints) == 0:
        return None

    # We process only first detection
    # TODO: Select detection by size or something else
    keypoints = keypoints[0]

    return keypoints


def get_image_annotation(image_path, class_name, keypoints):
    if keypoints is None:
        return pd.DataFrame()

    img = cv2.imread(image_path)
    height, width, channels = img.shape

    annotation = ImageAnnotation.from_parameters(image_path, (height, width), class_name, keypoints)
    return annotation.to_dataframe()


def list_directories(directory_path):
    return list(filter(lambda x: os.path.isdir(os.path.join(directory_path, x)), os.listdir(directory_path)))


def list_files_with_extension(directory_path, extensions):
    return [os.path.join(directory_path, image_name) for image_name in os.listdir(directory_path)
            if os.path.splitext(image_name)[-1] in extensions]


def main(args):
    if not path.exists(args.images_directory):
        print("Directory () doesn't exist".format(args.images_directory))
        return

    keypoint_detector = create_keypoint_detector(args)
    annotations_df = process_images(args.images_directory, keypoint_detector)

    output_path = os.path.join(args.images_directory, 'annotations.csv')
    annotations_df.to_csv(output_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
