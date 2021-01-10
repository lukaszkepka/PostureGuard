import argparse
import os.path as path
from typing import List

import cv2
import pandas as pd

from annotations import ImageAnnotation


def parse_args():
    ap = argparse.ArgumentParser(description="Extracts and displays consecutive frames from video file. Each frame can "
                                             "be classified as 'correct' or 'not_correct' by pressing keys 'a' or 'd'. "
                                             "When frame is classified it is saved to following path: "
                                             "<output_directory>/<class_name>/<video_file_name>_<frame_number>.jpg")
    ap.add_argument("-i", "--annotations_file_path", required=True,
                    help="Path to annotations in csv format")
    return ap.parse_args()


def display_annotations(image_annotations: ImageAnnotation):
    image = cv2.imread(image_annotations.file_path)
    keypoints = image_annotations.keypoints

    put_file_path(image, image_annotations.file_path)
    put_bounding_box(image, keypoints.bounding_box)
    for joint_name, localisation in keypoints.to_joint_dict().items():
        put_keypoint(image, localisation, joint_name)

    cv2.imshow('annotations', image)
    cv2.waitKey()


def put_file_path(image, file_path):
    cv2.putText(image, file_path, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                cv2.LINE_AA)


def put_bounding_box(image, bounding_box):
    bounding_box_start_point = (int(bounding_box[0][0]), int(bounding_box[0][1]))
    bounding_box_end_point = (int(bounding_box[1][0]), int(bounding_box[1][1]))
    cv2.rectangle(image, bounding_box_start_point, bounding_box_end_point, (0, 255, 255))


def put_keypoint(image, localisation, name):
    text_to_point_offset = (0, 10)
    text_localisation = tuple([localisation[0] + text_to_point_offset[0],
                               localisation[1] + text_to_point_offset[1]])

    cv2.circle(image, localisation, 3, (255, 0, 255), -1)
    cv2.putText(image, name, text_localisation, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                cv2.LINE_AA)


def data_frame_to_annotations_list(data_frame: pd.DataFrame) -> List[ImageAnnotation]:
    return [ImageAnnotation.from_data_frame(row[1].to_frame().transpose()) for row in data_frame.iterrows()]


def main(args):
    if not path.exists(args.annotations_file_path):
        print("File () doesn't exist".format(args.video_file_path))
        return

    annotations_data_frame = pd.read_csv(args.annotations_file_path)
    annotations_list = data_frame_to_annotations_list(annotations_data_frame)

    for annotation in annotations_list:

        if not path.exists(annotation.file_path):
            print("File () doesn't exist".format(args.video_file_path))
            continue

        display_annotations(annotation)


if __name__ == '__main__':
    args = parse_args()
    main(args)
