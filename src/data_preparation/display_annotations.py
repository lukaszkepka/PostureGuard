import argparse
import os.path as path

import cv2
import pandas as pd
from numpy.core.multiarray import ndarray

from annotations import ImageAnnotation, data_frame_to_annotations_list
from drawing.image_overlay import ImageOverlayPipeline, TextImageOverlayStep, BoundingBoxImageOverlayStep, \
    KeypointsImageOverlayStep

BOUNDING_BOX_COLOR = (0, 255, 255)
KEYPOINT_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 0, 0)


def parse_args():
    ap = argparse.ArgumentParser(description="Displays keypoints saved in annotations file")
    ap.add_argument("-i", "--annotations_file_path", required=True,
                    help="Path to annotations in csv format")
    return ap.parse_args()


def put_annotations_on_image(image: ndarray, image_annotations: ImageAnnotation):
    keypoints = image_annotations.keypoints

    image_overlay = ImageOverlayPipeline([
        TextImageOverlayStep([image_annotations.file_path, image_annotations.class_name], text_color=TEXT_COLOR),
        BoundingBoxImageOverlayStep(keypoints.bounding_box, color=BOUNDING_BOX_COLOR),
        KeypointsImageOverlayStep(keypoints, keypoint_color=KEYPOINT_COLOR, text_color=TEXT_COLOR)
    ])

    image_overlay.apply(image)


def display_annotations(image_annotations: ImageAnnotation):
    image = cv2.imread(image_annotations.file_path)
    put_annotations_on_image(image, image_annotations)
    cv2.imshow('annotations', image)
    cv2.waitKey()


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
