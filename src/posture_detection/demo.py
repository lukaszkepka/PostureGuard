import argparse
import os.path as path

import cv2
import numpy as np

from annotations import SUPPORTED_CLASSES, ImageAnnotation
from data_preparation.display_annotations import put_annotations_on_image
from keypoints_detection.factory import create_keypoint_detector
from posture_detection.simple_nn_model import SimpleNNModel


def parse_args():
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("-i", "--video_file_path", required=True,
                    help="Path to input video file")
    ap.add_argument("-p", "--posture_detector_model_path", required=True,
                    help="Path to trained model")
    ap.add_argument("-k", "--keypoint_detector_model_path", required=True,
                    help="Path to keypoint detection model")
    return ap.parse_args()


def main(args):
    if not path.exists(args.video_file_path):
        print("File {0} doesn't exist".format(args.video_file_path))
        return

    if not path.exists(args.posture_detector_model_path):
        print("Model {0} doesn't exist".format(args.posture_detector_model_path))
        return

    if not path.exists(args.keypoint_detector_model_path):
        print("Model {0} doesn't exist".format(args.posture_detector_model_path))
        return

    keypoint_detector = create_keypoint_detector(args.keypoint_detector_model_path)
    posture_detector = SimpleNNModel(args.posture_detector_model_path, load_weights=True)

    process_video(args.video_file_path, keypoint_detector, posture_detector)


def process_frame(frame, keypoint_detector, posture_detector):
    height, width, channels = frame.shape

    keypoints = keypoint_detector.detect(frame)
    if len(keypoints) == 0:
        return

    annotation = ImageAnnotation.from_parameters('', (height, width), '', keypoints[0])
    predictions = posture_detector.predict(annotation.to_dataframe())
    annotation.class_name = SUPPORTED_CLASSES[predictions[0]]
    put_annotations_on_image(frame, annotation)


def process_video(video_file_path, keypoint_detector, posture_detector):
    frame_num = 1
    display_interval = 3
    cap = cv2.VideoCapture(video_file_path)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret < 0 or frame is None:
            break

        frame_num += 1
        if frame_num % display_interval > 0:
            continue

        process_frame(frame, keypoint_detector, posture_detector)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        _ = cv2.waitKeyEx(1)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    main(args)
