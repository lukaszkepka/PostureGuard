import argparse
import os.path as path
from typing import List

import cv2
import pandas as pd

from annotations import ImageAnnotation
from posture_detection.preprocessing import PreProcessingPipeline, NormalizePointCoordinatesToBoundingBox, RemoveColumns


def parse_args():
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("-i", "--annotations_file_path", required=True,
                    help="Path to annotations in csv format")
    return ap.parse_args()


def main(args):
    if not path.exists(args.annotations_file_path):
        print("File () doesn't exist".format(args.video_file_path))
        return

    annotations_data_frame = pd.read_csv(args.annotations_file_path)
    preprocessing_pipeline = PreProcessingPipeline([
        RemoveColumns(ImageAnnotation.ATTRIBUTE_NAMES),
        NormalizePointCoordinatesToBoundingBox()
    ])

    for row_i in annotations_data_frame.iloc:
        preprocessing_pipeline.run(annotations_data_frame.iloc[row_i[0]])


if __name__ == '__main__':
    args = parse_args()
    main(args)
