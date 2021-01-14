import argparse
import os.path as path

import pandas as pd

from annotations import Keypoints
from posture_detection.preprocessing import PreProcessingPipeline, NormalizePointCoordinatesToBoundingBox, \
    KeepColumns, PointsToVectors


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
        KeepColumns(Keypoints.ATTRIBUTE_NAMES),
        NormalizePointCoordinatesToBoundingBox(),
        KeepColumns(Keypoints.BODY_POINTS_ATTRIBUTE_NAMES),
        PointsToVectors(starting_point_name='center')
    ])

    for row_i in annotations_data_frame.iloc:
        preprocessing_pipeline.run(annotations_data_frame.iloc[row_i[0]])

    pass


if __name__ == '__main__':
    args = parse_args()
    main(args)
