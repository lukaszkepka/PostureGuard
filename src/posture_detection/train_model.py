import argparse
import os.path as path
from typing import List

import cv2
import pandas as pd

from annotations import ImageAnnotation


def parse_args():
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("-i", "--annotations_file_path", required=True,
                    help="Path to annotations in csv format")
    return ap.parse_args()


def main(args):
    if not path.exists(args.annotations_file_path):
        print("File () doesn't exist".format(args.video_file_path))
        return


if __name__ == '__main__':
    args = parse_args()
    main(args)
