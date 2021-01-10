import os
from typing import List
import numpy as np
import cv2

from CenterNet.src.lib.detectors.detector_factory import detector_factory
from CenterNet.src.lib.opts import opts
from annotations import Keypoints
from keypoints_detection.KeypointDetector import KeypointDetector


class CenterNetKeypointDetectorAdapter(KeypointDetector):

    def __init__(self, model_path):
        center_net_args = self.get_center_net_args(model_path)
        self.opt = opts().init(center_net_args)
        self.detector = self._create_detector()
        self.debug = False

    def _create_detector(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpus_str
        detector = detector_factory[self.opt.task]
        return detector(self.opt)

    def detect(self, image_path) -> List[Keypoints]:
        ret = self.detector.run(image_path)
        multipose_detections = ret['results'][1]

        return [self.map_result_to_keypoints(multipose_detection) for multipose_detection in multipose_detections
                if multipose_detection[4] > self.opt.vis_thresh]

    @staticmethod
    def map_result_to_keypoints(multipose_detection):
        confidence = round(multipose_detection[4], 2)
        bounding_box = np.array(multipose_detection[:4], dtype=np.int32)
        points = np.array(multipose_detection[5:39], dtype=np.int32).reshape(Keypoints.NUMBER_OF_JOINTS, 2)
        return Keypoints.from_detection_result(bounding_box=bounding_box, confidence=confidence, points=points)

    @staticmethod
    def get_center_net_args(model_path):
        return [
            'multi_pose',
            '--load_model',
            model_path
        ]
