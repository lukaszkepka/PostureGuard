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
        bboxes = ret['results'][1]

        detections = [self.map_result_to_keypoints(bbox) for bbox in bboxes if bbox[4] > self.opt.vis_thresh]

        if self.debug:
            image = cv2.imread(image_path)
            for k in detections:
                cv2.circle(image, k.l_knee, 3, (255, 0, 255), -1)
                cv2.circle(image, k.r_knee, 3, (255, 0, 0), -1)

            cv2.imshow('keypoints', image)
            cv2.waitKey(0)

        return detections

    @staticmethod
    def map_result_to_keypoints(bbox):
        points = np.array(bbox[5:39], dtype=np.int32).reshape(Keypoints.NUMBER_OF_JOINTS, 2)
        return Keypoints.from_detection_result(bounding_box=bbox[:4], confidence=bbox[4], points=points)

    @staticmethod
    def get_center_net_args(model_path):
        return [
            'multi_pose',
            '--load_model',
            model_path
        ]
