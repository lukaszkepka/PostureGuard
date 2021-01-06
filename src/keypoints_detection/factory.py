from keypoints_detection.KeypointDetector import KeypointDetector
from keypoints_detection.adapters.center_net import CenterNetKeypointDetectorAdapter


def create_keypoint_detector(args, network='center_net') -> KeypointDetector:
    if network == 'center_net':
        return CenterNetKeypointDetectorAdapter(args.model_path)
