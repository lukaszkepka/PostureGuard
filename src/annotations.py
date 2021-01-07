import pandas as pd
import numpy as np

SUPPORTED_CLASSES = ['not_correct', 'correct']


class Keypoints:
    NUMBER_OF_JOINTS = 17
    _COLUMNS = [
        'bounding_box_1', 'bounding_box_2', 'bounding_box_3', 'bounding_box_4', 'confidence',
        'nose_x', 'nose_y',
        'r_eye_x', 'r_eye_y',
        'l_eye_x', 'l_eye_y',
        'r_ear_x', 'r_ear_y',
        'l_ear_x', 'l_ear_y',
        'r_shoulder_x', 'r_shoulder_y',
        'l_shoulder_x', 'l_shoulder_y',
        'r_elbow_x', 'r_elbow_y',
        'l_elbow_x', 'l_elbow_y',
        'r_hand_x', 'r_hand_y',
        'l_hand_x', 'l_hand_y',
        'r_hip_x', 'r_hip_y',
        'l_hip_x', 'l_hip_y',
        'r_knee_x', 'r_knee_y',
        'l_knee_x', 'l_knee_y',
        'r_feet_x', 'r_feet_y',
        'l_feet_x', 'l_feet_y']

    def __init__(self, bounding_box, confidence, points):
        if points.shape != (self.NUMBER_OF_JOINTS, 2):
            raise ValueError(f'Keypoints should be in shape ({self.NUMBER_OF_JOINTS}, 2)')

        keypoint_coords = list(points.reshape(2 * self.NUMBER_OF_JOINTS))
        attributes_list = bounding_box + [confidence] + keypoint_coords

        self._data_frame = pd.DataFrame.from_records([tuple(attributes_list)], columns=self._COLUMNS)

    def _get_keypoint_coords(self, keypoint_name):
        return self._data_frame[f'{keypoint_name}_x'], self._data_frame[f'{keypoint_name}_y']

    def bounding_box(self):
        return [
            (self._data_frame['bounding_box_1'], self._data_frame['bounding_box_2']),
            (self._data_frame['bounding_box_3'], self._data_frame['bounding_box_4']),
            self._data_frame['confidence']
        ]

    def nose(self):
        return self._get_keypoint_coords('nose')

    def r_eye(self):
        return self._get_keypoint_coords('r_eye')

    def l_eye(self):
        return self._get_keypoint_coords('l_eye')

    def r_ear(self):
        return self._get_keypoint_coords('r_ear')

    def l_ear(self):
        return self._get_keypoint_coords('l_ear')

    def r_shoulder(self):
        return self._get_keypoint_coords('r_shoulder')

    def l_shoulder(self):
        return self._get_keypoint_coords('l_shoulder')

    def r_elbow(self):
        return self._get_keypoint_coords('r_elbow')

    def l_elbow(self):
        return self._get_keypoint_coords('l_elbow')

    def r_hand(self):
        return self._get_keypoint_coords('r_hand')

    def l_hand(self):
        return self._get_keypoint_coords('l_hand')

    def r_hip(self):
        return self._get_keypoint_coords('r_hip')

    def l_hip(self):
        return self._get_keypoint_coords('l_hip')

    def r_knee(self):
        return self._get_keypoint_coords('r_knee')

    def l_knee(self):
        return self._get_keypoint_coords('l_knee')

    def r_feet(self):
        return self._get_keypoint_coords('r_feet')

    def l_feet(self):
        return self._get_keypoint_coords('l_feet')

    def to_dataframe(self) -> pd.DataFrame:
        return self._data_frame


class ImageAnnotation:
    _COLUMNS = ['file_path', 'original_size_h', 'original_size_w', 'class']

    def __init__(self, file_path, original_size, class_name, keypoints: Keypoints):
        self._data_frame = pd.DataFrame.from_records([
            (file_path, original_size[0], original_size[1], class_name)],
            columns=self._COLUMNS)
        self.keypoints = keypoints

    def to_dataframe(self) -> pd.DataFrame:
        return self._data_frame.join(self.keypoints.to_dataframe())
