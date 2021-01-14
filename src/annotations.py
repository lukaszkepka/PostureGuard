from typing import List

import pandas as pd

SUPPORTED_CLASSES = ['not_correct', 'correct']


class Keypoints:
    NUMBER_OF_JOINTS = 17
    ATTRIBUTE_NAMES = [
        'bounding_box_lu_x', 'bounding_box_lu_y', 'bounding_box_rd_x', 'bounding_box_rd_y', 'confidence',
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

    def __init__(self, data_frame: pd.DataFrame):
        if any(data_frame.columns != self.ATTRIBUTE_NAMES):
            raise ValueError('DataFrame columns doesn\'t matches Keypoint attributes')

        if len(data_frame) != 1:
            raise ValueError('DataFrame should have exactly one row')

        self._data_frame = data_frame

    @classmethod
    def from_data_frame(cls, data_frame: pd.DataFrame):
        return cls(data_frame)

    @classmethod
    def from_detection_result(cls, bounding_box, confidence, points):
        if points.shape != (cls.NUMBER_OF_JOINTS, 2):
            raise ValueError(f'Keypoints should be in shape ({cls.NUMBER_OF_JOINTS}, 2)')

        keypoint_coords = list(points.reshape(2 * cls.NUMBER_OF_JOINTS))
        bounding_box_coords = list(bounding_box)
        data_frame_values = tuple(bounding_box_coords + [confidence] + keypoint_coords)

        data_frame = pd.DataFrame.from_records([data_frame_values], columns=cls.ATTRIBUTE_NAMES)
        return cls(data_frame)

    @classmethod
    def get_point_attribute_name(cls, coordinate='x'):
        if coordinate not in ['x', 'y']:
            raise ValueError('coordinate should be x or y')

        return [attribute_name for attribute_name in cls.ATTRIBUTE_NAMES if attribute_name.endswith(coordinate)]

    def _get_keypoint_coords(self, keypoint_name):
        return _get_first_row_value_from_data_frame(self._data_frame, f'{keypoint_name}_x'), \
               _get_first_row_value_from_data_frame(self._data_frame, f'{keypoint_name}_y')

    @property
    def bounding_box(self):
        return [
            self._get_keypoint_coords('bounding_box_lu'),
            self._get_keypoint_coords('bounding_box_rd'),
            _get_first_row_value_from_data_frame(self._data_frame, 'confidence')
        ]

    @property
    def nose(self):
        return self._get_keypoint_coords('nose')

    @property
    def r_eye(self):
        return self._get_keypoint_coords('r_eye')

    @property
    def l_eye(self):
        return self._get_keypoint_coords('l_eye')

    @property
    def r_ear(self):
        return self._get_keypoint_coords('r_ear')

    @property
    def l_ear(self):
        return self._get_keypoint_coords('l_ear')

    @property
    def r_shoulder(self):
        return self._get_keypoint_coords('r_shoulder')

    @property
    def l_shoulder(self):
        return self._get_keypoint_coords('l_shoulder')

    @property
    def r_elbow(self):
        return self._get_keypoint_coords('r_elbow')

    @property
    def l_elbow(self):
        return self._get_keypoint_coords('l_elbow')

    @property
    def r_hand(self):
        return self._get_keypoint_coords('r_hand')

    @property
    def l_hand(self):
        return self._get_keypoint_coords('l_hand')

    @property
    def r_hip(self):
        return self._get_keypoint_coords('r_hip')

    @property
    def l_hip(self):
        return self._get_keypoint_coords('l_hip')

    @property
    def r_knee(self):
        return self._get_keypoint_coords('r_knee')

    @property
    def l_knee(self):
        return self._get_keypoint_coords('l_knee')

    @property
    def r_feet(self):
        return self._get_keypoint_coords('r_feet')

    @property
    def l_feet(self):
        return self._get_keypoint_coords('l_feet')

    def to_dataframe(self) -> pd.DataFrame:
        return self._data_frame

    def to_keypoint_dict(self):
        return {
            'nose': self.nose,
            'r_eye': self.r_eye,
            'l_eye': self.l_eye,
            'r_ear': self.r_ear,
            'l_ear': self.l_ear,
            'r_shoulder': self.r_shoulder,
            'l_shoulder': self.l_shoulder,
            'r_elbow': self.r_elbow,
            'l_elbow': self.l_elbow,
            'r_hand': self.r_hand,
            'l_hand': self.l_hand,
            'r_hip': self.r_hip,
            'l_hip': self.l_hip,
            'r_knee': self.r_knee,
            'l_knee': self.l_knee,
            'r_feet': self.r_feet,
            'l_feet': self.l_feet,
        }


class ImageAnnotation:
    ATTRIBUTE_NAMES = ['file_path', 'original_size_h', 'original_size_w', 'class']

    def __init__(self, data_frame: pd.DataFrame):
        self._data_frame = data_frame.filter(items=self.ATTRIBUTE_NAMES)
        self.keypoints = Keypoints.from_data_frame(data_frame.filter(items=Keypoints.ATTRIBUTE_NAMES))

    @classmethod
    def from_data_frame(cls, data_frame: pd.DataFrame):
        return cls(data_frame)

    @classmethod
    def from_parameters(cls, file_path, original_size, class_name, keypoints: Keypoints):
        data_frame = pd.DataFrame.from_records([
            (file_path, original_size[0], original_size[1], class_name)],
            columns=cls.ATTRIBUTE_NAMES)
        return cls(data_frame.join(keypoints.to_dataframe()))

    @property
    def file_path(self):
        return _get_first_row_value_from_data_frame(self._data_frame, 'file_path')

    def to_dataframe(self) -> pd.DataFrame:
        return self._data_frame.join(self.keypoints.to_dataframe())


def _get_first_row_value_from_data_frame(dataframe, column):
    return dataframe[column].values[0]


def data_frame_to_annotations_list(data_frame: pd.DataFrame) -> List[ImageAnnotation]:
    return [ImageAnnotation.from_data_frame(row[1].to_frame().transpose()) for row in data_frame.iterrows()]
