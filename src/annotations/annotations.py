import pandas as pd

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
        self.bounding_box = bounding_box
        self.confidence = confidence
        self.points = points

        if self.points.shape != (self.NUMBER_OF_JOINTS, 2):
            raise ValueError(f'Keypoints should be in shape ({self.NUMBER_OF_JOINTS}, 2)')

        self.nose = (points[0, 0], points[0, 1])

        self.r_eye = (points[1, 0], points[1, 1])
        self.l_eye = (points[2, 0], points[2, 1])

        self.r_ear = (points[3, 0], points[3, 1])
        self.l_ear = (points[4, 0], points[4, 1])

        self.r_shoulder = (points[5, 0], points[5, 1])
        self.l_shoulder = (points[6, 0], points[6, 1])

        self.r_elbow = (points[7, 0], points[7, 1])
        self.l_elbow = (points[8, 0], points[8, 1])

        self.r_hand = (points[9, 0], points[9, 1])
        self.l_hand = (points[10, 0], points[10, 1])

        self.r_hip = (points[11, 0], points[11, 1])
        self.l_hip = (points[12, 0], points[12, 1])

        self.r_knee = (points[13, 0], points[13, 1])
        self.l_knee = (points[14, 0], points[14, 1])

        self.r_feet = (points[15, 0], points[15, 1])
        self.l_feet = (points[16, 0], points[16, 1])

        attributes = [
            self.bounding_box[0],
            self.bounding_box[1],
            self.bounding_box[2],
            self.bounding_box[3],
            self.confidence,
            self.nose[0],
            self.nose[1],
            self.r_eye[0],
            self.r_eye[1],
            self.l_eye[0],
            self.l_eye[1],
            self.r_ear[0],
            self.r_ear[1],
            self.l_ear[0],
            self.l_ear[1],
            self.r_shoulder[0],
            self.r_shoulder[1],
            self.l_shoulder[0],
            self.l_shoulder[1],
            self.r_elbow[0],
            self.r_elbow[1],
            self.l_elbow[0],
            self.l_elbow[1],
            self.r_hand[0],
            self.r_hand[1],
            self.l_hand[0],
            self.l_hand[1],
            self.r_hip[0],
            self.r_hip[1],
            self.l_hip[0],
            self.l_hip[1],
            self.r_knee[0],
            self.r_knee[1],
            self.l_knee[0],
            self.l_knee[1],
            self.r_feet[0],
            self.r_feet[1],
            self.l_feet[0],
            self.l_feet[1],
        ]

        self._data_frame = pd.DataFrame.from_records([tuple(attributes)], columns=self._COLUMNS)

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
