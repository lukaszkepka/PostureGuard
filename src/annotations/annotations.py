import numpy as np


class Keypoints:
    NUMBER_OF_JOINTS = 17

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


class ImageAnnotation:
    def __init__(self, file_path, original_size, keypoints: Keypoints):
        self.file_path = file_path
        self.original_size = original_size
        self.keypoints = keypoints
