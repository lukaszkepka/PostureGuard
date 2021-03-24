from abc import abstractmethod
from typing import List

import cv2
import numpy as np

from annotations import Keypoints

DEFAULT_BOUNDING_BOX_COLOR = (0, 255, 255)
DEFAULT_KEYPOINT_COLOR = (0, 255, 0)
DEFAULT_TEXT_COLOR = (255, 0, 0)


class ImageOverlayStep:

    @abstractmethod
    def put_overlay_on_image(self, image: np.ndarray):
        pass


class BoundingBoxImageOverlayStep(ImageOverlayStep):

    def __init__(self, bounding_box, color=DEFAULT_BOUNDING_BOX_COLOR):
        self.color = color
        self.bounding_box = bounding_box

    def put_overlay_on_image(self, image: np.ndarray):
        cv2.rectangle(image, self.bounding_box[0], self.bounding_box[1], self.color)


class KeypointsImageOverlayStep(ImageOverlayStep):
    TEXT_TO_POINT_OFFSET = (0, 10)

    def __init__(self, keypoints: Keypoints, keypoint_color=DEFAULT_KEYPOINT_COLOR,
                 text_color=DEFAULT_BOUNDING_BOX_COLOR):
        self.keypoint_color = keypoint_color
        self.text_color = text_color
        self.keypoints = keypoints

    def _put_keypoint(self, image, localisation, name):
        text_localisation = tuple([localisation[0] + self.TEXT_TO_POINT_OFFSET[0],
                                   localisation[1] + self.TEXT_TO_POINT_OFFSET[1]])

        cv2.circle(image, localisation, 3, self.keypoint_color, -1)
        cv2.putText(image, name, text_localisation, cv2.FONT_HERSHEY_SIMPLEX, 1, self.text_color, 2,
                    cv2.LINE_AA)

    def put_overlay_on_image(self, image: np.ndarray):
        for joint_name, localisation in self.keypoints.to_keypoint_dict().items():
            self._put_keypoint(image, localisation, joint_name)


class ClassNameImageOverlay(ImageOverlayStep):
    TEXT_LOCALIZATION = (35, 45)

    def __init__(self, class_name, class_name_to_color_mapping):
        self.class_name = class_name
        self.class_name_to_color_mapping = class_name_to_color_mapping

        if self.class_name not in self.class_name_to_color_mapping:
            raise ValueError(f'class name {self.class_name} not defined in class_name_to_color_mapping')

    def put_overlay_on_image(self, image: np.ndarray):
        color = self.class_name_to_color_mapping[self.class_name]
        text = self.class_name.upper()

        cv2.putText(image, text, self.TEXT_LOCALIZATION, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                    cv2.LINE_AA)


class TextImageOverlayStep(ImageOverlayStep):
    TEXT_LOCALIZATION = (25, 25)
    NEW_LINE_OFFSET = (0, 25)

    def __init__(self, text_lines, text_color=DEFAULT_TEXT_COLOR):
        self.text_lines = text_lines
        self.text_color = text_color

    def put_overlay_on_image(self, image: np.ndarray):
        text_localisation = self.TEXT_LOCALIZATION

        for text_line in self.text_lines:
            text_localisation = tuple([text_localisation[0] + self.NEW_LINE_OFFSET[0],
                                       text_localisation[1] + self.NEW_LINE_OFFSET[1]])
            cv2.putText(image, text_line, text_localisation, cv2.FONT_HERSHEY_SIMPLEX, 1, self.text_color, 2,
                        cv2.LINE_AA)


class ImageOverlayPipeline:
    def __init__(self, steps: List[ImageOverlayStep]):
        self._steps = steps

    def apply(self, image: np.ndarray):
        for step in self._steps:
            step.put_overlay_on_image(image)
        return image
