import math
from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class PreProcessingStep(ABC):

    @abstractmethod
    def run(self, annotations: pd.DataFrame):
        pass


class KeepColumns(PreProcessingStep):

    def __init__(self, column_names):
        self._column_names = column_names

    def run(self, annotations: pd.DataFrame):
        columns_to_remove = set(annotations.columns) - set(self._column_names)
        return annotations.drop(labels=columns_to_remove)


class RemoveColumns(PreProcessingStep):

    def __init__(self, column_names):
        self._column_names = column_names

    def run(self, annotations: pd.DataFrame):
        return annotations.drop(labels=self._column_names)


class NormalizePointCoordinatesToBoundingBox(PreProcessingStep):

    def run(self, annotations: pd.DataFrame):
        x = annotations.filter(items=[
            'bounding_box_lu_x', 'bounding_box_rd_x',
            'nose_x', 'r_eye_x', 'l_eye_x',
            'r_ear_x', 'l_ear_x',
            'r_shoulder_x', 'l_shoulder_x',
            'r_elbow_x', 'l_elbow_x',
            'r_hand_x', 'l_hand_x',
            'r_hip_x', 'l_hip_x',
            'r_knee_x', 'l_knee_x',
            'r_feet_x', 'l_feet_x',
        ])
        y = annotations.filter(items=[
            'bounding_box_lu_y', 'bounding_box_rd_y',
            'nose_y', 'r_eye_y', 'l_eye_y',
            'r_ear_y', 'l_ear_y',
            'r_shoulder_y', 'l_shoulder_y',
            'r_elbow_y', 'l_elbow_y',
            'r_hand_y', 'l_hand_y',
            'r_hip_y', 'l_hip_y',
            'r_knee_y', 'l_knee_y',
            'r_feet_y', 'l_feet_y',
        ])

        # We take min and max points instead of bounding box boundaries
        # because points can be a little bit outside box.
        bounding_box_width = x.max() - x.min()
        bounding_box_height = y.max() - y.min()

        x = (x - annotations['bounding_box_lu_x']) / bounding_box_width
        y = (y - annotations['bounding_box_lu_y']) / bounding_box_height

        return annotations.combine(pd.concat([x, y]), lambda x, y: x if math.isnan(y) else y)


class PreProcessingPipeline:
    def __init__(self, steps: List[PreProcessingStep]):
        self._steps = steps

    def run(self, annotations: pd.DataFrame) -> pd.DataFrame:
        for step in self._steps:
            annotations = step.run(annotations)

        return annotations
