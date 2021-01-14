import math
from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from annotations import Keypoints


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
        x = annotations.filter(items=Keypoints.get_point_attribute_name(coordinate='x'))
        y = annotations.filter(items=Keypoints.get_point_attribute_name(coordinate='y'))

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
