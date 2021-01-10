from abc import abstractmethod
from typing import List

from annotations import Keypoints


class KeypointDetector:

    @abstractmethod
    def detect(self, image_path) -> List[Keypoints]:
        pass
