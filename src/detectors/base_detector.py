"""Base detector interface for all detection modules."""

from abc import ABC, abstractmethod
from typing import List, Optional
from loguru import logger

from ..utils.types import Frame, Alert, Detection


class BaseDetector(ABC):
    """
    Abstract base class for all detection modules.

    All detectors should inherit from this class and implement the detect method.
    """

    def __init__(self, config: dict, camera_id: str):
        """
        Initialize the detector.

        Args:
            config: Configuration dictionary for the detector
            camera_id: Unique identifier for the camera this detector is monitoring
        """
        self.config = config
        self.camera_id = camera_id
        self.enabled = config.get("enabled", True)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.alert_cooldown = config.get("alert_cooldown_seconds", 5)
        self._last_alert_time = {}

        logger.info(f"Initialized {self.__class__.__name__} for camera {camera_id}")

    @abstractmethod
    def detect(self, frame: Frame) -> List[Alert]:
        """
        Detect events in the given frame.

        Args:
            frame: Frame object containing image and detections

        Returns:
            List of Alert objects if any events are detected, empty list otherwise
        """
        pass

    @abstractmethod
    def preprocess(self, frame: Frame) -> Frame:
        """
        Preprocess the frame before detection.

        Args:
            frame: Input frame

        Returns:
            Preprocessed frame
        """
        pass

    def should_generate_alert(self, alert_type: str) -> bool:
        """
        Check if enough time has passed since the last alert of this type.

        Args:
            alert_type: Type of alert to check

        Returns:
            True if alert should be generated, False otherwise
        """
        import time
        current_time = time.time()

        if alert_type not in self._last_alert_time:
            self._last_alert_time[alert_type] = current_time
            return True

        time_since_last_alert = current_time - self._last_alert_time[alert_type]

        if time_since_last_alert >= self.alert_cooldown:
            self._last_alert_time[alert_type] = current_time
            return True

        return False

    def filter_detections_by_confidence(
        self, detections: List[Detection], threshold: Optional[float] = None
    ) -> List[Detection]:
        """
        Filter detections by confidence threshold.

        Args:
            detections: List of detections to filter
            threshold: Confidence threshold (uses self.confidence_threshold if None)

        Returns:
            Filtered list of detections
        """
        threshold = threshold or self.confidence_threshold
        return [d for d in detections if d.confidence >= threshold]

    def cleanup(self):
        """
        Cleanup resources used by the detector.
        Override this method if your detector needs to release resources.
        """
        logger.info(f"Cleaning up {self.__class__.__name__}")
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
