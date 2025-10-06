"""Fall detection module using pose estimation."""

import numpy as np
import uuid
from datetime import datetime
from typing import List
from loguru import logger

from .base_detector import BaseDetector
from ..utils.types import Frame, Alert, AlertType, AlertSeverity, Detection, Pose
from ..models import PoseEstimationModel


class FallDetector(BaseDetector):
    """
    Detects falls using pose estimation and geometric analysis.

    Uses multiple heuristics:
    1. Vertical displacement of key points (head, shoulders)
    2. Body orientation (angle from vertical)
    3. Aspect ratio of bounding box
    4. Ground contact detection
    """

    def __init__(self, config: dict, camera_id: str):
        """
        Initialize fall detector.

        Args:
            config: Configuration dictionary
            camera_id: Camera identifier
        """
        super().__init__(config, camera_id)

        # Fall detection parameters
        self.vertical_threshold = config.get("vertical_threshold", 0.6)
        self.angle_threshold = config.get("angle_threshold", 60)  # degrees
        self.aspect_ratio_threshold = config.get("aspect_ratio_threshold", 1.5)
        self.ground_proximity_threshold = config.get("ground_proximity_threshold", 0.9)

        # Temporal smoothing
        self.history_size = config.get("history_size", 5)
        self.pose_history = {}  # track_id -> list of poses

        # Initialize pose model
        model_config = config.get("pose_model", {})
        self.pose_model = PoseEstimationModel(
            model_path=model_config.get("model_path", "yolov8n-pose.pt"),
            device=model_config.get("device", "cuda"),
            conf_threshold=model_config.get("conf_threshold", 0.5),
        )

        logger.info(f"Fall detector initialized for camera {camera_id}")

    def preprocess(self, frame: Frame) -> Frame:
        """
        Preprocess frame by running pose estimation.

        Args:
            frame: Input frame

        Returns:
            Frame with detected poses
        """
        # Run pose estimation
        detections = self.pose_model.predict(frame.image)

        # Add detections to frame
        frame.detections = detections

        return frame

    def detect(self, frame: Frame) -> List[Alert]:
        """
        Detect falls in the given frame.

        Args:
            frame: Frame with pose detections

        Returns:
            List of fall alerts
        """
        if not self.enabled:
            return []

        alerts = []

        for detection in frame.detections:
            if detection.pose is None or detection.track_id is None:
                continue

            # Update pose history
            track_id = detection.track_id
            if track_id not in self.pose_history:
                self.pose_history[track_id] = []

            self.pose_history[track_id].append(detection.pose)

            # Keep only recent history
            if len(self.pose_history[track_id]) > self.history_size:
                self.pose_history[track_id].pop(0)

            # Detect fall
            is_fall, confidence, description = self._detect_fall(
                detection.pose, self.pose_history[track_id], frame.image.shape
            )

            if is_fall and confidence >= self.confidence_threshold:
                if self.should_generate_alert(f"fall_{track_id}"):
                    alert = Alert(
                        alert_id=str(uuid.uuid4()),
                        alert_type=AlertType.FALL_DETECTED,
                        severity=AlertSeverity.CRITICAL,
                        timestamp=frame.timestamp,
                        camera_id=self.camera_id,
                        confidence=confidence,
                        description=description,
                        bounding_boxes=[detection.bbox],
                        metadata={
                            "track_id": track_id,
                            "frame_number": frame.frame_number,
                        },
                        frame_snapshot=frame.image.copy(),
                    )
                    alerts.append(alert)
                    logger.warning(
                        f"Fall detected for track {track_id} with confidence {confidence:.2f}"
                    )

        # Clean up old tracks
        active_track_ids = {d.track_id for d in frame.detections if d.track_id is not None}
        tracks_to_remove = [tid for tid in self.pose_history.keys() if tid not in active_track_ids]
        for tid in tracks_to_remove:
            del self.pose_history[tid]

        return alerts

    def _detect_fall(
        self, pose: Pose, history: List[Pose], image_shape: tuple
    ) -> tuple:
        """
        Detect if a fall has occurred.

        Args:
            pose: Current pose
            history: Historical poses for this track
            image_shape: Shape of the image (height, width, channels)

        Returns:
            Tuple of (is_fall, confidence, description)
        """
        if len(pose.keypoints) < 17:  # COCO format has 17 keypoints
            return False, 0.0, ""

        # Extract key points
        nose = pose.get_keypoint(0)
        left_shoulder = pose.get_keypoint(5)
        right_shoulder = pose.get_keypoint(6)
        left_hip = pose.get_keypoint(11)
        right_hip = pose.get_keypoint(12)
        left_ankle = pose.get_keypoint(15)
        right_ankle = pose.get_keypoint(16)

        # Check if key points are visible
        keypoints_visible = [
            kp for kp in [nose, left_shoulder, right_shoulder, left_hip, right_hip]
            if kp is not None and kp[2] > 0.5
        ]

        if len(keypoints_visible) < 3:
            return False, 0.0, ""

        fall_indicators = []
        reasons = []

        # 1. Check vertical position (head close to ground)
        if nose and nose[2] > 0.5:
            head_y = nose[1]
            image_height = image_shape[0]
            vertical_ratio = head_y / image_height

            if vertical_ratio > self.vertical_threshold:
                fall_indicators.append(0.8)
                reasons.append("Head near ground level")

        # 2. Check body orientation (horizontal posture)
        if left_shoulder and right_shoulder and left_hip and right_hip:
            if all(kp[2] > 0.5 for kp in [left_shoulder, right_shoulder, left_hip, right_hip]):
                # Calculate torso angle from vertical
                shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
                shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
                hip_center_x = (left_hip[0] + right_hip[0]) / 2
                hip_center_y = (left_hip[1] + right_hip[1]) / 2

                # Calculate angle from vertical
                dx = hip_center_x - shoulder_center_x
                dy = hip_center_y - shoulder_center_y

                if abs(dy) > 1:  # Avoid division by zero
                    angle = np.abs(np.degrees(np.arctan(dx / dy)))

                    if angle > self.angle_threshold:
                        fall_indicators.append(0.9)
                        reasons.append(f"Body angle from vertical: {angle:.1f}°")

        # 3. Check bounding box aspect ratio (wide and short)
        bbox = pose.bbox
        aspect_ratio = bbox.width / bbox.height if bbox.height > 0 else 0

        if aspect_ratio > self.aspect_ratio_threshold:
            fall_indicators.append(0.7)
            reasons.append(f"Horizontal body posture (aspect ratio: {aspect_ratio:.2f})")

        # 4. Check for sudden vertical displacement (using history)
        if len(history) >= 3:
            # Compare current nose position with historical average
            historical_nose_y = []
            for hist_pose in history[:-1]:
                hist_nose = hist_pose.get_keypoint(0)
                if hist_nose and hist_nose[2] > 0.5:
                    historical_nose_y.append(hist_nose[1])

            if historical_nose_y and nose and nose[2] > 0.5:
                avg_historical_y = np.mean(historical_nose_y)
                displacement = nose[1] - avg_historical_y

                # Normalize by image height
                normalized_displacement = displacement / image_shape[0]

                if normalized_displacement > 0.15:  # Sudden downward movement
                    fall_indicators.append(0.85)
                    reasons.append("Sudden downward movement detected")

        # Calculate overall confidence
        if len(fall_indicators) == 0:
            return False, 0.0, ""

        confidence = np.mean(fall_indicators)
        is_fall = confidence >= self.confidence_threshold and len(fall_indicators) >= 2

        description = "Fall detected: " + "; ".join(reasons) if is_fall else ""

        return is_fall, confidence, description

    def cleanup(self):
        """Cleanup resources."""
        super().cleanup()
        self.pose_model.cleanup()
        self.pose_history.clear()
