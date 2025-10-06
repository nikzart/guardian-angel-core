"""Bullying and fight detection module using pose and motion analysis."""

import numpy as np
import uuid
from datetime import datetime
from typing import List, Dict, Tuple
from collections import deque, defaultdict
from loguru import logger

from .base_detector import BaseDetector
from ..utils.types import Frame, Alert, AlertType, AlertSeverity, Detection, Pose
from ..models import PoseEstimationModel


class BullyingDetector(BaseDetector):
    """
    Detects bullying behaviors and fights using multiple indicators:

    1. Aggressive gestures (pushing, hitting motions)
    2. Group cornering (multiple people surrounding one)
    3. Rapid movements indicating physical altercation
    4. Prolonged proximity with aggressive postures
    """

    def __init__(self, config: dict, camera_id: str):
        """
        Initialize bullying detector.

        Args:
            config: Configuration dictionary
            camera_id: Camera identifier
        """
        super().__init__(config, camera_id)

        # Detection thresholds
        self.group_distance_threshold = config.get("group_distance_threshold", 150)  # pixels
        self.group_min_size = config.get("group_min_size", 3)
        self.rapid_movement_threshold = config.get("rapid_movement_threshold", 50)  # pixels/frame
        self.aggressive_pose_threshold = config.get("aggressive_pose_threshold", 0.6)

        # Temporal analysis
        self.history_window = config.get("history_window", 30)  # frames
        self.position_history = defaultdict(lambda: deque(maxlen=self.history_window))
        self.pose_history = defaultdict(lambda: deque(maxlen=self.history_window))

        # Initialize pose model
        model_config = config.get("pose_model", {})
        self.pose_model = PoseEstimationModel(
            model_path=model_config.get("model_path", "yolov8n-pose.pt"),
            device=model_config.get("device", "cuda"),
            conf_threshold=model_config.get("conf_threshold", 0.5),
        )

        logger.info(f"Bullying detector initialized for camera {camera_id}")

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
        frame.detections = detections
        return frame

    def detect(self, frame: Frame) -> List[Alert]:
        """
        Detect bullying behaviors in the given frame.

        Args:
            frame: Frame with pose detections

        Returns:
            List of bullying/fight alerts
        """
        if not self.enabled:
            return []

        alerts = []

        # Update histories
        self._update_histories(frame)

        # Check for various bullying indicators

        # 1. Group cornering detection
        cornering_alerts = self._detect_group_cornering(frame)
        alerts.extend(cornering_alerts)

        # 2. Fight detection (rapid movements)
        fight_alerts = self._detect_fights(frame)
        alerts.extend(fight_alerts)

        # 3. Aggressive gesture detection
        aggression_alerts = self._detect_aggressive_gestures(frame)
        alerts.extend(aggression_alerts)

        return alerts

    def _update_histories(self, frame: Frame):
        """Update position and pose histories for all tracked persons."""
        current_tracks = set()

        for detection in frame.detections:
            if detection.track_id is not None:
                track_id = detection.track_id
                current_tracks.add(track_id)

                # Update position history
                center = detection.bbox.center
                self.position_history[track_id].append((frame.frame_number, center))

                # Update pose history
                if detection.pose:
                    self.pose_history[track_id].append((frame.frame_number, detection.pose))

        # Clean up old tracks
        all_tracks = set(self.position_history.keys())
        inactive_tracks = all_tracks - current_tracks
        for track_id in inactive_tracks:
            if len(self.position_history[track_id]) > 0:
                last_frame = self.position_history[track_id][-1][0]
                if frame.frame_number - last_frame > self.history_window:
                    del self.position_history[track_id]
                    if track_id in self.pose_history:
                        del self.pose_history[track_id]

    def _detect_group_cornering(self, frame: Frame) -> List[Alert]:
        """
        Detect situations where multiple people surround a single individual.

        Returns:
            List of alerts for group cornering situations
        """
        alerts = []

        if len(frame.detections) < self.group_min_size:
            return alerts

        # For each person, check if they are surrounded
        for i, target_detection in enumerate(frame.detections):
            if target_detection.track_id is None:
                continue

            target_center = target_detection.bbox.center
            nearby_people = []

            # Find people near the target
            for j, other_detection in enumerate(frame.detections):
                if i == j or other_detection.track_id is None:
                    continue

                other_center = other_detection.bbox.center
                distance = np.linalg.norm(
                    np.array(target_center) - np.array(other_center)
                )

                if distance < self.group_distance_threshold:
                    nearby_people.append((other_detection, distance))

            # Check if target is surrounded (3+ people nearby from different directions)
            if len(nearby_people) >= self.group_min_size - 1:
                # Calculate angles of surrounding people
                angles = []
                for other_detection, _ in nearby_people:
                    other_center = other_detection.bbox.center
                    dx = other_center[0] - target_center[0]
                    dy = other_center[1] - target_center[1]
                    angle = np.arctan2(dy, dx)
                    angles.append(angle)

                # Check if people are distributed around the target (not all on one side)
                angles = sorted(angles)
                max_gap = 0
                for k in range(len(angles)):
                    gap = angles[(k + 1) % len(angles)] - angles[k]
                    if gap < 0:
                        gap += 2 * np.pi
                    max_gap = max(max_gap, gap)

                # If max gap is less than 270 degrees, people are surrounding from multiple sides
                is_surrounded = max_gap < (3 * np.pi / 2)

                if is_surrounded:
                    if self.should_generate_alert(f"cornering_{target_detection.track_id}"):
                        alert = Alert(
                            alert_id=str(uuid.uuid4()),
                            alert_type=AlertType.GROUP_CORNERING,
                            severity=AlertSeverity.HIGH,
                            timestamp=frame.timestamp,
                            camera_id=self.camera_id,
                            confidence=0.8,
                            description=f"Person (ID: {target_detection.track_id}) surrounded by {len(nearby_people)} individuals",
                            bounding_boxes=[target_detection.bbox] + [d[0].bbox for d in nearby_people],
                            metadata={
                                "target_track_id": target_detection.track_id,
                                "surrounding_track_ids": [d[0].track_id for d in nearby_people],
                                "frame_number": frame.frame_number,
                            },
                            frame_snapshot=frame.image.copy(),
                        )
                        alerts.append(alert)
                        logger.warning(
                            f"Group cornering detected: Person {target_detection.track_id} "
                            f"surrounded by {len(nearby_people)} people"
                        )

        return alerts

    def _detect_fights(self, frame: Frame) -> List[Alert]:
        """
        Detect fights based on rapid movements and close proximity.

        Returns:
            List of fight alerts
        """
        alerts = []

        # Analyze movement patterns
        for detection in frame.detections:
            if detection.track_id is None:
                continue

            track_id = detection.track_id
            history = self.position_history[track_id]

            if len(history) < 5:
                continue

            # Calculate recent movement velocity
            recent_positions = list(history)[-5:]
            velocities = []

            for i in range(1, len(recent_positions)):
                frame_diff = recent_positions[i][0] - recent_positions[i-1][0]
                if frame_diff == 0:
                    continue

                pos_diff = np.array(recent_positions[i][1]) - np.array(recent_positions[i-1][1])
                velocity = np.linalg.norm(pos_diff) / frame_diff
                velocities.append(velocity)

            if not velocities:
                continue

            avg_velocity = np.mean(velocities)

            # Check if movement is rapid (indicating possible fight)
            if avg_velocity > self.rapid_movement_threshold:
                # Check if there are other people nearby with rapid movement
                nearby_rapid_movers = []

                for other_detection in frame.detections:
                    if (other_detection.track_id is None or
                        other_detection.track_id == track_id):
                        continue

                    other_track_id = other_detection.track_id
                    other_history = self.position_history[other_track_id]

                    if len(other_history) < 5:
                        continue

                    # Check proximity
                    distance = np.linalg.norm(
                        np.array(detection.bbox.center) -
                        np.array(other_detection.bbox.center)
                    )

                    if distance < self.group_distance_threshold * 0.7:  # Closer threshold for fights
                        # Check if other person also has rapid movement
                        other_recent = list(other_history)[-5:]
                        other_velocities = []

                        for i in range(1, len(other_recent)):
                            frame_diff = other_recent[i][0] - other_recent[i-1][0]
                            if frame_diff == 0:
                                continue

                            pos_diff = np.array(other_recent[i][1]) - np.array(other_recent[i-1][1])
                            other_velocity = np.linalg.norm(pos_diff) / frame_diff
                            other_velocities.append(other_velocity)

                        if other_velocities and np.mean(other_velocities) > self.rapid_movement_threshold:
                            nearby_rapid_movers.append(other_detection)

                # If multiple people have rapid movement in close proximity, likely a fight
                if len(nearby_rapid_movers) > 0:
                    involved_ids = [track_id] + [d.track_id for d in nearby_rapid_movers]
                    alert_key = f"fight_{'_'.join(map(str, sorted(involved_ids)))}"

                    if self.should_generate_alert(alert_key):
                        alert = Alert(
                            alert_id=str(uuid.uuid4()),
                            alert_type=AlertType.FIGHT_DETECTED,
                            severity=AlertSeverity.CRITICAL,
                            timestamp=frame.timestamp,
                            camera_id=self.camera_id,
                            confidence=0.85,
                            description=f"Possible fight detected involving {len(involved_ids)} individuals with rapid movements",
                            bounding_boxes=[detection.bbox] + [d.bbox for d in nearby_rapid_movers],
                            metadata={
                                "involved_track_ids": involved_ids,
                                "average_velocity": float(avg_velocity),
                                "frame_number": frame.frame_number,
                            },
                            frame_snapshot=frame.image.copy(),
                        )
                        alerts.append(alert)
                        logger.warning(f"Fight detected involving tracks: {involved_ids}")

        return alerts

    def _detect_aggressive_gestures(self, frame: Frame) -> List[Alert]:
        """
        Detect aggressive gestures based on pose analysis.

        Returns:
            List of alerts for aggressive gestures
        """
        alerts = []

        for detection in frame.detections:
            if detection.pose is None or detection.track_id is None:
                continue

            # Analyze pose for aggressive indicators
            is_aggressive, confidence, description = self._analyze_aggressive_pose(
                detection.pose
            )

            if is_aggressive and confidence >= self.aggressive_pose_threshold:
                if self.should_generate_alert(f"aggression_{detection.track_id}"):
                    alert = Alert(
                        alert_id=str(uuid.uuid4()),
                        alert_type=AlertType.AGGRESSIVE_GESTURE,
                        severity=AlertSeverity.MEDIUM,
                        timestamp=frame.timestamp,
                        camera_id=self.camera_id,
                        confidence=confidence,
                        description=description,
                        bounding_boxes=[detection.bbox],
                        metadata={
                            "track_id": detection.track_id,
                            "frame_number": frame.frame_number,
                        },
                        frame_snapshot=frame.image.copy(),
                    )
                    alerts.append(alert)
                    logger.warning(f"Aggressive gesture detected for track {detection.track_id}")

        return alerts

    def _analyze_aggressive_pose(self, pose: Pose) -> Tuple[bool, float, str]:
        """
        Analyze pose for aggressive indicators.

        Args:
            pose: Pose to analyze

        Returns:
            Tuple of (is_aggressive, confidence, description)
        """
        indicators = []
        reasons = []

        # Extract key points
        left_shoulder = pose.get_keypoint(5)
        right_shoulder = pose.get_keypoint(6)
        left_elbow = pose.get_keypoint(7)
        right_elbow = pose.get_keypoint(8)
        left_wrist = pose.get_keypoint(9)
        right_wrist = pose.get_keypoint(10)

        # Check for raised arms (common in aggressive gestures)
        if left_shoulder and left_elbow and left_wrist:
            if all(kp[2] > 0.5 for kp in [left_shoulder, left_elbow, left_wrist]):
                # Check if wrist is above shoulder (raised arm)
                if left_wrist[1] < left_shoulder[1]:
                    # Check if arm is extended
                    shoulder_elbow_dist = np.linalg.norm(
                        np.array(left_shoulder[:2]) - np.array(left_elbow[:2])
                    )
                    elbow_wrist_dist = np.linalg.norm(
                        np.array(left_elbow[:2]) - np.array(left_wrist[:2])
                    )

                    if elbow_wrist_dist > shoulder_elbow_dist * 0.7:
                        indicators.append(0.7)
                        reasons.append("Raised extended left arm")

        # Similar check for right arm
        if right_shoulder and right_elbow and right_wrist:
            if all(kp[2] > 0.5 for kp in [right_shoulder, right_elbow, right_wrist]):
                if right_wrist[1] < right_shoulder[1]:
                    shoulder_elbow_dist = np.linalg.norm(
                        np.array(right_shoulder[:2]) - np.array(right_elbow[:2])
                    )
                    elbow_wrist_dist = np.linalg.norm(
                        np.array(right_elbow[:2]) - np.array(right_wrist[:2])
                    )

                    if elbow_wrist_dist > shoulder_elbow_dist * 0.7:
                        indicators.append(0.7)
                        reasons.append("Raised extended right arm")

        if len(indicators) == 0:
            return False, 0.0, ""

        confidence = np.mean(indicators)
        is_aggressive = confidence >= self.aggressive_pose_threshold

        description = "Aggressive gesture detected: " + "; ".join(reasons) if is_aggressive else ""

        return is_aggressive, confidence, description

    def cleanup(self):
        """Cleanup resources."""
        super().cleanup()
        self.pose_model.cleanup()
        self.position_history.clear()
        self.pose_history.clear()
