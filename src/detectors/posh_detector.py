"""POSH (Prevention of Sexual Harassment) behavioral anomaly detection module."""

import numpy as np
import uuid
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
from loguru import logger

from .base_detector import BaseDetector
from ..utils.types import Frame, Alert, AlertType, AlertSeverity, Detection
from ..models import PoseEstimationModel


class Zone:
    """Represents a spatial zone in the scene."""

    def __init__(self, name: str, polygon: List[Tuple[float, float]], is_restricted: bool = False):
        """
        Initialize a zone.

        Args:
            name: Zone name
            polygon: List of (x, y) points defining the zone boundary
            is_restricted: Whether this is a restricted/isolated area
        """
        self.name = name
        self.polygon = np.array(polygon)
        self.is_restricted = is_restricted

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """
        Check if a point is inside the zone using ray casting algorithm.

        Args:
            point: (x, y) coordinate

        Returns:
            True if point is inside zone
        """
        x, y = point
        n = len(self.polygon)
        inside = False

        p1x, p1y = self.polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = self.polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside


class POSHDetector(BaseDetector):
    """
    Detects behavioral anomalies that may indicate POSH violations.

    Monitors:
    1. Individuals being led to isolated/restricted areas
    2. Unusual proximity patterns between staff and students
    3. Isolated interactions in low-traffic zones
    4. Prolonged one-on-one interactions in isolated areas
    """

    def __init__(self, config: dict, camera_id: str):
        """
        Initialize POSH detector.

        Args:
            config: Configuration dictionary
            camera_id: Camera identifier
        """
        super().__init__(config, camera_id)

        # Detection parameters
        self.isolation_distance_threshold = config.get("isolation_distance_threshold", 200)  # pixels
        self.proximity_threshold = config.get("proximity_threshold", 100)  # pixels
        self.prolonged_interaction_frames = config.get("prolonged_interaction_frames", 150)  # ~5 sec at 30fps
        self.leading_trajectory_threshold = config.get("leading_trajectory_threshold", 100)  # pixels

        # Zone configuration
        self.zones = self._load_zones(config.get("zones", []))

        # Tracking histories
        self.position_history = defaultdict(lambda: deque(maxlen=100))
        self.interaction_history = defaultdict(lambda: {"start_frame": None, "duration": 0, "location": None})
        self.zone_entry_history = defaultdict(list)

        # Initialize pose model
        model_config = config.get("pose_model", {})
        self.pose_model = PoseEstimationModel(
            model_path=model_config.get("model_path", "yolov8n-pose.pt"),
            device=model_config.get("device", "cuda"),
            conf_threshold=model_config.get("conf_threshold", 0.5),
        )

        logger.info(f"POSH detector initialized for camera {camera_id} with {len(self.zones)} zones")

    def _load_zones(self, zones_config: List[Dict]) -> List[Zone]:
        """Load zone definitions from configuration."""
        zones = []
        for zone_config in zones_config:
            zone = Zone(
                name=zone_config["name"],
                polygon=zone_config["polygon"],
                is_restricted=zone_config.get("is_restricted", False),
            )
            zones.append(zone)
        return zones

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
        Detect POSH-related behavioral anomalies.

        Args:
            frame: Frame with pose detections

        Returns:
            List of POSH-related alerts
        """
        if not self.enabled:
            return []

        alerts = []

        # Update position histories
        self._update_histories(frame)

        # 1. Detect individuals being led to isolated areas
        leading_alerts = self._detect_leading_to_isolated_area(frame)
        alerts.extend(leading_alerts)

        # 2. Detect isolated interactions
        isolation_alerts = self._detect_isolated_interactions(frame)
        alerts.extend(isolation_alerts)

        # 3. Detect entry into restricted zones
        restricted_alerts = self._detect_restricted_zone_entry(frame)
        alerts.extend(restricted_alerts)

        return alerts

    def _update_histories(self, frame: Frame):
        """Update position and interaction histories."""
        current_tracks = set()

        for detection in frame.detections:
            if detection.track_id is not None:
                track_id = detection.track_id
                current_tracks.add(track_id)

                # Update position history
                center = detection.bbox.center
                self.position_history[track_id].append({
                    "frame": frame.frame_number,
                    "position": center,
                    "timestamp": frame.timestamp,
                })

        # Clean up old tracks
        all_tracks = set(self.position_history.keys())
        inactive_tracks = all_tracks - current_tracks

        for track_id in list(inactive_tracks):
            if len(self.position_history[track_id]) > 0:
                last_frame = self.position_history[track_id][-1]["frame"]
                if frame.frame_number - last_frame > 100:
                    del self.position_history[track_id]
                    if track_id in self.interaction_history:
                        del self.interaction_history[track_id]

    def _detect_leading_to_isolated_area(self, frame: Frame) -> List[Alert]:
        """
        Detect cases where one person appears to be leading another to an isolated area.

        Returns:
            List of alerts
        """
        alerts = []

        if len(frame.detections) < 2:
            return alerts

        # Analyze pairs of people
        for i, person1 in enumerate(frame.detections):
            if person1.track_id is None:
                continue

            for j, person2 in enumerate(frame.detections[i+1:], start=i+1):
                if person2.track_id is None:
                    continue

                # Check if they are close together
                distance = np.linalg.norm(
                    np.array(person1.bbox.center) - np.array(person2.bbox.center)
                )

                if distance < self.proximity_threshold * 1.5:
                    # Analyze their trajectories
                    is_leading, confidence = self._analyze_leading_behavior(
                        person1.track_id, person2.track_id, frame
                    )

                    if is_leading:
                        # Check if heading towards isolated area
                        current_isolation = self._calculate_isolation_score(
                            person1.bbox.center, frame.detections
                        )

                        # Check historical isolation
                        if len(self.position_history[person1.track_id]) >= 20:
                            past_positions = list(self.position_history[person1.track_id])[-20:-10]
                            if past_positions:
                                avg_past_pos = np.mean([p["position"] for p in past_positions], axis=0)
                                past_isolation = self._calculate_isolation_score(
                                    tuple(avg_past_pos), frame.detections
                                )

                                # If moving from less isolated to more isolated area
                                if current_isolation > past_isolation * 1.5:
                                    alert_key = f"leading_{person1.track_id}_{person2.track_id}"

                                    if self.should_generate_alert(alert_key):
                                        alert = Alert(
                                            alert_id=str(uuid.uuid4()),
                                            alert_type=AlertType.ISOLATED_AREA_CONCERN,
                                            severity=AlertSeverity.HIGH,
                                            timestamp=frame.timestamp,
                                            camera_id=self.camera_id,
                                            confidence=confidence,
                                            description=f"Potential concern: One individual appears to be leading another to an isolated area",
                                            bounding_boxes=[person1.bbox, person2.bbox],
                                            metadata={
                                                "track_ids": [person1.track_id, person2.track_id],
                                                "isolation_score": float(current_isolation),
                                                "frame_number": frame.frame_number,
                                            },
                                            frame_snapshot=frame.image.copy(),
                                        )
                                        alerts.append(alert)
                                        logger.warning(
                                            f"Leading to isolated area detected: "
                                            f"tracks {person1.track_id} and {person2.track_id}"
                                        )

        return alerts

    def _detect_isolated_interactions(self, frame: Frame) -> List[Alert]:
        """
        Detect prolonged one-on-one interactions in isolated areas.

        Returns:
            List of alerts
        """
        alerts = []

        if len(frame.detections) < 2:
            # Reset all interaction histories if alone
            for track_id in self.interaction_history:
                self.interaction_history[track_id]["start_frame"] = None
                self.interaction_history[track_id]["duration"] = 0
            return alerts

        # Find pairs in close proximity
        for i, person1 in enumerate(frame.detections):
            if person1.track_id is None:
                continue

            for j, person2 in enumerate(frame.detections[i+1:], start=i+1):
                if person2.track_id is None:
                    continue

                distance = np.linalg.norm(
                    np.array(person1.bbox.center) - np.array(person2.bbox.center)
                )

                if distance < self.proximity_threshold:
                    # Check if they are isolated from others
                    isolation_score = self._calculate_isolation_score(
                        person1.bbox.center, frame.detections, exclude_ids=[person1.track_id, person2.track_id]
                    )

                    if isolation_score > 0.7:  # High isolation
                        # Track duration of interaction
                        pair_key = tuple(sorted([person1.track_id, person2.track_id]))

                        if self.interaction_history[pair_key]["start_frame"] is None:
                            self.interaction_history[pair_key]["start_frame"] = frame.frame_number
                            self.interaction_history[pair_key]["location"] = person1.bbox.center
                        else:
                            duration = frame.frame_number - self.interaction_history[pair_key]["start_frame"]
                            self.interaction_history[pair_key]["duration"] = duration

                            # Alert if interaction is prolonged
                            if duration >= self.prolonged_interaction_frames:
                                alert_key = f"isolated_interaction_{pair_key[0]}_{pair_key[1]}"

                                if self.should_generate_alert(alert_key):
                                    alert = Alert(
                                        alert_id=str(uuid.uuid4()),
                                        alert_type=AlertType.POSH_VIOLATION,
                                        severity=AlertSeverity.MEDIUM,
                                        timestamp=frame.timestamp,
                                        camera_id=self.camera_id,
                                        confidence=0.75,
                                        description=f"Prolonged isolated interaction detected (~{duration/30:.1f} seconds)",
                                        bounding_boxes=[person1.bbox, person2.bbox],
                                        metadata={
                                            "track_ids": list(pair_key),
                                            "duration_frames": duration,
                                            "isolation_score": float(isolation_score),
                                            "frame_number": frame.frame_number,
                                        },
                                        frame_snapshot=frame.image.copy(),
                                    )
                                    alerts.append(alert)
                                    logger.warning(
                                        f"Prolonged isolated interaction: tracks {pair_key}, "
                                        f"duration {duration} frames"
                                    )
                else:
                    # Reset if they moved apart
                    pair_key = tuple(sorted([person1.track_id, person2.track_id]))
                    if pair_key in self.interaction_history:
                        self.interaction_history[pair_key]["start_frame"] = None
                        self.interaction_history[pair_key]["duration"] = 0

        return alerts

    def _detect_restricted_zone_entry(self, frame: Frame) -> List[Alert]:
        """
        Detect entry into restricted/isolated zones.

        Returns:
            List of alerts
        """
        alerts = []

        for detection in frame.detections:
            if detection.track_id is None:
                continue

            center = detection.bbox.center

            # Check which zones the person is in
            for zone in self.zones:
                if zone.is_restricted and zone.contains_point(center):
                    alert_key = f"restricted_zone_{detection.track_id}_{zone.name}"

                    if self.should_generate_alert(alert_key):
                        alert = Alert(
                            alert_id=str(uuid.uuid4()),
                            alert_type=AlertType.POSH_VIOLATION,
                            severity=AlertSeverity.MEDIUM,
                            timestamp=frame.timestamp,
                            camera_id=self.camera_id,
                            confidence=0.9,
                            description=f"Person entered restricted zone: {zone.name}",
                            bounding_boxes=[detection.bbox],
                            metadata={
                                "track_id": detection.track_id,
                                "zone_name": zone.name,
                                "frame_number": frame.frame_number,
                            },
                            frame_snapshot=frame.image.copy(),
                        )
                        alerts.append(alert)
                        logger.warning(
                            f"Restricted zone entry: track {detection.track_id} "
                            f"entered {zone.name}"
                        )

        return alerts

    def _analyze_leading_behavior(
        self, leader_id: int, follower_id: int, frame: Frame
    ) -> Tuple[bool, float]:
        """
        Analyze if one person is leading another based on trajectories.

        Returns:
            Tuple of (is_leading, confidence)
        """
        leader_history = self.position_history[leader_id]
        follower_history = self.position_history[follower_id]

        if len(leader_history) < 10 or len(follower_history) < 10:
            return False, 0.0

        # Get recent positions
        leader_positions = [p["position"] for p in list(leader_history)[-10:]]
        follower_positions = [p["position"] for p in list(follower_history)[-10:]]

        # Calculate movement directions
        leader_direction = np.array(leader_positions[-1]) - np.array(leader_positions[0])
        follower_direction = np.array(follower_positions[-1]) - np.array(follower_positions[0])

        # Check if moving in similar direction
        if np.linalg.norm(leader_direction) > 5 and np.linalg.norm(follower_direction) > 5:
            # Normalize and compute similarity
            leader_dir_norm = leader_direction / np.linalg.norm(leader_direction)
            follower_dir_norm = follower_direction / np.linalg.norm(follower_direction)

            similarity = np.dot(leader_dir_norm, follower_dir_norm)

            # Check if leader is ahead in the direction of movement
            current_leader_pos = np.array(leader_positions[-1])
            current_follower_pos = np.array(follower_positions[-1])

            relative_pos = current_leader_pos - current_follower_pos
            is_ahead = np.dot(relative_pos, leader_dir_norm) > 0

            if similarity > 0.7 and is_ahead:
                return True, 0.8

        return False, 0.0

    def _calculate_isolation_score(
        self,
        position: Tuple[float, float],
        all_detections: List[Detection],
        exclude_ids: Optional[List[int]] = None,
    ) -> float:
        """
        Calculate how isolated a position is from other people.

        Args:
            position: (x, y) position to check
            all_detections: All detections in frame
            exclude_ids: Track IDs to exclude from calculation

        Returns:
            Isolation score between 0 (not isolated) and 1 (very isolated)
        """
        exclude_ids = exclude_ids or []

        distances = []
        for detection in all_detections:
            if detection.track_id in exclude_ids:
                continue

            distance = np.linalg.norm(
                np.array(position) - np.array(detection.bbox.center)
            )
            distances.append(distance)

        if not distances:
            return 1.0  # Completely isolated

        # Calculate score based on nearest neighbors
        min_distance = min(distances)

        # Normalize to 0-1 range (higher = more isolated)
        # Using isolation_distance_threshold as reference
        score = min(min_distance / self.isolation_distance_threshold, 1.0)

        return score

    def cleanup(self):
        """Cleanup resources."""
        super().cleanup()
        self.pose_model.cleanup()
        self.position_history.clear()
        self.interaction_history.clear()
        self.zone_entry_history.clear()
