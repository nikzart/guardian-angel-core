"""Multi-object tracking using DeepSORT-like algorithm."""

import numpy as np
from typing import List, Optional, Tuple
from collections import deque, defaultdict
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from loguru import logger

from ..utils.types import Detection, BoundingBox


class Track:
    """Represents a single tracked object."""

    def __init__(self, track_id: int, detection: Detection, max_age: int = 30):
        """
        Initialize a track.

        Args:
            track_id: Unique track ID
            detection: Initial detection
            max_age: Maximum frames to keep track without updates
        """
        self.track_id = track_id
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.max_age = max_age

        # Initialize Kalman filter for position tracking
        self.kf = self._init_kalman_filter(detection.bbox)

        # History of positions for trajectory analysis
        self.history = deque(maxlen=30)
        self.history.append(detection.bbox.center)

        # Store class information
        self.class_id = detection.class_id
        self.class_name = detection.class_name

    def _init_kalman_filter(self, bbox: BoundingBox) -> KalmanFilter:
        """Initialize Kalman filter for tracking."""
        kf = KalmanFilter(dim_x=7, dim_z=4)

        # State: [x, y, area, aspect_ratio, dx, dy, darea]
        # Measurement: [x, y, area, aspect_ratio]

        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # x
            [0, 1, 0, 0, 0, 1, 0],  # y
            [0, 0, 1, 0, 0, 0, 1],  # area
            [0, 0, 0, 1, 0, 0, 0],  # aspect ratio
            [0, 0, 0, 0, 1, 0, 0],  # dx
            [0, 0, 0, 0, 0, 1, 0],  # dy
            [0, 0, 0, 0, 0, 0, 1],  # darea
        ])

        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        kf.R *= 10.0  # Measurement uncertainty
        kf.P[4:, 4:] *= 1000.0  # High uncertainty for velocity
        kf.Q[-1, -1] *= 0.01  # Process uncertainty

        # Initialize state
        cx, cy = bbox.center
        area = bbox.area
        aspect_ratio = bbox.width / bbox.height if bbox.height > 0 else 1.0

        kf.x[:4] = [cx, cy, area, aspect_ratio]

        return kf

    def predict(self) -> BoundingBox:
        """
        Predict next position using Kalman filter.

        Returns:
            Predicted bounding box
        """
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

        # Convert state back to bbox
        cx, cy, area, aspect_ratio = self.kf.x[:4]

        # Calculate width and height from area and aspect ratio
        height = np.sqrt(area / aspect_ratio)
        width = aspect_ratio * height

        x1 = cx - width / 2
        y1 = cy - height / 2
        x2 = cx + width / 2
        y2 = cy + height / 2

        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

    def update(self, detection: Detection):
        """
        Update track with new detection.

        Args:
            detection: New detection to update track
        """
        self.time_since_update = 0
        self.hits += 1

        # Update Kalman filter
        bbox = detection.bbox
        cx, cy = bbox.center
        area = bbox.area
        aspect_ratio = bbox.width / bbox.height if bbox.height > 0 else 1.0

        measurement = np.array([cx, cy, area, aspect_ratio])
        self.kf.update(measurement)

        # Update history
        self.history.append((cx, cy))

    def is_confirmed(self, min_hits: int = 3) -> bool:
        """Check if track is confirmed (enough consecutive hits)."""
        return self.hits >= min_hits

    def is_deleted(self) -> bool:
        """Check if track should be deleted."""
        return self.time_since_update > self.max_age

    def get_trajectory(self) -> List[Tuple[float, float]]:
        """Get trajectory history."""
        return list(self.history)


class ObjectTracker:
    """
    Multi-object tracker using Kalman filtering and Hungarian algorithm.

    Similar to DeepSORT but without deep appearance features.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        """
        Initialize tracker.

        Args:
            max_age: Maximum frames to keep track without updates
            min_hits: Minimum hits to confirm a track
            iou_threshold: IOU threshold for matching detections to tracks
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks: List[Track] = []
        self.next_track_id = 0

        logger.info(
            f"Initialized ObjectTracker with max_age={max_age}, "
            f"min_hits={min_hits}, iou_threshold={iou_threshold}"
        )

    def update(self, detections: List[Detection]) -> List[Detection]:
        """
        Update tracker with new detections.

        Args:
            detections: List of new detections

        Returns:
            List of detections with updated track IDs
        """
        # Predict new locations for existing tracks
        predicted_bboxes = []
        for track in self.tracks:
            predicted_bboxes.append(track.predict())

        # Match detections to tracks
        matched, unmatched_detections, unmatched_tracks = self._match(
            detections, predicted_bboxes
        )

        # Update matched tracks
        for detection_idx, track_idx in matched:
            self.tracks[track_idx].update(detections[detection_idx])
            detections[detection_idx].track_id = self.tracks[track_idx].track_id

        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            self._create_track(detections[detection_idx])

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Return detections with confirmed tracks only
        confirmed_detections = []
        for detection in detections:
            if detection.track_id is not None:
                track = self._get_track_by_id(detection.track_id)
                if track and track.is_confirmed(self.min_hits):
                    confirmed_detections.append(detection)

        return confirmed_detections

    def _match(
        self, detections: List[Detection], predicted_bboxes: List[BoundingBox]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to tracks using Hungarian algorithm.

        Returns:
            Tuple of (matched pairs, unmatched detection indices, unmatched track indices)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        # Compute IOU cost matrix
        iou_matrix = np.zeros((len(detections), len(self.tracks)))

        for d_idx, detection in enumerate(detections):
            for t_idx, pred_bbox in enumerate(predicted_bboxes):
                iou_matrix[d_idx, t_idx] = self._iou(detection.bbox, pred_bbox)

        # Use Hungarian algorithm to find optimal assignment
        # Convert IOU to cost (1 - IOU)
        cost_matrix = 1 - iou_matrix

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] >= self.iou_threshold:
                matched.append((row, col))
                unmatched_detections.remove(row)
                unmatched_tracks.remove(col)

        return matched, unmatched_detections, unmatched_tracks

    def _iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate Intersection over Union (IOU) between two bounding boxes."""
        # Calculate intersection area
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        # Calculate union area
        area1 = bbox1.area
        area2 = bbox2.area
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _create_track(self, detection: Detection):
        """Create a new track from detection."""
        track = Track(self.next_track_id, detection, self.max_age)
        self.tracks.append(track)
        detection.track_id = self.next_track_id
        self.next_track_id += 1

    def _get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None

    def get_active_tracks(self) -> List[Track]:
        """Get all active confirmed tracks."""
        return [t for t in self.tracks if t.is_confirmed(self.min_hits)]

    def get_track_count(self) -> int:
        """Get number of active tracks."""
        return len(self.get_active_tracks())

    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.next_track_id = 0
        logger.info("Tracker reset")
