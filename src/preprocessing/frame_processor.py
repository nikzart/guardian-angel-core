"""Frame preprocessing pipeline for model input preparation."""

import cv2
import numpy as np
from typing import Tuple, Optional
from loguru import logger

from ..utils.types import Frame


class FrameProcessor:
    """
    Handles frame preprocessing operations.

    Includes resizing, normalization, and format conversion.
    """

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Initialize frame processor.

        Args:
            target_size: Target (width, height) for resizing. None to keep original size.
            normalize: Whether to normalize pixel values
            mean: Mean values for normalization (RGB)
            std: Standard deviation values for normalization (RGB)
        """
        self.target_size = target_size
        self.normalize = normalize
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

        logger.info(
            f"Initialized FrameProcessor with target_size={target_size}, normalize={normalize}"
        )

    def resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio with padding.

        Args:
            image: Input image
            target_size: Target (width, height)

        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size

        # Calculate scaling factor to maintain aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=image.dtype)

        # Calculate padding offsets (center the image)
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2

        # Place resized image in the center
        padded[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized

        return padded

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using mean and std.

        Args:
            image: Input image in [0, 255] range

        Returns:
            Normalized image
        """
        # Convert to float32 and normalize to [0, 1]
        normalized = image.astype(np.float32) / 255.0

        # Apply mean and std normalization
        if self.normalize:
            normalized = (normalized - self.mean) / self.std

        return normalized

    def preprocess(self, frame: Frame) -> Tuple[np.ndarray, Frame]:
        """
        Preprocess a frame for model input.

        Args:
            frame: Input frame

        Returns:
            Tuple of (preprocessed image as numpy array, original frame)
        """
        image = frame.image.copy()

        # Resize if target size is specified
        if self.target_size is not None:
            image = self.resize(image, self.target_size)

        # Normalize
        processed = self.normalize_image(image)

        return processed, frame

    def denormalize(self, image: np.ndarray) -> np.ndarray:
        """
        Denormalize image back to [0, 255] range.

        Args:
            image: Normalized image

        Returns:
            Denormalized image
        """
        if self.normalize:
            image = (image * self.std) + self.mean

        image = (image * 255.0).clip(0, 255).astype(np.uint8)
        return image

    def draw_detections(
        self,
        image: np.ndarray,
        frame: Frame,
        show_poses: bool = True,
        show_boxes: bool = True,
    ) -> np.ndarray:
        """
        Draw detections on image for visualization.

        Args:
            image: Image to draw on
            frame: Frame containing detections
            show_poses: Whether to draw pose keypoints
            show_boxes: Whether to draw bounding boxes

        Returns:
            Image with drawn detections
        """
        output = image.copy()

        for detection in frame.detections:
            if show_boxes and detection.bbox:
                bbox = detection.bbox
                x1, y1 = int(bbox.x1), int(bbox.y1)
                x2, y2 = int(bbox.x2), int(bbox.y2)

                # Draw bounding box
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                label = f"{detection.class_name}: {detection.confidence:.2f}"
                if detection.track_id is not None:
                    label = f"ID{detection.track_id} {label}"

                cv2.putText(
                    output,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            if show_poses and detection.pose:
                self._draw_pose(output, detection.pose)

        return output

    def _draw_pose(self, image: np.ndarray, pose) -> None:
        """
        Draw pose keypoints and skeleton on image.

        Args:
            image: Image to draw on
            pose: Pose object with keypoints
        """
        # COCO keypoint indices for skeleton connections
        skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # Head
            [6, 12], [7, 13], [6, 7],  # Torso
            [6, 8], [7, 9], [8, 10], [9, 11],  # Arms
            [12, 14], [13, 15], [14, 16], [15, 17],  # Legs (alternative)
        ]

        # Draw keypoints
        for i, (x, y, conf) in enumerate(pose.keypoints):
            if conf > 0.5:  # Only draw confident keypoints
                cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), -1)

        # Draw skeleton
        for connection in skeleton:
            pt1_idx, pt2_idx = connection
            if pt1_idx < len(pose.keypoints) and pt2_idx < len(pose.keypoints):
                x1, y1, conf1 = pose.keypoints[pt1_idx]
                x2, y2, conf2 = pose.keypoints[pt2_idx]

                if conf1 > 0.5 and conf2 > 0.5:
                    cv2.line(
                        image,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (255, 0, 0),
                        2,
                    )

    def apply_privacy_filter(self, image: np.ndarray, frame: Frame) -> np.ndarray:
        """
        Apply privacy filter by blurring faces and keeping only pose skeletons.

        Args:
            image: Input image
            frame: Frame with detections

        Returns:
            Privacy-filtered image
        """
        output = np.zeros_like(image)

        # Draw only pose skeletons, no faces or identifying features
        for detection in frame.detections:
            if detection.pose:
                self._draw_pose(output, detection.pose)

        return output
