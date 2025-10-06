"""Pose estimation model wrapper using YOLOv8-pose."""

import torch
import numpy as np
from typing import List, Optional
from ultralytics import YOLO
from loguru import logger

from ..utils.types import Detection, BoundingBox, Pose


class PoseEstimationModel:
    """
    Wrapper for YOLOv8-pose model for human pose estimation.

    Provides edge-optimized inference with ONNX/TensorRT support.
    """

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        device: str = "cuda",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ):
        """
        Initialize pose estimation model.

        Args:
            model_path: Path to YOLOv8-pose model weights
            device: Device to run inference on ('cuda', 'cpu', or 'mps')
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Load model
        logger.info(f"Loading pose model from {model_path} on {self.device}")
        self.model = YOLO(model_path)

        # Warmup
        self._warmup()

        logger.success(f"Pose model loaded successfully on {self.device}")

    def _get_device(self, device: str) -> str:
        """Get appropriate device for inference."""
        if device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif device == "mps" and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _warmup(self, size: tuple = (640, 640)):
        """Warmup model with dummy input."""
        dummy_img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        try:
            self.model.predict(
                dummy_img,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def predict(self, image: np.ndarray) -> List[Detection]:
        """
        Run pose estimation on image.

        Args:
            image: Input image (BGR format)

        Returns:
            List of Detection objects with pose information
        """
        try:
            # Run inference
            results = self.model.predict(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )[0]

            detections = []

            # Process results
            if results.keypoints is not None:
                boxes = results.boxes
                keypoints = results.keypoints

                for i in range(len(boxes)):
                    box = boxes[i]
                    kpts = keypoints[i]

                    # Extract bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())

                    bbox = BoundingBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        confidence=conf,
                    )

                    # Extract keypoints (17 keypoints for COCO format)
                    # Format: (x, y, confidence) for each keypoint
                    kpts_array = kpts.data[0].cpu().numpy()  # Shape: (17, 3)

                    pose = Pose(keypoints=kpts_array, bbox=bbox)

                    detection = Detection(
                        bbox=bbox,
                        class_id=cls,
                        class_name="person",
                        confidence=conf,
                        pose=pose,
                    )

                    detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"Error during pose prediction: {e}")
            return []

    def export_onnx(self, output_path: str, img_size: int = 640):
        """
        Export model to ONNX format for edge deployment.

        Args:
            output_path: Path to save ONNX model
            img_size: Input image size
        """
        try:
            logger.info(f"Exporting model to ONNX: {output_path}")
            self.model.export(
                format="onnx",
                imgsz=img_size,
                simplify=True,
                opset=12,
            )
            logger.success(f"Model exported to ONNX successfully")
        except Exception as e:
            logger.error(f"Failed to export model to ONNX: {e}")

    def export_tensorrt(self, output_path: str, img_size: int = 640):
        """
        Export model to TensorRT format for NVIDIA edge devices.

        Args:
            output_path: Path to save TensorRT model
            img_size: Input image size
        """
        try:
            logger.info(f"Exporting model to TensorRT: {output_path}")
            self.model.export(
                format="engine",
                imgsz=img_size,
                half=True,  # FP16 for better performance
            )
            logger.success(f"Model exported to TensorRT successfully")
        except Exception as e:
            logger.error(f"Failed to export model to TensorRT: {e}")

    @staticmethod
    def get_keypoint_names() -> List[str]:
        """
        Get COCO keypoint names.

        Returns:
            List of keypoint names in order
        """
        return [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]

    def cleanup(self):
        """Cleanup model resources."""
        if hasattr(self, "model"):
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        logger.info("Pose model resources cleaned up")
