"""Camera stream handler for RTSP and video file inputs."""

import cv2
import time
from datetime import datetime
from threading import Thread, Lock, Event
from queue import Queue, Empty
from typing import Optional, Callable
from loguru import logger
import numpy as np

from ..utils.types import Frame


class CameraStream:
    """
    Handles video streaming from RTSP cameras or video files.

    Supports multi-threaded frame reading with buffering to prevent blocking.
    """

    def __init__(
        self,
        camera_id: str,
        source: str,
        target_fps: int = 30,
        buffer_size: int = 64,
        reconnect_delay: int = 5,
    ):
        """
        Initialize camera stream.

        Args:
            camera_id: Unique identifier for this camera
            source: RTSP URL, video file path, or camera index (0, 1, etc.)
            target_fps: Target frames per second for processing
            buffer_size: Maximum number of frames to buffer
            reconnect_delay: Delay in seconds before reconnecting on failure
        """
        self.camera_id = camera_id
        self.source = source
        self.target_fps = target_fps
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay

        self.frame_queue = Queue(maxsize=buffer_size)
        self.capture: Optional[cv2.VideoCapture] = None
        self.thread: Optional[Thread] = None
        self.lock = Lock()
        self.stopped = Event()

        self.frame_count = 0
        self.dropped_frames = 0
        self.last_frame_time = 0

        logger.info(f"Initialized CameraStream for {camera_id} from source: {source}")

    def start(self) -> bool:
        """
        Start the camera stream in a separate thread.

        Returns:
            True if successfully started, False otherwise
        """
        if not self._connect():
            return False

        self.stopped.clear()
        self.thread = Thread(target=self._reader_thread, daemon=True)
        self.thread.start()

        logger.info(f"Started camera stream for {self.camera_id}")
        return True

    def _connect(self) -> bool:
        """
        Connect to the video source.

        Returns:
            True if successfully connected, False otherwise
        """
        try:
            # Parse source - if it's a digit, convert to int for webcam
            source = self.source
            if isinstance(source, str) and source.isdigit():
                source = int(source)

            self.capture = cv2.VideoCapture(source)

            # Configure capture settings for better performance
            if isinstance(source, str) and source.startswith("rtsp"):
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
                self.capture.set(cv2.CAP_PROP_FPS, self.target_fps)

            if not self.capture.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False

            # Read a test frame
            ret, frame = self.capture.read()
            if not ret or frame is None:
                logger.error(f"Failed to read test frame from {self.source}")
                return False

            logger.success(f"Successfully connected to {self.source}")
            return True

        except Exception as e:
            logger.error(f"Error connecting to {self.source}: {e}")
            return False

    def _reader_thread(self):
        """Background thread that continuously reads frames from the source."""
        consecutive_failures = 0
        max_failures = 100  # Increased from 10 for better resilience

        while not self.stopped.is_set():
            try:
                if self.capture is None or not self.capture.isOpened():
                    logger.warning(f"Connection lost for {self.camera_id}, reconnecting (attempt {consecutive_failures + 1})...")
                    if self._connect():
                        logger.success(f"Reconnected to {self.camera_id} after {consecutive_failures} failures")
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= max_failures:
                            logger.error(
                                f"Max reconnection attempts ({max_failures}) reached for {self.camera_id}"
                            )
                            break
                        # Exponential backoff with max of 60 seconds
                        backoff_delay = min(self.reconnect_delay * (2 ** min(consecutive_failures // 5, 4)), 60)
                        logger.info(f"Waiting {backoff_delay:.1f}s before next reconnection attempt")
                        time.sleep(backoff_delay)
                        continue

                ret, frame = self.capture.read()

                if not ret or frame is None:
                    logger.warning(f"Failed to read frame from {self.camera_id}")
                    consecutive_failures += 1
                    time.sleep(0.1)
                    continue

                consecutive_failures = 0

                # Apply FPS throttling
                current_time = time.time()
                elapsed = current_time - self.last_frame_time
                target_elapsed = 1.0 / self.target_fps

                if elapsed < target_elapsed:
                    time.sleep(target_elapsed - elapsed)
                    continue

                self.last_frame_time = current_time

                # Try to add frame to queue (non-blocking)
                try:
                    self.frame_queue.put_nowait(
                        (frame.copy(), datetime.now(), self.frame_count)
                    )
                    self.frame_count += 1
                except:
                    # Queue is full, drop oldest frame
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(
                            (frame.copy(), datetime.now(), self.frame_count)
                        )
                        self.dropped_frames += 1
                        self.frame_count += 1
                    except:
                        pass

            except Exception as e:
                logger.error(f"Error in reader thread for {self.camera_id}: {e}")
                time.sleep(0.1)

        logger.info(f"Reader thread stopped for {self.camera_id}")

    def read(self, timeout: float = 1.0) -> Optional[Frame]:
        """
        Read the next available frame from the stream.

        Args:
            timeout: Maximum time to wait for a frame (seconds)

        Returns:
            Frame object if available, None otherwise
        """
        try:
            image, timestamp, frame_number = self.frame_queue.get(timeout=timeout)
            return Frame(
                image=image,
                timestamp=timestamp,
                frame_number=frame_number,
                camera_id=self.camera_id,
            )
        except Empty:
            return None

    def stop(self):
        """Stop the camera stream and release resources."""
        logger.info(f"Stopping camera stream for {self.camera_id}")
        self.stopped.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=5)

        if self.capture is not None:
            self.capture.release()

        # Clear the queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break

        logger.info(
            f"Camera stream stopped for {self.camera_id}. "
            f"Total frames: {self.frame_count}, Dropped: {self.dropped_frames}"
        )

    def is_running(self) -> bool:
        """Check if the stream is currently running."""
        return not self.stopped.is_set() and self.thread is not None and self.thread.is_alive()

    def get_stats(self) -> dict:
        """Get stream statistics."""
        return {
            "camera_id": self.camera_id,
            "frame_count": self.frame_count,
            "dropped_frames": self.dropped_frames,
            "queue_size": self.frame_queue.qsize(),
            "is_running": self.is_running(),
        }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
