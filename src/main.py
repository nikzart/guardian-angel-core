"""Main orchestration module for Guardian Angel system."""

import yaml
import signal
import sys
from pathlib import Path
from threading import Thread, Event
from typing import List, Dict, Optional
from loguru import logger
import uvicorn

from .streaming import CameraStream
from .tracking import ObjectTracker
from .detectors import FallDetector, BullyingDetector, POSHDetector
from .postprocessing import AlertManager
from .utils.types import Frame
from .utils.config_manager import ConfigManager


class GuardianAngelSystem:
    """
    Main orchestration class for the Guardian Angel safety detection system.

    Manages multiple camera streams, detectors, and alert processing.
    """

    def __init__(self, config_path: str = "configs/system_config.yaml"):
        """
        Initialize the system.

        Args:
            config_path: Path to system configuration file
        """
        # Use ConfigManager instead of direct YAML loading
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.to_dict()

        # Setup logging
        self._setup_logging()

        # Initialize components
        self.camera_streams: Dict[str, CameraStream] = {}
        self.trackers: Dict[str, ObjectTracker] = {}
        self.detectors: Dict[str, List] = {}
        self.alert_manager = AlertManager(self.config.get("alerts", {}))

        # Processing threads
        self.processing_threads: List[Thread] = []
        self.stop_event = Event()

        # API server
        self.api_server: Optional[Thread] = None
        self.api_app = None

        # Initialize system components
        self._init_cameras()
        self._init_detectors()

        logger.success("Guardian Angel system initialized")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)

    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = self.config.get("system", {}).get("log_level", "INFO")

        # Remove default logger
        logger.remove()

        # Add console logger
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
        )

        # Add file logger
        logger.add(
            "logs/guardian_angel_{time}.log",
            rotation="100 MB",
            retention="30 days",
            level=log_level,
        )

    def _init_cameras(self):
        """Initialize camera streams."""
        cameras = self.config.get("cameras", [])

        for camera_config in cameras:
            if not camera_config.get("enabled", True):
                continue

            camera_id = camera_config["camera_id"]

            # Create camera stream
            stream = CameraStream(
                camera_id=camera_id,
                source=camera_config["source"],
                target_fps=camera_config.get("target_fps", 30),
            )

            self.camera_streams[camera_id] = stream

            # Create tracker for this camera
            tracking_config = self.config.get("tracking", {})
            tracker = ObjectTracker(
                max_age=tracking_config.get("max_age", 30),
                min_hits=tracking_config.get("min_hits", 3),
                iou_threshold=tracking_config.get("iou_threshold", 0.3),
            )

            self.trackers[camera_id] = tracker

            logger.info(f"Initialized camera: {camera_id} - {camera_config.get('name', 'Unnamed')}")

    def _init_detectors(self):
        """Initialize detection modules for each camera."""
        for camera_id in self.camera_streams.keys():
            detectors = []

            # Fall detector
            if self.config.get("fall_detection", {}).get("enabled", True):
                fall_config = self.config["fall_detection"].copy()
                fall_config["pose_model"] = self.config.get("pose_model", {})
                detectors.append(FallDetector(fall_config, camera_id))

            # Bullying detector
            if self.config.get("bullying_detection", {}).get("enabled", True):
                bullying_config = self.config["bullying_detection"].copy()
                bullying_config["pose_model"] = self.config.get("pose_model", {})
                detectors.append(BullyingDetector(bullying_config, camera_id))

            # POSH detector
            if self.config.get("posh_detection", {}).get("enabled", True):
                posh_config = self.config["posh_detection"].copy()
                posh_config["pose_model"] = self.config.get("pose_model", {})
                detectors.append(POSHDetector(posh_config, camera_id))

            self.detectors[camera_id] = detectors

            logger.info(f"Initialized {len(detectors)} detectors for camera {camera_id}")

    def _process_camera(self, camera_id: str):
        """
        Process frames from a single camera.

        Args:
            camera_id: Camera identifier
        """
        stream = self.camera_streams[camera_id]
        tracker = self.trackers[camera_id]
        detectors = self.detectors[camera_id]

        logger.info(f"Started processing thread for camera {camera_id}")

        frame_count = 0

        while not self.stop_event.is_set():
            try:
                # Read frame
                frame = stream.read(timeout=1.0)

                if frame is None:
                    continue

                frame_count += 1

                # Process with each detector (they run pose estimation internally)
                all_alerts = []

                for detector in detectors:
                    # Preprocess (runs pose estimation)
                    frame = detector.preprocess(frame)

                    # Update tracker with detections
                    frame.detections = tracker.update(frame.detections)

                    # Detect events
                    alerts = detector.detect(frame)
                    all_alerts.extend(alerts)

                # Save alerts
                for alert in all_alerts:
                    self.alert_manager.save_alert(alert)

                # Log progress periodically
                if frame_count % 100 == 0:
                    logger.debug(
                        f"Camera {camera_id}: Processed {frame_count} frames, "
                        f"{tracker.get_track_count()} active tracks"
                    )

            except Exception as e:
                logger.error(f"Error processing camera {camera_id}: {e}")
                continue

        logger.info(f"Processing thread stopped for camera {camera_id}")

    def start(self):
        """Start the system."""
        logger.info("Starting Guardian Angel system...")

        # Start API server if enabled (only if not already running)
        dashboard_config = self.config.get("dashboard", {})
        if dashboard_config.get("enabled", False) and self.api_server is None:
            self._start_api_server(dashboard_config)

        # Start camera streams
        for camera_id, stream in self.camera_streams.items():
            if not stream.start():
                logger.error(f"Failed to start camera {camera_id}")
                continue

        # Start processing threads
        for camera_id in self.camera_streams.keys():
            thread = Thread(
                target=self._process_camera,
                args=(camera_id,),
                daemon=True,
                name=f"Processor-{camera_id}"
            )
            thread.start()
            self.processing_threads.append(thread)

        logger.success(
            f"Guardian Angel system started with {len(self.camera_streams)} cameras"
        )

    def _start_api_server(self, dashboard_config: dict):
        """Start the FastAPI dashboard server."""
        try:
            from .api.app import create_app

            # Create FastAPI app
            self.api_app = create_app(
                config_manager=self.config_manager,
                system_instance=self,
                dashboard_config=dashboard_config
            )

            host = dashboard_config.get("host", "0.0.0.0")
            port = dashboard_config.get("port", 8080)

            # Start API server in separate thread
            def run_server():
                uvicorn.run(
                    self.api_app,
                    host=host,
                    port=port,
                    log_level="info"
                )

            self.api_server = Thread(target=run_server, daemon=True, name="API-Server")
            self.api_server.start()

            logger.success(f"API dashboard server started at http://{host}:{port}")

        except Exception as e:
            logger.error(f"Failed to start API server: {e}")

    def stop(self):
        """Stop the system."""
        logger.info("Stopping Guardian Angel system...")

        # Signal threads to stop
        self.stop_event.set()

        # Stop camera streams
        for camera_id, stream in self.camera_streams.items():
            stream.stop()

        # Wait for processing threads
        for thread in self.processing_threads:
            thread.join(timeout=5)

        # Cleanup detectors
        for camera_id, detectors in self.detectors.items():
            for detector in detectors:
                detector.cleanup()

        logger.success("Guardian Angel system stopped")

    def get_status(self) -> dict:
        """
        Get system status.

        Returns:
            Dictionary with system status information
        """
        status = {
            "running": not self.stop_event.is_set(),
            "cameras": {},
        }

        for camera_id, stream in self.camera_streams.items():
            status["cameras"][camera_id] = stream.get_stats()

        return status

    def run(self):
        """Run the system (blocking)."""
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Start system
        self.start()

        # Wait for stop signal
        try:
            self.stop_event.wait()
        except KeyboardInterrupt:
            pass

        # Stop system
        self.stop()

    def _signal_handler(self, signum, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_event.set()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Guardian Angel School Safety System")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/system_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (shorter duration)"
    )

    args = parser.parse_args()

    # Create system
    system = GuardianAngelSystem(config_path=args.config)

    if args.test:
        logger.info("Running in test mode (60 seconds)")
        system.start()

        import time
        time.sleep(60)

        system.stop()
    else:
        # Run normally
        system.run()


if __name__ == "__main__":
    main()
