"""Alert management and persistence."""

import sqlite3
import json
import cv2
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
from loguru import logger
import threading

from ..utils.types import Alert


class AlertManager:
    """
    Manages alert storage, retrieval, and persistence.

    Handles database operations and video clip saving.
    """

    def __init__(self, config: dict):
        """
        Initialize alert manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Database configuration
        self.save_to_db = config.get("save_to_database", True)
        self.db_path = config.get("database_path", "data/alerts.db")

        # Video clip configuration
        self.save_clips = config.get("save_video_clips", True)
        self.clips_path = config.get("video_clips_path", "data/video_clips")
        self.clip_duration = config.get("clip_duration_seconds", 10)

        # Privacy settings
        self.blur_faces = config.get("blur_faces", False)

        # Rate limiting
        self.max_alerts_per_minute = config.get("max_alerts_per_minute", 10)
        self.recent_alerts = []
        self.lock = threading.Lock()

        # Initialize storage
        self._init_storage()

        logger.info(f"Alert manager initialized with database at {self.db_path}")

    def _init_storage(self):
        """Initialize database and file storage."""
        if self.save_to_db:
            # Create database directory
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            # Initialize database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    description TEXT,
                    metadata TEXT,
                    video_clip_path TEXT,
                    reviewed BOOLEAN DEFAULT 0,
                    reviewer_notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON alerts(timestamp)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_camera_id ON alerts(camera_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_alert_type ON alerts(alert_type)
            """)

            conn.commit()
            conn.close()

        if self.save_clips:
            # Create video clips directory
            Path(self.clips_path).mkdir(parents=True, exist_ok=True)

    def save_alert(self, alert: Alert) -> bool:
        """
        Save an alert to storage.

        Args:
            alert: Alert to save

        Returns:
            True if saved successfully
        """
        with self.lock:
            # Check rate limiting
            if not self._check_rate_limit():
                logger.warning("Alert rate limit exceeded, skipping alert")
                return False

            try:
                # Save video clip if available
                if self.save_clips and alert.frame_snapshot is not None:
                    clip_path = self._save_video_clip(alert)
                    alert.video_clip_path = clip_path

                # Save to database
                if self.save_to_db:
                    self._save_to_database(alert)

                # Update rate limiting tracker
                self.recent_alerts.append(datetime.now())

                logger.info(f"Alert {alert.alert_id} saved successfully")
                return True

            except Exception as e:
                logger.error(f"Error saving alert {alert.alert_id}: {e}")
                return False

    def _check_rate_limit(self) -> bool:
        """
        Check if alert rate limit is exceeded.

        Returns:
            True if within rate limit
        """
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)

        # Clean up old alerts
        self.recent_alerts = [t for t in self.recent_alerts if t > one_minute_ago]

        return len(self.recent_alerts) < self.max_alerts_per_minute

    def _save_to_database(self, alert: Alert):
        """Save alert to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO alerts (
                alert_id, alert_type, severity, timestamp, camera_id,
                confidence, description, metadata, video_clip_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.alert_id,
            alert.alert_type.value,
            alert.severity.value,
            alert.timestamp.isoformat(),
            alert.camera_id,
            alert.confidence,
            alert.description,
            json.dumps(alert.metadata),
            alert.video_clip_path,
        ))

        conn.commit()
        conn.close()

    def _save_video_clip(self, alert: Alert) -> Optional[str]:
        """
        Save video clip for alert.

        Args:
            alert: Alert with frame snapshot

        Returns:
            Path to saved video clip
        """
        try:
            # Create filename
            timestamp_str = alert.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"{alert.camera_id}_{alert.alert_type.value}_{timestamp_str}_{alert.alert_id[:8]}.jpg"
            filepath = Path(self.clips_path) / filename

            # Apply privacy filter if enabled
            frame = alert.frame_snapshot.copy()
            if self.blur_faces:
                frame = self._blur_faces(frame)

            # Save frame as image (in production, would save video clip)
            cv2.imwrite(str(filepath), frame)

            return str(filepath)

        except Exception as e:
            logger.error(f"Error saving video clip: {e}")
            return None

    def _blur_faces(self, frame):
        """Apply face blurring for privacy (placeholder implementation)."""
        # In production, would use face detection and apply Gaussian blur
        # For now, just return the frame as-is
        return frame

    def get_alerts(
        self,
        camera_id: Optional[str] = None,
        alert_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        reviewed: Optional[bool] = None,
        limit: int = 100,
    ) -> List[dict]:
        """
        Retrieve alerts from database.

        Args:
            camera_id: Filter by camera ID
            alert_type: Filter by alert type
            start_time: Filter by start timestamp
            end_time: Filter by end timestamp
            reviewed: Filter by review status
            limit: Maximum number of alerts to return

        Returns:
            List of alert dictionaries
        """
        if not self.save_to_db:
            return []

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Build query
        query = "SELECT * FROM alerts WHERE 1=1"
        params = []

        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)

        if alert_type:
            query += " AND alert_type = ?"
            params.append(alert_type)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())

        if reviewed is not None:
            query += " AND reviewed = ?"
            params.append(1 if reviewed else 0)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        alerts = []
        for row in rows:
            alert_dict = dict(row)
            alert_dict["metadata"] = json.loads(alert_dict["metadata"]) if alert_dict["metadata"] else {}
            alerts.append(alert_dict)

        conn.close()
        return alerts

    def mark_reviewed(self, alert_id: str, notes: Optional[str] = None) -> bool:
        """
        Mark an alert as reviewed.

        Args:
            alert_id: Alert ID to mark as reviewed
            notes: Optional reviewer notes

        Returns:
            True if successful
        """
        if not self.save_to_db:
            return False

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE alerts
                SET reviewed = 1, reviewer_notes = ?
                WHERE alert_id = ?
            """, (notes, alert_id))

            conn.commit()
            conn.close()

            logger.info(f"Alert {alert_id} marked as reviewed")
            return True

        except Exception as e:
            logger.error(f"Error marking alert as reviewed: {e}")
            return False

    def get_statistics(self, camera_id: Optional[str] = None, days: int = 7) -> dict:
        """
        Get alert statistics.

        Args:
            camera_id: Filter by camera ID
            days: Number of days to include in statistics

        Returns:
            Dictionary with statistics
        """
        if not self.save_to_db:
            return {}

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        start_time = datetime.now() - timedelta(days=days)

        # Total alerts
        query = "SELECT COUNT(*) FROM alerts WHERE timestamp >= ?"
        params = [start_time.isoformat()]

        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)

        cursor.execute(query, params)
        total_alerts = cursor.fetchone()[0]

        # Alerts by type
        query = "SELECT alert_type, COUNT(*) FROM alerts WHERE timestamp >= ?"
        params = [start_time.isoformat()]

        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)

        query += " GROUP BY alert_type"

        cursor.execute(query, params)
        alerts_by_type = dict(cursor.fetchall())

        # Alerts by severity
        query = "SELECT severity, COUNT(*) FROM alerts WHERE timestamp >= ?"
        params = [start_time.isoformat()]

        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)

        query += " GROUP BY severity"

        cursor.execute(query, params)
        alerts_by_severity = dict(cursor.fetchall())

        conn.close()

        return {
            "total_alerts": total_alerts,
            "alerts_by_type": alerts_by_type,
            "alerts_by_severity": alerts_by_severity,
            "period_days": days,
        }

    def cleanup_old_alerts(self, retention_days: int = 30):
        """
        Clean up old alerts and video clips.

        Args:
            retention_days: Number of days to retain alerts
        """
        if not self.save_to_db:
            return

        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get video clip paths before deletion
            cursor.execute("""
                SELECT video_clip_path FROM alerts
                WHERE timestamp < ? AND video_clip_path IS NOT NULL
            """, (cutoff_date.isoformat(),))

            clip_paths = [row[0] for row in cursor.fetchall()]

            # Delete from database
            cursor.execute("DELETE FROM alerts WHERE timestamp < ?", (cutoff_date.isoformat(),))
            deleted_count = cursor.rowcount

            conn.commit()
            conn.close()

            # Delete video clips
            for clip_path in clip_paths:
                try:
                    Path(clip_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Error deleting clip {clip_path}: {e}")

            logger.info(
                f"Cleaned up {deleted_count} alerts older than {retention_days} days"
            )

        except Exception as e:
            logger.error(f"Error cleaning up old alerts: {e}")
