"""Thread-safe configuration manager for runtime updates."""

import yaml
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from copy import deepcopy
from loguru import logger


class ConfigManager:
    """
    Thread-safe configuration manager.

    Handles configuration loading, validation, updates, and backups.
    """

    def __init__(self, config_path: str):
        """
        Initialize config manager.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.lock = threading.RLock()
        self.backup_dir = Path("configs/backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Load initial configuration
        self.reload()

        logger.info(f"ConfigManager initialized with {config_path}")

    def reload(self) -> bool:
        """
        Reload configuration from file.

        Returns:
            True if successful
        """
        try:
            with self.lock:
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"Configuration reloaded from {self.config_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False

    def get(self, key_path: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key path.

        Args:
            key_path: Dot-separated key path (e.g., "system.log_level")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        with self.lock:
            if key_path is None:
                return deepcopy(self.config)

            keys = key_path.split('.')
            value = self.config

            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default

            return deepcopy(value)

    def set(self, key_path: str, value: Any, save: bool = True) -> bool:
        """
        Set configuration value by dot-separated key path.

        Args:
            key_path: Dot-separated key path
            value: Value to set
            save: Whether to save to file immediately

        Returns:
            True if successful
        """
        try:
            with self.lock:
                keys = key_path.split('.')
                config = self.config

                # Navigate to parent
                for key in keys[:-1]:
                    if key not in config:
                        config[key] = {}
                    config = config[key]

                # Set value
                config[keys[-1]] = value

                if save:
                    return self.save()

                return True

        except Exception as e:
            logger.error(f"Failed to set config value: {e}")
            return False

    def update(self, updates: Dict[str, Any], save: bool = True) -> bool:
        """
        Update multiple configuration values.

        Args:
            updates: Dictionary of key paths and values
            save: Whether to save to file immediately

        Returns:
            True if successful
        """
        try:
            with self.lock:
                for key_path, value in updates.items():
                    self.set(key_path, value, save=False)

                if save:
                    return self.save()

                return True

        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False

    def save(self, backup: bool = True) -> bool:
        """
        Save configuration to file.

        Args:
            backup: Whether to create backup before saving

        Returns:
            True if successful
        """
        try:
            with self.lock:
                # Create backup
                if backup and self.config_path.exists():
                    self._create_backup()

                # Write configuration
                with open(self.config_path, 'w') as f:
                    yaml.safe_dump(self.config, f, default_flow_style=False, sort_keys=False)

                logger.info(f"Configuration saved to {self.config_path}")
                return True

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False

    def _create_backup(self):
        """Create a backup of the current configuration."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"config_backup_{timestamp}.yaml"

            with open(self.config_path, 'r') as src:
                with open(backup_path, 'w') as dst:
                    dst.write(src.read())

            logger.info(f"Configuration backup created: {backup_path}")

            # Keep only last 10 backups
            self._cleanup_old_backups(keep=10)

        except Exception as e:
            logger.warning(f"Failed to create config backup: {e}")

    def _cleanup_old_backups(self, keep: int = 10):
        """
        Remove old backup files, keeping only the most recent.

        Args:
            keep: Number of backups to keep
        """
        try:
            backups = sorted(self.backup_dir.glob("config_backup_*.yaml"))

            if len(backups) > keep:
                for backup in backups[:-keep]:
                    backup.unlink()
                    logger.debug(f"Removed old backup: {backup}")

        except Exception as e:
            logger.warning(f"Failed to cleanup old backups: {e}")

    def restore_backup(self, backup_filename: str) -> bool:
        """
        Restore configuration from a backup file.

        Args:
            backup_filename: Name of backup file

        Returns:
            True if successful
        """
        try:
            backup_path = self.backup_dir / backup_filename

            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_filename}")
                return False

            with self.lock:
                # Read backup
                with open(backup_path, 'r') as f:
                    backup_config = yaml.safe_load(f)

                # Validate backup (basic check)
                if not isinstance(backup_config, dict):
                    logger.error("Invalid backup configuration")
                    return False

                # Create backup of current config before restoring
                self._create_backup()

                # Restore
                self.config = backup_config
                self.save(backup=False)

                logger.success(f"Configuration restored from {backup_filename}")
                return True

        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False

    def list_backups(self) -> list:
        """
        List available backup files.

        Returns:
            List of backup filenames with timestamps
        """
        backups = []
        for backup_path in sorted(self.backup_dir.glob("config_backup_*.yaml"), reverse=True):
            backups.append({
                "filename": backup_path.name,
                "created": datetime.fromtimestamp(backup_path.stat().st_mtime).isoformat(),
                "size": backup_path.stat().st_size,
            })
        return backups

    def validate(self, config: Optional[Dict] = None) -> tuple:
        """
        Validate configuration.

        Args:
            config: Configuration to validate (uses current if None)

        Returns:
            Tuple of (is_valid, errors_list)
        """
        config = config or self.config
        errors = []

        # Required top-level keys
        required_keys = ["system", "cameras", "tracking", "pose_model", "alerts"]

        for key in required_keys:
            if key not in config:
                errors.append(f"Missing required key: {key}")

        # Validate cameras
        if "cameras" in config:
            if not isinstance(config["cameras"], list):
                errors.append("'cameras' must be a list")
            else:
                for i, camera in enumerate(config["cameras"]):
                    if "camera_id" not in camera:
                        errors.append(f"Camera {i}: missing 'camera_id'")
                    if "source" not in camera:
                        errors.append(f"Camera {i}: missing 'source'")

        # Validate numeric ranges
        validations = [
            ("tracking.max_age", 1, 100),
            ("tracking.min_hits", 1, 10),
            ("tracking.iou_threshold", 0.0, 1.0),
            ("fall_detection.confidence_threshold", 0.0, 1.0),
            ("bullying_detection.confidence_threshold", 0.0, 1.0),
            ("posh_detection.confidence_threshold", 0.0, 1.0),
        ]

        for key_path, min_val, max_val in validations:
            value = self._get_nested(config, key_path)
            if value is not None:
                if not (min_val <= value <= max_val):
                    errors.append(f"{key_path} must be between {min_val} and {max_val}")

        is_valid = len(errors) == 0
        return is_valid, errors

    def _get_nested(self, config: Dict, key_path: str) -> Any:
        """Get nested value from config dict."""
        keys = key_path.split('.')
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value

    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        with self.lock:
            return deepcopy(self.config)

    def from_dict(self, config: Dict[str, Any], save: bool = True) -> bool:
        """
        Load configuration from dictionary.

        Args:
            config: Configuration dictionary
            save: Whether to save to file

        Returns:
            True if successful
        """
        try:
            # Validate first
            is_valid, errors = self.validate(config)

            if not is_valid:
                logger.error(f"Invalid configuration: {errors}")
                return False

            with self.lock:
                self.config = deepcopy(config)

                if save:
                    return self.save()

                return True

        except Exception as e:
            logger.error(f"Failed to load configuration from dict: {e}")
            return False
