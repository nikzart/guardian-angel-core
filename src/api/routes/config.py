"""Configuration management API routes."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
from pydantic import BaseModel
from loguru import logger

from ..auth import get_current_user


router = APIRouter(prefix="/api/config", tags=["configuration"])


# Pydantic models for request/response
class ConfigUpdate(BaseModel):
    """Model for configuration updates."""
    updates: Dict[str, Any]


class ConfigResponse(BaseModel):
    """Model for configuration response."""
    config: Dict[str, Any]


class ValidationResponse(BaseModel):
    """Model for validation response."""
    valid: bool
    errors: List[str] = []


class BackupInfo(BaseModel):
    """Model for backup information."""
    filename: str
    created: str
    size: int


# These will be injected by the app
config_manager = None
system_instance = None


@router.get("/", response_model=ConfigResponse)
async def get_config(user: str = Depends(get_current_user)):
    """
    Get current configuration.

    Returns:
        Complete configuration dictionary
    """
    if config_manager is None:
        raise HTTPException(status_code=500, detail="Config manager not initialized")

    try:
        config = config_manager.to_dict()
        return ConfigResponse(config=config)
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/section/{section}", response_model=Dict[str, Any])
async def get_config_section(section: str, user: str = Depends(get_current_user)):
    """
    Get specific configuration section.

    Args:
        section: Section name (e.g., 'cameras', 'fall_detection')

    Returns:
        Configuration section
    """
    if config_manager is None:
        raise HTTPException(status_code=500, detail="Config manager not initialized")

    try:
        section_config = config_manager.get(section)

        if section_config is None:
            raise HTTPException(status_code=404, detail=f"Section '{section}' not found")

        return section_config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting config section: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/", response_model=Dict[str, str])
async def update_config(
    update: ConfigUpdate,
    user: str = Depends(get_current_user)
):
    """
    Update configuration values.

    Args:
        update: Configuration updates (key paths and values)

    Returns:
        Success message
    """
    if config_manager is None:
        raise HTTPException(status_code=500, detail="Config manager not initialized")

    try:
        # Validate updates before applying
        temp_config = config_manager.to_dict()

        for key_path, value in update.updates.items():
            keys = key_path.split('.')
            config = temp_config

            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]

            config[keys[-1]] = value

        # Validate
        is_valid, errors = config_manager.validate(temp_config)

        if not is_valid:
            return {
                "status": "error",
                "message": "Validation failed",
                "errors": errors
            }

        # Apply updates
        success = config_manager.update(update.updates, save=True)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to update configuration")

        logger.info(f"Configuration updated by {user}: {list(update.updates.keys())}")

        return {
            "status": "success",
            "message": f"Updated {len(update.updates)} configuration values"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate", response_model=ValidationResponse)
async def validate_config(
    config: Dict[str, Any],
    user: str = Depends(get_current_user)
):
    """
    Validate configuration without applying.

    Args:
        config: Configuration to validate

    Returns:
        Validation result
    """
    if config_manager is None:
        raise HTTPException(status_code=500, detail="Config manager not initialized")

    try:
        is_valid, errors = config_manager.validate(config)
        return ValidationResponse(valid=is_valid, errors=errors)
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reload", response_model=Dict[str, str])
async def reload_config(user: str = Depends(get_current_user)):
    """
    Reload configuration from file.

    Returns:
        Success message
    """
    if config_manager is None:
        raise HTTPException(status_code=500, detail="Config manager not initialized")

    try:
        success = config_manager.reload()

        if not success:
            raise HTTPException(status_code=500, detail="Failed to reload configuration")

        logger.info(f"Configuration reloaded by {user}")

        return {
            "status": "success",
            "message": "Configuration reloaded from file"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reloading configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backups", response_model=List[BackupInfo])
async def list_backups(user: str = Depends(get_current_user)):
    """
    List available configuration backups.

    Returns:
        List of backup files
    """
    if config_manager is None:
        raise HTTPException(status_code=500, detail="Config manager not initialized")

    try:
        backups = config_manager.list_backups()
        return backups
    except Exception as e:
        logger.error(f"Error listing backups: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backups/{filename}/restore", response_model=Dict[str, str])
async def restore_backup(
    filename: str,
    user: str = Depends(get_current_user)
):
    """
    Restore configuration from backup.

    Args:
        filename: Backup filename

    Returns:
        Success message
    """
    if config_manager is None:
        raise HTTPException(status_code=500, detail="Config manager not initialized")

    try:
        success = config_manager.restore_backup(filename)

        if not success:
            raise HTTPException(status_code=404, detail="Backup not found or restore failed")

        logger.warning(f"Configuration restored from backup '{filename}' by {user}")

        return {
            "status": "success",
            "message": f"Configuration restored from {filename}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restoring backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schema")
async def get_config_schema(user: str = Depends(get_current_user)):
    """
    Get configuration schema for validation.

    Returns:
        JSON schema describing configuration structure
    """
    # This is a simplified schema - expand as needed
    schema = {
        "type": "object",
        "properties": {
            "system": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "version": {"type": "string"},
                    "log_level": {
                        "type": "string",
                        "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                    }
                }
            },
            "cameras": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["camera_id", "source"],
                    "properties": {
                        "camera_id": {"type": "string"},
                        "name": {"type": "string"},
                        "source": {"type": "string"},
                        "enabled": {"type": "boolean"},
                        "target_fps": {"type": "integer", "minimum": 1, "maximum": 60}
                    }
                }
            },
            "tracking": {
                "type": "object",
                "properties": {
                    "max_age": {"type": "integer", "minimum": 1, "maximum": 100},
                    "min_hits": {"type": "integer", "minimum": 1, "maximum": 10},
                    "iou_threshold": {"type": "number", "minimum": 0, "maximum": 1}
                }
            },
            "fall_detection": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1}
                }
            },
            "bullying_detection": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1}
                }
            },
            "posh_detection": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1}
                }
            }
        }
    }

    return schema
