"""System monitoring and status API routes."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any
from pydantic import BaseModel
from loguru import logger
import psutil
import platform

from ..auth import get_current_user


router = APIRouter(prefix="/api/system", tags=["system"])


# Pydantic models
class SystemInfo(BaseModel):
    """Model for system information."""
    name: str
    version: str
    platform: str
    python_version: str


class SystemStatus(BaseModel):
    """Model for system status."""
    running: bool
    cameras: Dict[str, Any]
    cpu_percent: float
    memory_percent: float
    disk_percent: float


class CameraStreamStatus(BaseModel):
    """Model for camera stream status."""
    camera_id: str
    is_running: bool
    frame_count: int
    dropped_frames: int
    queue_size: int
    fps: float


# Will be injected by app
config_manager = None
system_instance = None


@router.get("/info", response_model=SystemInfo)
async def get_system_info(user: str = Depends(get_current_user)):
    """
    Get system information.

    Returns:
        System info
    """
    if config_manager is None:
        raise HTTPException(status_code=500, detail="Config manager not initialized")

    try:
        system_config = config_manager.get("system", {})

        return SystemInfo(
            name=system_config.get("name", "Guardian Angel"),
            version=system_config.get("version", "1.0.0"),
            platform=platform.system(),
            python_version=platform.python_version()
        )

    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=SystemStatus)
async def get_system_status(user: str = Depends(get_current_user)):
    """
    Get system runtime status.

    Returns:
        System status including resource usage
    """
    try:
        # Get system status
        status = {
            "running": False,
            "cameras": {}
        }

        if system_instance is not None:
            if hasattr(system_instance, 'get_status'):
                status = system_instance.get_status()

        # Get resource usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return SystemStatus(
            running=status.get("running", False),
            cameras=status.get("cameras", {}),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent
        )

    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cameras/status", response_model=List[CameraStreamStatus])
async def get_cameras_status(user: str = Depends(get_current_user)):
    """
    Get status of all camera streams.

    Returns:
        List of camera stream statuses
    """
    if system_instance is None:
        raise HTTPException(status_code=500, detail="System not initialized")

    try:
        statuses = []

        if hasattr(system_instance, 'camera_streams'):
            for camera_id, stream in system_instance.camera_streams.items():
                stats = stream.get_stats()

                statuses.append(CameraStreamStatus(
                    camera_id=camera_id,
                    is_running=stats.get("is_running", False),
                    frame_count=stats.get("frame_count", 0),
                    dropped_frames=stats.get("dropped_frames", 0),
                    queue_size=stats.get("queue_size", 0),
                    fps=30.0  # Get from config
                ))

        return statuses

    except Exception as e:
        logger.error(f"Error getting cameras status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restart", response_model=Dict[str, str])
async def restart_system(user: str = Depends(get_current_user)):
    """
    Restart the system (stop and reinitialize all components).

    Returns:
        Success message
    """
    if system_instance is None:
        raise HTTPException(status_code=500, detail="System not initialized")

    try:
        logger.warning(f"System restart requested by {user}")

        # Stop system completely
        if hasattr(system_instance, 'stop'):
            system_instance.stop()

        # Reload config
        if config_manager is not None:
            config_manager.reload()
            system_instance.config = config_manager.to_dict()

        # Reinitialize cameras
        system_instance._init_cameras()

        # Reinitialize detectors
        system_instance._init_detectors()

        # Start system with fresh components
        if hasattr(system_instance, 'start'):
            system_instance.start()

        logger.success("System restarted successfully with reinitialized components")

        return {
            "status": "success",
            "message": "System restarted successfully"
        }

    except Exception as e:
        logger.error(f"Error restarting system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop", response_model=Dict[str, str])
async def stop_system(user: str = Depends(get_current_user)):
    """
    Stop the system.

    Returns:
        Success message
    """
    if system_instance is None:
        raise HTTPException(status_code=500, detail="System not initialized")

    try:
        logger.warning(f"System stop requested by {user}")

        if hasattr(system_instance, 'stop'):
            system_instance.stop()

        return {
            "status": "success",
            "message": "System stopped successfully"
        }

    except Exception as e:
        logger.error(f"Error stopping system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start", response_model=Dict[str, str])
async def start_system(user: str = Depends(get_current_user)):
    """
    Start the system.

    Returns:
        Success message
    """
    if system_instance is None:
        raise HTTPException(status_code=500, detail="System not initialized")

    try:
        logger.info(f"System start requested by {user}")

        if hasattr(system_instance, 'start'):
            system_instance.start()

        return {
            "status": "success",
            "message": "System started successfully"
        }

    except Exception as e:
        logger.error(f"Error starting system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check endpoint (no auth required).

    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "timestamp": logger._core.now().isoformat()
    }
