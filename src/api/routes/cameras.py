"""Camera management API routes."""

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Depends, Response
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any
from pydantic import BaseModel
from loguru import logger
import io

from ..auth import get_current_user


router = APIRouter(prefix="/api/cameras", tags=["cameras"])


# Pydantic models
class CameraCreate(BaseModel):
    """Model for creating a camera."""
    camera_id: str
    name: str
    source: str
    enabled: bool = True
    target_fps: int = 30


class CameraUpdate(BaseModel):
    """Model for updating a camera."""
    name: str = None
    source: str = None
    enabled: bool = None
    target_fps: int = None


class CameraStatus(BaseModel):
    """Model for camera status."""
    camera_id: str
    name: str
    enabled: bool
    is_running: bool
    frame_count: int
    dropped_frames: int
    fps: float = 0.0


class ConnectionTestResult(BaseModel):
    """Model for connection test result."""
    success: bool
    message: str
    width: int = None
    height: int = None
    fps: float = None


# Will be injected by app
config_manager = None
system_instance = None


@router.get("/", response_model=List[Dict[str, Any]])
async def list_cameras(user: str = Depends(get_current_user)):
    """
    List all configured cameras.

    Returns:
        List of camera configurations
    """
    if config_manager is None:
        raise HTTPException(status_code=500, detail="Config manager not initialized")

    try:
        cameras = config_manager.get("cameras", [])
        return cameras
    except Exception as e:
        logger.error(f"Error listing cameras: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{camera_id}", response_model=Dict[str, Any])
async def get_camera(camera_id: str, user: str = Depends(get_current_user)):
    """
    Get specific camera configuration.

    Args:
        camera_id: Camera identifier

    Returns:
        Camera configuration
    """
    if config_manager is None:
        raise HTTPException(status_code=500, detail="Config manager not initialized")

    try:
        cameras = config_manager.get("cameras", [])

        for camera in cameras:
            if camera.get("camera_id") == camera_id:
                return camera

        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting camera: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=Dict[str, str])
async def create_camera(
    camera: CameraCreate,
    user: str = Depends(get_current_user)
):
    """
    Create a new camera configuration.

    Args:
        camera: Camera configuration

    Returns:
        Success message
    """
    if config_manager is None:
        raise HTTPException(status_code=500, detail="Config manager not initialized")

    try:
        cameras = config_manager.get("cameras", [])

        # Check if camera_id already exists
        if any(c.get("camera_id") == camera.camera_id for c in cameras):
            raise HTTPException(
                status_code=400,
                detail=f"Camera '{camera.camera_id}' already exists"
            )

        # Add new camera
        cameras.append(camera.dict())

        # Save to config
        success = config_manager.set("cameras", cameras, save=True)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to save camera")

        logger.info(f"Camera '{camera.camera_id}' created by {user}")

        return {
            "status": "success",
            "message": f"Camera '{camera.camera_id}' created"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating camera: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{camera_id}", response_model=Dict[str, str])
async def update_camera(
    camera_id: str,
    updates: CameraUpdate,
    user: str = Depends(get_current_user)
):
    """
    Update camera configuration.

    Args:
        camera_id: Camera identifier
        updates: Fields to update

    Returns:
        Success message
    """
    if config_manager is None:
        raise HTTPException(status_code=500, detail="Config manager not initialized")

    try:
        cameras = config_manager.get("cameras", [])
        camera_found = False

        for i, camera in enumerate(cameras):
            if camera.get("camera_id") == camera_id:
                camera_found = True

                # Update fields
                update_dict = updates.dict(exclude_unset=True)
                for key, value in update_dict.items():
                    camera[key] = value

                cameras[i] = camera
                break

        if not camera_found:
            raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")

        # Save to config
        success = config_manager.set("cameras", cameras, save=True)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to save camera updates")

        logger.info(f"Camera '{camera_id}' updated by {user}")

        return {
            "status": "success",
            "message": f"Camera '{camera_id}' updated"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating camera: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{camera_id}", response_model=Dict[str, str])
async def delete_camera(
    camera_id: str,
    user: str = Depends(get_current_user)
):
    """
    Delete camera configuration.

    Args:
        camera_id: Camera identifier

    Returns:
        Success message
    """
    if config_manager is None:
        raise HTTPException(status_code=500, detail="Config manager not initialized")

    try:
        cameras = config_manager.get("cameras", [])
        initial_count = len(cameras)

        cameras = [c for c in cameras if c.get("camera_id") != camera_id]

        if len(cameras) == initial_count:
            raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")

        # Save to config
        success = config_manager.set("cameras", cameras, save=True)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to save camera deletion")

        logger.warning(f"Camera '{camera_id}' deleted by {user}")

        return {
            "status": "success",
            "message": f"Camera '{camera_id}' deleted"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting camera: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{camera_id}/status", response_model=CameraStatus)
async def get_camera_status(
    camera_id: str,
    user: str = Depends(get_current_user)
):
    """
    Get camera runtime status.

    Args:
        camera_id: Camera identifier

    Returns:
        Camera status
    """
    if system_instance is None:
        raise HTTPException(status_code=500, detail="System not initialized")

    try:
        # Get camera config
        cameras = config_manager.get("cameras", [])
        camera_config = None

        for camera in cameras:
            if camera.get("camera_id") == camera_id:
                camera_config = camera
                break

        if camera_config is None:
            raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")

        # Get runtime status
        stats = {"is_running": False, "frame_count": 0, "dropped_frames": 0}

        if hasattr(system_instance, 'camera_streams') and camera_id in system_instance.camera_streams:
            stream = system_instance.camera_streams[camera_id]
            stream_stats = stream.get_stats()
            stats.update(stream_stats)

        return CameraStatus(
            camera_id=camera_id,
            name=camera_config.get("name", "Unnamed"),
            enabled=camera_config.get("enabled", True),
            is_running=stats.get("is_running", False),
            frame_count=stats.get("frame_count", 0),
            dropped_frames=stats.get("dropped_frames", 0),
            fps=camera_config.get("target_fps", 30),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting camera status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{camera_id}/test", response_model=ConnectionTestResult)
async def test_camera_connection(
    camera_id: str = None,
    source: str = None,
    user: str = Depends(get_current_user)
):
    """
    Test camera connection.

    Args:
        camera_id: Camera identifier (optional if source provided)
        source: RTSP URL or video source (optional if camera_id provided)

    Returns:
        Connection test result
    """
    try:
        # Get source
        if source is None:
            if camera_id is None:
                raise HTTPException(
                    status_code=400,
                    detail="Either camera_id or source must be provided"
                )

            cameras = config_manager.get("cameras", [])
            camera_config = None

            for camera in cameras:
                if camera.get("camera_id") == camera_id:
                    camera_config = camera
                    break

            if camera_config is None:
                raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")

            source = camera_config.get("source")

        # Test connection
        logger.info(f"Testing connection to {source}")

        # Parse source
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            return ConnectionTestResult(
                success=False,
                message="Failed to open video source"
            )

        # Try to read a frame
        ret, frame = cap.read()

        if not ret or frame is None:
            cap.release()
            return ConnectionTestResult(
                success=False,
                message="Failed to read frame from source"
            )

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        cap.release()

        return ConnectionTestResult(
            success=True,
            message="Connection successful",
            width=width,
            height=height,
            fps=fps if fps > 0 else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing camera connection: {e}")
        return ConnectionTestResult(
            success=False,
            message=f"Error: {str(e)}"
        )


@router.get("/{camera_id}/preview")
async def get_camera_preview(
    camera_id: str,
    user: str = Depends(get_current_user)
):
    """
    Get camera preview frame (JPEG).

    Args:
        camera_id: Camera identifier

    Returns:
        JPEG image
    """
    if system_instance is None:
        raise HTTPException(status_code=500, detail="System not initialized")

    try:
        # Get camera stream
        if not hasattr(system_instance, 'camera_streams') or camera_id not in system_instance.camera_streams:
            raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not running")

        stream = system_instance.camera_streams[camera_id]

        # Read frame
        frame_obj = stream.read(timeout=2.0)

        if frame_obj is None:
            raise HTTPException(status_code=503, detail="No frame available")

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', frame_obj.image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting camera preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))
