"""Alert management API routes."""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
from loguru import logger

from ..auth import get_current_user


router = APIRouter(prefix="/api/alerts", tags=["alerts"])


# Pydantic models
class AlertResponse(BaseModel):
    """Model for alert response."""
    alert_id: str
    alert_type: str
    severity: str
    timestamp: str
    camera_id: str
    confidence: float
    description: str
    metadata: Dict[str, Any]
    video_clip_path: Optional[str]
    reviewed: bool


class AlertReview(BaseModel):
    """Model for alert review."""
    notes: Optional[str] = None


class AlertStatistics(BaseModel):
    """Model for alert statistics."""
    total_alerts: int
    alerts_by_type: Dict[str, int]
    alerts_by_severity: Dict[str, int]
    period_days: int


# Will be injected by app
alert_manager = None


@router.get("/", response_model=List[AlertResponse])
async def get_alerts(
    camera_id: Optional[str] = None,
    alert_type: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    reviewed: Optional[bool] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    user: str = Depends(get_current_user)
):
    """
    Get alerts with filtering.

    Args:
        camera_id: Filter by camera ID
        alert_type: Filter by alert type
        start_time: Filter by start timestamp (ISO format)
        end_time: Filter by end timestamp (ISO format)
        reviewed: Filter by review status
        limit: Maximum number of alerts

    Returns:
        List of alerts
    """
    if alert_manager is None:
        raise HTTPException(status_code=500, detail="Alert manager not initialized")

    try:
        # Parse timestamps
        start_dt = None
        end_dt = None

        if start_time:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if end_time:
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))

        # Get alerts
        alerts = alert_manager.get_alerts(
            camera_id=camera_id,
            alert_type=alert_type,
            start_time=start_dt,
            end_time=end_dt,
            reviewed=reviewed,
            limit=limit
        )

        # Convert to response format
        return [AlertResponse(**alert) for alert in alerts]

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid timestamp format: {e}")
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: str,
    user: str = Depends(get_current_user)
):
    """
    Get specific alert by ID.

    Args:
        alert_id: Alert identifier

    Returns:
        Alert details
    """
    if alert_manager is None:
        raise HTTPException(status_code=500, detail="Alert manager not initialized")

    try:
        alerts = alert_manager.get_alerts(limit=10000)

        for alert in alerts:
            if alert["alert_id"] == alert_id:
                return AlertResponse(**alert)

        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{alert_id}/review", response_model=Dict[str, str])
async def review_alert(
    alert_id: str,
    review: AlertReview,
    user: str = Depends(get_current_user)
):
    """
    Mark alert as reviewed.

    Args:
        alert_id: Alert identifier
        review: Review notes

    Returns:
        Success message
    """
    if alert_manager is None:
        raise HTTPException(status_code=500, detail="Alert manager not initialized")

    try:
        success = alert_manager.mark_reviewed(alert_id, review.notes)

        if not success:
            raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")

        logger.info(f"Alert '{alert_id}' reviewed by {user}")

        return {
            "status": "success",
            "message": f"Alert '{alert_id}' marked as reviewed"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reviewing alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics/summary", response_model=AlertStatistics)
async def get_alert_statistics(
    camera_id: Optional[str] = None,
    days: int = Query(default=7, ge=1, le=365),
    user: str = Depends(get_current_user)
):
    """
    Get alert statistics.

    Args:
        camera_id: Filter by camera ID
        days: Number of days to include

    Returns:
        Alert statistics
    """
    if alert_manager is None:
        raise HTTPException(status_code=500, detail="Alert manager not initialized")

    try:
        stats = alert_manager.get_statistics(camera_id=camera_id, days=days)
        return AlertStatistics(**stats)

    except Exception as e:
        logger.error(f"Error getting alert statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recent/count")
async def get_recent_alert_count(
    minutes: int = Query(default=60, ge=1, le=1440),
    user: str = Depends(get_current_user)
):
    """
    Get count of recent alerts.

    Args:
        minutes: Time window in minutes

    Returns:
        Alert counts by type
    """
    if alert_manager is None:
        raise HTTPException(status_code=500, detail="Alert manager not initialized")

    try:
        start_time = datetime.now() - timedelta(minutes=minutes)

        alerts = alert_manager.get_alerts(
            start_time=start_time,
            limit=10000
        )

        # Count by type
        counts = {}
        for alert in alerts:
            alert_type = alert["alert_type"]
            counts[alert_type] = counts.get(alert_type, 0) + 1

        return {
            "total": len(alerts),
            "by_type": counts,
            "time_window_minutes": minutes
        }

    except Exception as e:
        logger.error(f"Error getting recent alert count: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cleanup", response_model=Dict[str, str])
async def cleanup_old_alerts(
    retention_days: int = Query(default=30, ge=1, le=365),
    user: str = Depends(get_current_user)
):
    """
    Clean up old alerts and video clips.

    Args:
        retention_days: Number of days to retain alerts

    Returns:
        Success message
    """
    if alert_manager is None:
        raise HTTPException(status_code=500, detail="Alert manager not initialized")

    try:
        alert_manager.cleanup_old_alerts(retention_days=retention_days)

        logger.info(f"Old alerts cleaned up by {user} (retention: {retention_days} days)")

        return {
            "status": "success",
            "message": f"Alerts older than {retention_days} days have been removed"
        }

    except Exception as e:
        logger.error(f"Error cleaning up alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))
