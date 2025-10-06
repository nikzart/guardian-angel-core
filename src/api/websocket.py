"""WebSocket handler for real-time updates."""

from fastapi import WebSocket, WebSocketDisconnect, status
from typing import List, Set
from loguru import logger
import json
import asyncio
from datetime import datetime
import base64


class ConnectionManager:
    """
    Manages WebSocket connections for real-time updates.

    Handles alert broadcasting and system status updates.
    """

    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Set[WebSocket] = set()
        self.alert_queue: asyncio.Queue = asyncio.Queue()

    async def connect(self, websocket: WebSocket):
        """
        Accept and register a new WebSocket connection.

        Args:
            websocket: WebSocket connection
        """
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """
        Remove a WebSocket connection.

        Args:
            websocket: WebSocket connection
        """
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """
        Send a message to a specific client.

        Args:
            message: Message to send
            websocket: Target WebSocket connection
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        """
        Broadcast a message to all connected clients.

        Args:
            message: Message to broadcast
        """
        disconnected = set()

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_alert(self, alert: dict):
        """
        Broadcast an alert to all connected clients.

        Args:
            alert: Alert dictionary
        """
        message = {
            "type": "alert",
            "timestamp": datetime.now().isoformat(),
            "data": alert
        }

        await self.broadcast(message)

    async def broadcast_system_status(self, status: dict):
        """
        Broadcast system status update.

        Args:
            status: System status dictionary
        """
        message = {
            "type": "system_status",
            "timestamp": datetime.now().isoformat(),
            "data": status
        }

        await self.broadcast(message)

    async def broadcast_camera_status(self, camera_id: str, status: dict):
        """
        Broadcast camera status update.

        Args:
            camera_id: Camera identifier
            status: Camera status dictionary
        """
        message = {
            "type": "camera_status",
            "timestamp": datetime.now().isoformat(),
            "camera_id": camera_id,
            "data": status
        }

        await self.broadcast(message)

    def has_connections(self) -> bool:
        """
        Check if there are active connections.

        Returns:
            True if there are active connections
        """
        return len(self.active_connections) > 0

    def get_connection_count(self) -> int:
        """
        Get number of active connections.

        Returns:
            Connection count
        """
        return len(self.active_connections)


# Global connection manager instance
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, auth_manager=None):
    """
    WebSocket endpoint handler.

    Args:
        websocket: WebSocket connection
        auth_manager: Authentication manager (optional)
    """
    # Must accept the connection BEFORE any other operations
    await websocket.accept()

    # Authenticate if auth is required
    if auth_manager and auth_manager.auth_required:
        authenticated = False

        # Check Authorization header (for Basic Auth)
        auth_header = websocket.headers.get("authorization")
        if auth_header and auth_header.startswith("Basic "):
            try:
                credentials = base64.b64decode(auth_header.split(" ")[1]).decode("utf-8")
                username, password = credentials.split(":", 1)

                if username == auth_manager.username and password == auth_manager.password:
                    authenticated = True
            except Exception as e:
                logger.warning(f"Failed to parse WebSocket auth header: {e}")

        # Check query parameters as fallback
        if not authenticated:
            query_params = dict(websocket.query_params)
            username = query_params.get("username")
            password = query_params.get("password")

            if username == auth_manager.username and password == auth_manager.password:
                authenticated = True
                logger.info("WebSocket authentication successful")

        if not authenticated:
            logger.warning(f"WebSocket connection rejected: authentication failed (authenticated={authenticated})")
            # Send error message before closing
            await websocket.send_json({
                "type": "error",
                "message": "Authentication failed",
                "timestamp": datetime.now().isoformat()
            })
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

    # Register the connection manually (don't call manager.connect as it calls accept again)
    manager.active_connections.add(websocket)
    logger.info(f"WebSocket client connected. Total connections: {len(manager.active_connections)}")

    try:
        # Send initial connection confirmation
        await manager.send_personal_message(
            {
                "type": "connection",
                "status": "connected",
                "timestamp": datetime.now().isoformat()
            },
            websocket
        )

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)

                # Handle different message types
                message_type = message.get("type")

                if message_type == "ping":
                    # Respond to ping
                    await manager.send_personal_message(
                        {
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        },
                        websocket
                    )

                elif message_type == "subscribe":
                    # Handle subscription requests
                    topics = message.get("topics", [])
                    await manager.send_personal_message(
                        {
                            "type": "subscribed",
                            "topics": topics,
                            "timestamp": datetime.now().isoformat()
                        },
                        websocket
                    )

                else:
                    logger.warning(f"Unknown message type: {message_type}")

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "message": "Invalid JSON format",
                        "timestamp": datetime.now().isoformat()
                    },
                    websocket
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected normally")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def start_status_broadcaster(system_instance, interval: int = 5):
    """
    Background task to broadcast system status periodically.

    Args:
        system_instance: Guardian Angel system instance
        interval: Broadcast interval in seconds
    """
    while True:
        try:
            if manager.has_connections() and system_instance is not None:
                # Get system status
                if hasattr(system_instance, 'get_status'):
                    status = system_instance.get_status()
                    await manager.broadcast_system_status(status)

            await asyncio.sleep(interval)

        except Exception as e:
            logger.error(f"Error in status broadcaster: {e}")
            await asyncio.sleep(interval)
