"""Main FastAPI application for Guardian Angel web dashboard."""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path
from loguru import logger
import asyncio

from .routes import config, cameras, alerts, system
from .websocket import websocket_endpoint, manager, start_status_broadcaster
from . import auth as auth_module


def create_app(config_manager, system_instance, dashboard_config: dict):
    """
    Create and configure FastAPI application.

    Args:
        config_manager: ConfigManager instance
        system_instance: GuardianAngelSystem instance
        dashboard_config: Dashboard configuration

    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="Guardian Angel Dashboard",
        description="Web dashboard for Guardian Angel School Safety System",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc"
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize authentication
    auth_module.auth_manager = auth_module.AuthManager(
        username=dashboard_config.get("username", "admin"),
        password=dashboard_config.get("password", "change_me_in_production"),
        auth_required=dashboard_config.get("auth_required", True)
    )

    # Inject dependencies into route modules
    config.config_manager = config_manager
    config.system_instance = system_instance

    cameras.config_manager = config_manager
    cameras.system_instance = system_instance

    alerts.alert_manager = system_instance.alert_manager if hasattr(system_instance, 'alert_manager') else None

    system.config_manager = config_manager
    system.system_instance = system_instance

    # Include routers
    app.include_router(config.router)
    app.include_router(cameras.router)
    app.include_router(alerts.router)
    app.include_router(system.router)

    # Setup static files and templates
    static_path = Path(__file__).parent.parent / "web" / "static"
    templates_path = Path(__file__).parent.parent / "web" / "templates"

    static_path.mkdir(parents=True, exist_ok=True)
    templates_path.mkdir(parents=True, exist_ok=True)

    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    templates = Jinja2Templates(directory=str(templates_path))

    # WebSocket endpoint
    @app.websocket("/ws")
    async def websocket_route(websocket):
        """WebSocket endpoint for real-time updates."""
        await websocket_endpoint(websocket)

    # Web UI routes
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Main dashboard page."""
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/config", response_class=HTMLResponse)
    async def config_page(request: Request):
        """Configuration page."""
        return templates.TemplateResponse("config.html", {"request": request})

    @app.get("/cameras", response_class=HTMLResponse)
    async def cameras_page(request: Request):
        """Camera management page."""
        return templates.TemplateResponse("cameras.html", {"request": request})

    @app.get("/zones", response_class=HTMLResponse)
    async def zones_page(request: Request):
        """Zone editor page."""
        return templates.TemplateResponse("zones.html", {"request": request})

    @app.get("/alerts", response_class=HTMLResponse)
    async def alerts_page(request: Request):
        """Alert review page."""
        return templates.TemplateResponse("alerts.html", {"request": request})

    # Startup event
    @app.on_event("startup")
    async def startup_event():
        """Startup tasks."""
        logger.info("Starting FastAPI dashboard server")

        # Start WebSocket status broadcaster if enabled
        if dashboard_config.get("websocket_enabled", True):
            asyncio.create_task(start_status_broadcaster(system_instance, interval=5))
            logger.info("WebSocket status broadcaster started")

    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Shutdown tasks."""
        logger.info("Shutting down FastAPI dashboard server")

    logger.success("FastAPI application created")

    return app
