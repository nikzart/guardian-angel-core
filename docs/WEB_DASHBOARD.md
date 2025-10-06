# Guardian Angel Web Dashboard

A web-based configuration and monitoring interface for the Guardian Angel School Safety System.

## Features

- **Real-time Dashboard** - Live system status, camera feeds, and alert monitoring
- **Camera Management** - Add, edit, delete, and test cameras via UI
- **Configuration Editor** - Modify all system settings without editing YAML files
- **Alert Review** - Browse, filter, and review security alerts
- **Zone Editor** - Draw and configure detection zones on camera feeds
- **WebSocket Updates** - Real-time push notifications for alerts and status changes

## Quick Start

### 1. Enable the Dashboard

Edit `configs/system_config.yaml`:

```yaml
dashboard:
  enabled: true  # Enable the web dashboard
  host: "0.0.0.0"
  port: 8080
  websocket_enabled: true
  auth_required: true
  username: "admin"
  password: "your_secure_password_here"  # CHANGE THIS!
```

### 2. Start the System

```bash
python src/main.py --config configs/system_config.yaml
```

### 3. Access the Dashboard

Open your web browser and navigate to:

```
http://localhost:8080
```

**Default credentials:**
- Username: `admin`
- Password: (whatever you set in config)

## Dashboard Pages

### Main Dashboard (`/`)

**Overview of system status:**
- Total cameras and active camera count
- Recent alerts count (24 hours)
- CPU/Memory/Disk usage
- Live camera status indicators
- Recent alerts feed
- Alert statistics charts (by type and severity)

**Controls:**
- Start/Stop/Restart system buttons
- Real-time WebSocket connection status

### Camera Management (`/cameras`)

**Manage your camera feeds:**
- View all configured cameras
- Live camera preview thumbnails
- Add new cameras via modal form
- Test RTSP connections before saving
- Delete cameras
- See frame count and FPS for each camera

**Add Camera:**
1. Click "Add Camera" button
2. Fill in camera details:
   - Camera ID (unique identifier)
   - Name (display name)
   - Source (RTSP URL, video file, or webcam index)
   - Target FPS
3. Click "Test Connection" to verify
4. Click "Add Camera" to save

### Configuration Editor (`/config`)

**Modify all system settings via forms:**

**Sections:**
- System Settings (name, log level)
- Tracking (max age, min hits, IOU threshold)
- Pose Model (model path, device, confidence)
- Fall Detection (enable/disable, thresholds, parameters)
- Bullying Detection (enable/disable, group settings, movement thresholds)
- POSH Detection (enable/disable, isolation settings, interaction timing)
- Alert Settings (database, video clips, rate limiting)
- Privacy Settings (face blurring, retention)

**Usage:**
1. Navigate between sections using left sidebar
2. Modify values using forms
3. Sliders show current values in real-time
4. Click "Save Configuration" to apply changes
5. Configuration is backed up automatically

### Alert Management (`/alerts`)

**Review and manage security alerts:**

**Features:**
- Filter by camera, type, reviewed status
- Sortable table with all alert details
- Alert severity badges (low/medium/high/critical)
- Mark alerts as reviewed with notes
- Real-time alert notifications via WebSocket
- Alert confidence scores

**Filters:**
- All Cameras / Specific camera
- All Types / Fall/Fight/Bullying/POSH
- All Status / Unreviewed / Reviewed
- Result limit (default 50)

### Zone Editor (`/zones`)

**Draw detection zones for POSH detection:**

**Features:**
- Select camera to configure
- Interactive canvas for drawing polygons
- Mark zones as restricted/isolated areas
- Save zone coordinates to configuration
- Visual overlay on camera feed

**Usage:**
1. Select camera from dropdown
2. Click "Draw Zone" to start
3. Click points on canvas to draw polygon
4. Double-click to complete zone
5. Name zone and mark as restricted if needed
6. Click "Save Zones" to apply

## API Endpoints

The dashboard exposes a REST API for programmatic access:

### Configuration
- `GET /api/config/` - Get full configuration
- `GET /api/config/section/{section}` - Get specific section
- `PUT /api/config/` - Update configuration
- `POST /api/config/validate` - Validate config without applying
- `POST /api/config/reload` - Reload from file
- `GET /api/config/backups` - List backups
- `POST /api/config/backups/{filename}/restore` - Restore backup

### Cameras
- `GET /api/cameras/` - List all cameras
- `GET /api/cameras/{camera_id}` - Get camera details
- `POST /api/cameras/` - Create camera
- `PUT /api/cameras/{camera_id}` - Update camera
- `DELETE /api/cameras/{camera_id}` - Delete camera
- `GET /api/cameras/{camera_id}/status` - Get runtime status
- `POST /api/cameras/{camera_id}/test` - Test connection
- `GET /api/cameras/{camera_id}/preview` - Get preview frame (JPEG)

### Alerts
- `GET /api/alerts/` - Get alerts with filtering
- `GET /api/alerts/{alert_id}` - Get specific alert
- `POST /api/alerts/{alert_id}/review` - Mark as reviewed
- `GET /api/alerts/statistics/summary` - Get statistics
- `GET /api/alerts/recent/count` - Get recent count
- `DELETE /api/alerts/cleanup` - Clean up old alerts

### System
- `GET /api/system/info` - Get system information
- `GET /api/system/status` - Get runtime status
- `GET /api/system/cameras/status` - Get all camera statuses
- `POST /api/system/start` - Start system
- `POST /api/system/stop` - Stop system
- `POST /api/system/restart` - Restart system
- `GET /api/system/health` - Health check (no auth)

### WebSocket
- `WS /ws` - WebSocket connection for real-time updates

## WebSocket Protocol

Connect to `ws://localhost:8080/ws` for real-time updates.

**Message Types:**

**Client → Server:**
```json
{
  "type": "ping"
}

{
  "type": "subscribe",
  "topics": ["alerts", "system_status", "camera_status"]
}
```

**Server → Client:**
```json
{
  "type": "alert",
  "timestamp": "2025-01-15T10:30:00",
  "data": { /* alert object */ }
}

{
  "type": "system_status",
  "timestamp": "2025-01-15T10:30:00",
  "data": { /* status object */ }
}

{
  "type": "camera_status",
  "timestamp": "2025-01-15T10:30:00",
  "camera_id": "camera_01",
  "data": { /* camera status */ }
}
```

## Security

### Authentication

The dashboard uses HTTP Basic Authentication. Configure credentials in `system_config.yaml`:

```yaml
dashboard:
  auth_required: true  # Set to false to disable auth (NOT RECOMMENDED)
  username: "admin"
  password: "strong_password"
```

**Best Practices:**
- Always use strong passwords
- Change default password immediately
- Use HTTPS in production (reverse proxy recommended)
- Enable `auth_required` in production
- Restrict dashboard host to localhost if possible

### HTTPS Setup

For production, use a reverse proxy (nginx/Apache) with SSL:

```nginx
server {
    listen 443 ssl;
    server_name guardian-angel.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws {
        proxy_pass http://localhost:8080/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Configuration Backups

All configuration changes are automatically backed up to `configs/backups/`.

**Manual Restore:**
1. Go to Configuration page
2. Look at recent backups
3. Click "Restore" on desired backup
4. Confirm restoration

**Automatic Cleanup:**
- Keeps last 10 backups
- Older backups are automatically deleted

## Troubleshooting

### Dashboard Won't Start

**Check:**
1. Is `dashboard.enabled: true` in config?
2. Is port 8080 already in use? (`lsof -i :8080`)
3. Check logs for errors: `tail -f logs/guardian_angel_*.log`

**Solution:**
- Change port in config
- Stop conflicting service
- Check firewall settings

### Can't Connect to Dashboard

**Check:**
1. Is system running? (`ps aux | grep guardian`)
2. Correct URL? (`http://localhost:8080` not `https`)
3. Firewall blocking port?

**Solution:**
- Start system: `python src/main.py`
- Check host setting (use `0.0.0.0` for network access)
- Open firewall port: `sudo ufw allow 8080`

### WebSocket Not Connecting

**Check:**
1. Browser console for errors
2. WebSocket enabled in config?
3. Reverse proxy WebSocket configuration

**Solution:**
- Enable websocket in config
- Check reverse proxy WebSocket headers
- Try without proxy first

### Camera Preview Not Showing

**Check:**
1. Is camera stream running?
2. Camera producing frames?
3. Browser image format support

**Solution:**
- Restart camera stream
- Check camera source URL
- Try different browser

### Configuration Changes Not Applying

**Check:**
1. Click "Save Configuration" button?
2. Check for validation errors
3. System restarted after change?

**Solution:**
- Save configuration first
- Fix validation errors shown
- Some changes require restart

## Development

### Adding Custom Pages

1. Create HTML template in `src/web/templates/`
2. Create JavaScript in `src/web/static/js/`
3. Add route in `src/api/app.py`

### Adding API Endpoints

1. Create route file in `src/api/routes/`
2. Define endpoints with FastAPI decorators
3. Include router in `src/api/app.py`

### Styling

- Bootstrap 5 for UI components
- Custom styles in `src/web/static/css/style.css`
- Bootstrap Icons for icons
- Chart.js for visualizations

## API Documentation

Once the dashboard is running, visit:
- Swagger UI: `http://localhost:8080/api/docs`
- ReDoc: `http://localhost:8080/api/redoc`

## Support

For issues or questions:
- GitHub Issues: [repository-url]/issues
- Documentation: `docs/` directory
- System logs: `logs/guardian_angel_*.log`
