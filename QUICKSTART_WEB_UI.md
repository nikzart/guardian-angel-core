# Quick Start: Web Dashboard

Get started with the Guardian Angel web interface in 5 minutes.

## Step 1: Enable the Dashboard

Edit `configs/system_config.yaml`:

```yaml
dashboard:
  enabled: true  # ← Change this to true
  host: "0.0.0.0"
  port: 8080
  websocket_enabled: true
  auth_required: true
  username: "admin"
  password: "change_me_in_production"  # ← CHANGE THIS!
```

## Step 2: Install Dependencies

If you haven't already:

```bash
pip install -r requirements.txt
```

Additional packages for web dashboard:
```bash
pip install fastapi uvicorn python-multipart psutil
```

## Step 3: Start the System

```bash
python src/main.py --config configs/system_config.yaml
```

You should see:
```
[INFO] API dashboard server started at http://0.0.0.0:8080
[SUCCESS] Guardian Angel system started with X cameras
```

## Step 4: Open the Dashboard

Open your web browser and go to:

```
http://localhost:8080
```

**Login with:**
- Username: `admin`
- Password: (whatever you set in Step 1)

## Step 5: Configure Your First Camera

1. Click **"Cameras"** in the navigation bar
2. Click **"Add Camera"** button
3. Fill in the form:
   ```
   Camera ID: camera_01
   Name: Main Entrance
   Source: rtsp://admin:password@192.168.1.100:554/stream
   Target FPS: 30
   ```
4. Click **"Test Connection"** to verify
5. Click **"Add Camera"** to save

## Step 6: Adjust Detection Settings

1. Click **"Configuration"** in navigation
2. Select **"Fall Detection"** from sidebar
3. Adjust sliders:
   - Confidence Threshold: 0.6
   - Alert Cooldown: 5 seconds
4. Click **"Save Configuration"** at top

## Step 7: Monitor Alerts

1. Click **"Alerts"** in navigation
2. See real-time alerts as they occur
3. Filter by camera, type, or status
4. Click **"Review"** to mark alerts as seen

## Common Tasks

### Add a Camera from Video File

```
Source: /path/to/video.mp4
```

### Test with Webcam

```
Source: 0
```
(0 for first webcam, 1 for second, etc.)

### Configure Restricted Zones

1. Go to **Zones** page
2. Select camera
3. Click **"Draw Zone"**
4. Click to draw polygon on video feed
5. Mark as "Restricted"
6. Click **"Save Zones"**

### Enable/Disable Detectors

**Via UI:**
1. Configuration → Select detector
2. Toggle "Enable" switch
3. Save Configuration

**Via Config File:**
```yaml
fall_detection:
  enabled: false  # Disable fall detection

bullying_detection:
  enabled: true  # Enable bullying detection
```

### View System Status

Dashboard homepage shows:
- 📹 Active cameras
- 🔔 Recent alerts
- 💻 CPU/Memory usage
- 📊 Alert statistics

### System Controls

Top-right corner:
- ▶️ **Start** - Start detection
- ⏹️ **Stop** - Stop detection
- 🔄 **Restart** - Restart with new config

## Troubleshooting

### "Connection refused"

**Problem:** Can't access dashboard at localhost:8080

**Solution:**
```bash
# Check if system is running
ps aux | grep python

# Check if port is in use
lsof -i :8080

# Start the system
python src/main.py
```

### "Authentication failed"

**Problem:** Wrong username/password

**Solution:**
1. Check credentials in `configs/system_config.yaml`
2. Default is `admin` / `change_me_in_production`
3. Edit config file and restart system

### Camera preview not showing

**Problem:** Black screen or broken image

**Solution:**
1. Check camera is running (green indicator)
2. Test camera connection in Cameras page
3. Verify RTSP URL is correct
4. Check camera permissions

### Configuration changes not applying

**Problem:** Changed settings but no effect

**Solution:**
1. Click "Save Configuration" button
2. Some settings require system restart
3. Use "Restart" button in dashboard
4. Check logs for errors

### WebSocket disconnected

**Problem:** Red "Disconnected" indicator

**Solution:**
1. Refresh the page
2. Check system is running
3. Check firewall settings
4. Verify `websocket_enabled: true` in config

## Next Steps

- 📖 Read full [Web Dashboard Documentation](docs/WEB_DASHBOARD.md)
- 🔐 Configure [HTTPS with reverse proxy](docs/WEB_DASHBOARD.md#https-setup)
- 🎯 Fine-tune [detection thresholds](docs/WEB_DASHBOARD.md#configuration-editor)
- 📊 Set up [alert notifications](docs/WEB_DASHBOARD.md#websocket-protocol)
- 🗺️ Draw [detection zones](docs/WEB_DASHBOARD.md#zone-editor)

## API Access

Access the API docs at:
- **Swagger UI:** http://localhost:8080/api/docs
- **ReDoc:** http://localhost:8080/api/redoc

Example API call:
```bash
# Get all cameras (with auth)
curl -u admin:password http://localhost:8080/api/cameras/

# Get system status
curl -u admin:password http://localhost:8080/api/system/status
```

## Security Reminder

⚠️ **Before deploying to production:**

1. ✅ Change default password
2. ✅ Enable HTTPS (use reverse proxy)
3. ✅ Restrict access by IP if possible
4. ✅ Keep `auth_required: true`
5. ✅ Use strong passwords
6. ✅ Regular security updates

## Support

Having issues?
- Check `logs/guardian_angel_*.log` for errors
- Visit GitHub Issues
- Read full documentation in `docs/`
