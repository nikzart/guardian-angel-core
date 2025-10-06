# Guardian Angel Core - School Safety Detection System

An edge AI system for real-time safety monitoring in educational environments using computer vision and pose estimation.

**🆕 Now with Web Dashboard!** Configure and monitor your system through an intuitive web interface.

## Features

### Detection Modules

1. **Fall Detection** - Detects students falling in playgrounds, corridors, or stairs
   - Pose-based vertical displacement analysis
   - Body orientation detection
   - Temporal smoothing for accuracy

2. **Bullying & Fight Detection** - Identifies aggressive behaviors and physical altercations
   - Group cornering detection (multiple people surrounding one)
   - Rapid movement analysis for fights
   - Aggressive gesture recognition

3. **POSH (Behavioral Anomaly) Detection** - Flags concerning interactions
   - Leading to isolated areas
   - Prolonged isolated interactions
   - Restricted zone entry monitoring

## Architecture

```
┌─────────────────┐
│  RTSP Cameras   │
└────────┬────────┘
         │
    ┌────▼─────┐
    │  Stream  │
    │  Handler │
    └────┬─────┘
         │
    ┌────▼──────┐
    │   Pose    │
    │ Estimation│ (YOLOv8-pose)
    └────┬──────┘
         │
    ┌────▼────────┐
    │   Tracking  │
    │ (DeepSORT)  │
    └────┬────────┘
         │
    ┌────▼────────────────────────┐
    │  Detection Modules          │
    │  ┌────────┬─────────┬─────┐ │
    │  │  Fall  │Bullying │POSH │ │
    │  └────────┴─────────┴─────┘ │
    └────┬────────────────────────┘
         │
    ┌────▼───────┐
    │   Alert    │
    │  Manager   │
    └────────────┘
```

## Quick Start

### 🌐 Web Dashboard (Recommended)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Enable dashboard in configs/system_config.yaml
dashboard:
  enabled: true
  host: "0.0.0.0"
  port: 8080
  username: "admin"
  password: "your_password"  # CHANGE THIS!

# 3. Start system with dashboard
python src/main.py

# 4. Open browser
http://localhost:8080
```

**📖 Full guide:** See [QUICKSTART_WEB_UI.md](QUICKSTART_WEB_UI.md)

**🎯 Dashboard Features:**
- ✅ Configure all settings via web forms
- ✅ Add/manage cameras with live preview
- ✅ Real-time alert monitoring with filtering
- ✅ Draw detection zones on camera feeds
- ✅ System control (start/stop/restart)
- ✅ WebSocket real-time updates
- ✅ Alert statistics and charts

### CLI Mode (Advanced)

```bash
# Clone repository
git clone <repository-url>
cd guardian-angel-core

# Install dependencies
pip install -r requirements.txt

# Configure system
cp configs/system_config.yaml configs/my_config.yaml
# Edit my_config.yaml with your camera settings

# Run system (no dashboard)
python src/main.py --config configs/my_config.yaml
```

### Option 2: NVIDIA Jetson Installation

```bash
# Run installation script
chmod +x deploy/install_jetson.sh
./deploy/install_jetson.sh

# Configure and run
python src/main.py
```

### Option 3: Docker Deployment

```bash
# Build and run with Docker Compose
cd deploy/docker
docker-compose up -d

# View logs
docker-compose logs -f

# Stop system
docker-compose down
```

## Configuration

Edit `configs/system_config.yaml` to configure:

### Camera Setup

```yaml
cameras:
  - camera_id: "camera_01"
    name: "Main Entrance"
    source: "rtsp://admin:password@192.168.1.100:554/stream"
    enabled: true
    target_fps: 30
```

### Detection Thresholds

```yaml
fall_detection:
  enabled: true
  confidence_threshold: 0.6
  alert_cooldown_seconds: 5

bullying_detection:
  enabled: true
  confidence_threshold: 0.5
  group_min_size: 3

posh_detection:
  enabled: true
  prolonged_interaction_frames: 150
```

### Zone Configuration (for POSH detection)

```yaml
posh_detection:
  zones:
    - name: "Stairwell Corner"
      is_restricted: true
      polygon:
        - [50, 50]
        - [150, 50]
        - [150, 150]
        - [50, 150]
```

## Usage

### Run System

```bash
# Standard run
python src/main.py

# With custom config
python src/main.py --config configs/my_config.yaml

# Test mode (60 seconds)
python src/main.py --test
```

### View Alerts

Alerts are saved to `data/alerts.db` (SQLite database) and video clips to `data/video_clips/`.

```python
from src.postprocessing import AlertManager

# Initialize alert manager
manager = AlertManager(config)

# Get recent alerts
alerts = manager.get_alerts(limit=10)

# Get statistics
stats = manager.get_statistics(days=7)
```

## Performance

### Hardware Requirements

**Minimum:**
- NVIDIA Jetson Nano (4GB)
- 1-2 cameras at 15-20 FPS

**Recommended:**
- NVIDIA Jetson Xavier NX or AGX Xavier
- 4+ cameras at 30 FPS
- 16GB+ RAM

### Optimization

1. **Model Optimization:**
   ```python
   from src.models import PoseEstimationModel

   model = PoseEstimationModel("yolov8n-pose.pt")
   model.export_tensorrt("yolov8n-pose.engine")  # For Jetson
   ```

2. **Adjust FPS:**
   Lower `target_fps` in config for better performance

3. **Disable Detectors:**
   Set `enabled: false` for unused detection modules

## Privacy & Compliance

- **On-device processing** - No cloud upload required
- **Pose-only mode** - Option to save only skeleton data
- **Configurable retention** - Automatic cleanup of old alerts
- **Face blurring** - Optional privacy filter for saved clips

```yaml
privacy:
  blur_faces: true
  save_only_pose_skeleton: true
  retention_days: 30
```

## Development

### Project Structure

```
guardian-angel-core/
├── src/
│   ├── detectors/         # Detection modules
│   ├── models/            # Model wrappers
│   ├── tracking/          # Multi-object tracking
│   ├── streaming/         # Camera input handling
│   ├── preprocessing/     # Frame processing
│   ├── postprocessing/    # Alert management
│   ├── utils/             # Utilities and types
│   └── main.py            # Main orchestration
├── configs/               # Configuration files
├── deploy/                # Deployment scripts
├── models/                # Model weights
├── data/                  # Alert database and clips
├── tests/                 # Unit tests
└── notebooks/             # Development notebooks
```

### Adding Custom Detectors

```python
from src.detectors import BaseDetector
from src.utils.types import Frame, Alert

class CustomDetector(BaseDetector):
    def preprocess(self, frame: Frame) -> Frame:
        # Preprocess frame
        return frame

    def detect(self, frame: Frame) -> List[Alert]:
        # Detection logic
        return alerts
```

## Limitations & Considerations

### Technical Limitations

1. **Context Understanding** - Cannot distinguish playful behavior from actual bullying
2. **Occlusion** - Performance degrades with crowded scenes
3. **Lighting** - Requires adequate lighting for pose estimation
4. **Privacy Trade-offs** - Continuous monitoring raises privacy concerns

### Deployment Considerations

1. **Legal Compliance** - Ensure compliance with COPPA, FERPA, local surveillance laws
2. **Stakeholder Consent** - Inform students, parents, and staff
3. **Human-in-the-Loop** - All alerts should be reviewed by trained personnel
4. **False Positives** - Tune thresholds to balance detection vs false alarms

## Troubleshooting

### Camera Connection Issues

```bash
# Test RTSP stream
ffplay rtsp://admin:password@192.168.1.100:554/stream

# Check logs
tail -f logs/guardian_angel_*.log
```

### Low FPS / Performance Issues

1. Reduce `target_fps` in camera config
2. Use smaller model: `yolov8n-pose.pt` instead of `yolov8m-pose.pt`
3. Reduce number of concurrent cameras
4. Export to TensorRT for Jetson devices

### High False Positive Rate

1. Increase `confidence_threshold` for detectors
2. Adjust detection-specific thresholds
3. Increase `alert_cooldown_seconds`

## Citation

If you use this system in research, please cite:

```bibtex
@software{guardian_angel_2025,
  title={Guardian Angel Core: Edge AI School Safety Detection System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/guardian-angel-core}
}
```

## License

[Specify your license here]

## Disclaimer

This system is designed as a **supplementary safety tool** and should not replace human supervision or existing safety protocols. All alerts must be reviewed by trained personnel. The developers are not responsible for false negatives, false positives, or any incidents that occur while the system is in use.

## Support

For issues, questions, or contributions:
- GitHub Issues: [repository-url]/issues
- Documentation: [docs-url]
- Email: support@example.com
