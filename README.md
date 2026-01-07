# ROS2 Video Recorder MCP Server

A standalone Model Context Protocol (MCP) server that records video directly from ROS2 image topics. This server uses `rclpy` and OpenCV to subscribe to camera topics and save video files, allowing AI assistants and other MCP clients to control video recording. Also includes a CLI tool for manual control and a Python library for programmatic use.

## Features

- **Direct ROS2 Integration**: Subscribes directly to ROS2 image topics using rclpy
- **Standalone Recording**: No external ROS2 packages required - uses OpenCV for video encoding
- **Auto-Detection**: Automatically detects FPS and resolution from the topic (enabled by default)
- **Real-time Status**: Dynamic status updates showing FPS, resolution, and frame count
- **MCP Tools**:
  - `start_recording`: Start recording from any ROS2 image topic
  - `stop_recording`: Stop the current recording session
  - `get_recording_status`: Check recording status
- **CLI Interface**: Command-line tool for manual control
- **Python Library**: Use `VideoRecorderManager` in your own code

## Prerequisites

- **ROS2** (Humble or later) installed and sourced
- **Python 3.10** (required by ROS2 Humble)
- **cv_bridge** ROS2 package

## Installation

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd ros2-video-recorder
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   # or with uv
   uv sync
   ```

## Usage

### MCP Server Configuration

Add to your MCP client configuration file (e.g., `mcp_config.json`):

```json
{
  "mcpServers": {
    "ros2-video-recorder": {
      "disabled": false,
      "timeout": 60,
      "type": "stdio",
      "command": "bash",
      "args": [
        "-c",
        "source /opt/ros/humble/setup.bash && cd /path/to/ros2-video-recorder && python3 -m ros2_video_recorder_mcp.server"
      ],
      "env": {
        "ROS_DOMAIN_ID": "0"
      }
    }
  }
}
```

**Important**: Update `/path/to/ros2-video-recorder` to your actual installation path.

### MCP Tools

#### start_recording

Start recording video from a ROS2 image topic.

**Parameters:**
- `camera_topic` (string, default: "/camera_input"): ROS2 topic for camera images
- `fps` (integer, optional): Frame rate for recording (1-120). If not specified, auto-detects from topic
- `image_height` (integer, optional): Image height in pixels. If not specified, auto-detects from first frame
- `image_width` (integer, optional): Image width in pixels. If not specified, auto-detects from first frame
- `auto_fps` (boolean, default: true): Auto-detect FPS from topic (disabled if fps is explicitly set)
- `auto_resolution` (boolean, default: true): Auto-detect resolution (disabled if width/height are set)
- `overlay_timestamp` (boolean, default: false): Overlay timestamp on frames
- `folder_path` (string, optional): Directory to save videos
- `video_codec` (string, default: "mp4v"): Video codec (mp4v, XVID, etc.)
- `file_prefix` (string): Filename prefix
- `file_postfix` (string): Filename postfix
- `file_type` (string, default: "mp4"): Video file extension

**Example Output:**
```
Recording started successfully!
Output file: /home/user/videos/2026-01-07_07-02-21.mp4
Camera topic: /camera/image_raw
FPS: 30
Resolution: 1280x720
Overlay timestamp: True
Video length: continuous
```

#### stop_recording

Stop the current recording session.

**Example Output:**
```
Recording stopped successfully.
Video saved: /home/user/videos/2026-01-07_07-02-21.mp4
FPS: 30
Resolution: 1280x720
Frame count: 150
Duration: 5s
File size: 0.65 MB
```

#### get_recording_status

Check the current recording status.

**Example Output:**
```
Status: Recording in progress
Camera topic: /camera/image_raw
FPS: 30
Resolution: 1280x720
Frames recorded: 42
```

## CLI Usage

### Start Recording
```bash
# Source ROS2 first
source /opt/ros/humble/setup.bash

# Auto-detect FPS and resolution
uv run ros2-recorder-cli start --camera-topic /camera/image_raw

# With timestamp overlay
uv run ros2-recorder-cli start --camera-topic /camera/image_raw --timestamp

# Manual FPS and resolution
uv run ros2-recorder-cli start \
  --camera-topic /camera/image_raw \
  --fps 30 \
  --width 1920 \
  --height 1080

# Full options with custom output
uv run ros2-recorder-cli start \
  --camera-topic /camera/image_raw \
  --timestamp \
  --output ~/videos \
  --prefix "robot1_" \
  --postfix "_test"

# Enable verbose ROS2 logging (for debugging)
uv run ros2-recorder-cli start --camera-topic /camera/image_raw --verbose
```

**Output:**
```
Recording started successfully!
Output file: /home/user/videos/2026-01-07_07-02-21.mp4
Camera topic: /camera/image_raw
FPS: Auto-detect
Resolution: Auto-detect
Overlay timestamp: False
Video length: continuous

Recording in progress. Press Ctrl+C to stop...
Status: FPS=30, Resolution=1280x720, Frames=150
```

### Record for Specific Duration
```bash
# Record for 10 seconds then stop
uv run ros2-recorder-cli record --duration 10 --camera-topic /camera/image_raw
```

### Stop Recording
```bash
uv run ros2-recorder-cli stop
```

**Output:**
```
Recording stopped successfully.
Video saved: /home/user/videos/2026-01-07_07-02-21.mp4
FPS: 30
Resolution: 1280x720
Frame count: 150
Duration: 5s
File size: 0.65 MB
```

### Check Status
```bash
uv run ros2-recorder-cli status
```

**Output:**
```
Status: Recording in progress
Camera topic: /camera/image_raw
FPS: 30
Resolution: 1280x720
Frames recorded: 42
```

## Output Location

### Default Behavior
- **No environment variables**: Videos saved to `./videos/`
- **With `MCP_CLIENT_OUTPUT_DIR`**: Videos saved to `{MCP_CLIENT_OUTPUT_DIR}/videos/`

### Filename Format
- Default: `YYYY-MM-DD_HH-MM-SS.mp4`
- With prefix/postfix: `{prefix}YYYY-MM-DD_HH-MM-SS{postfix}.mp4`
- Example: `robot1_2026-01-06_14-30-45_test.mp4`

## Auto-Detection Features

### FPS Auto-Detection
By default, the recorder automatically detects the frame rate from the ROS2 topic:
- Calculates FPS from message timestamps (averages last 10 intervals)
- Works with any topic publishing rate
- Can be overridden with `--fps` parameter

### Resolution Auto-Detection
By default, the recorder automatically detects resolution from the first frame:
- Reads actual image dimensions from the topic
- No need to know camera specifications in advance
- Can be overridden with `--width` and `--height` parameters

### Manual Override
You can disable auto-detection and set manual values:
```bash
uv run ros2-recorder-cli start \
  --camera-topic /camera/image_raw \
  --fps 60 \
  --width 1920 \
  --height 1080
```

## Advanced Configuration

## Using as a Python Library

```python
import asyncio
from ros2_video_recorder_mcp import VideoRecorderManager

async def main():
    manager = VideoRecorderManager()

    # Start recording with auto-detection
    result = await manager.start_recording(
        camera_topic="/camera/image_raw",
        overlay_timestamp=True,
        auto_fps=True,
        auto_resolution=True
    )
    print(result)

    # Record for 10 seconds
    await asyncio.sleep(10)

    # Stop recording
    result = await manager.stop_recording()
    print(result)

    # Cleanup
    await manager.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

### With Manual Settings
```python
# Specify exact FPS and resolution
result = await manager.start_recording(
    camera_topic="/camera/image_raw",
    fps=60,
    image_width=1920,
    image_height=1080,
    overlay_timestamp=True,
    verbose=True  # Enable ROS2 logging
)
```

## Troubleshooting

### Import Error: "Failed to import rclpy"
Make sure you're using Python 3.10 (ROS2 Humble requirement) and have sourced ROS2:
```bash
source /opt/ros/humble/setup.bash
python3 --version  # Should show Python 3.10.x
```

### NumPy Version Error
ROS2 Humble's cv_bridge requires NumPy 1.x. The project automatically handles this:
```toml
numpy = "<2.0.0"  # Compatible with cv_bridge
```

### No video output
Check:
1. Camera topic exists: `ros2 topic list | grep camera`
2. Images are being published: `ros2 topic echo /camera/image_raw --no-arr`
3. Output folder has write permissions

### Auto-detection timeout
If auto-detection takes too long:
1. Check that images are being published to the topic
2. Verify topic has active publishers: `ros2 topic info /camera/image_raw`
3. Try manual settings with `--fps` and `--width`/`--height`
4. Enable verbose logging: `--verbose`

### "Recording already in progress"
Stop the current recording first:
```bash
uv run ros2-recorder-cli stop
```

### Enable Debug Logging
Use the `--verbose` flag to see ROS2 log messages:
```bash
uv run ros2-recorder-cli start --camera-topic /camera/image_raw --verbose
```

## Project Structure

```
ros2-video-recorder/
├── ros2_video_recorder_mcp/
│   ├── __init__.py          # Package exports
│   ├── server.py            # MCP server implementation
│   ├── recorder_manager.py  # Core video recorder (ROS2 node + manager)
│   └── cli.py               # Standalone CLI tool
├── pyproject.toml           # Project dependencies and metadata
└── README.md                # This file
```

## How It Works

1. **VideoRecorderNode**: A `rclpy.Node` that subscribes to image topics
   - Auto-detects FPS by averaging frame timestamps
   - Auto-detects resolution from first frame dimensions
   - Uses event-based synchronization for minimal latency
2. **VideoRecorderManager**: Manages the recorder node lifecycle and state
   - Runs ROS2 executor in background thread
   - Handles state persistence for cross-process communication
3. **MCP Server**: Exposes manager functions as MCP tools
   - Clean output by default (verbose mode available)
4. **CLI**: Provides command-line access to the manager
   - Real-time status updates during recording
   - Graceful Ctrl+C handling

The recorder runs in a background thread, continuously processing incoming ROS2 images and writing them to a video file using OpenCV's `VideoWriter`.

## Recent Improvements

### v1.1 (Current)
- ✨ **Auto-detection**: FPS and resolution detected automatically from topic
- ⚡ **Event-based Sync**: Replaced polling with asyncio events for faster startup
- 📊 **Real-time Status**: Dynamic status line showing FPS, resolution, frames
- 🔧 **Better Compatibility**: Fixed NumPy version, Python 3.10 requirement
- 🚀 **Performance**: Removed unnecessary delays (~0.5s faster startup)
- 🐛 **Clean Output**: Minimal logging by default, `--verbose` flag for debugging

## License

MIT License - See LICENSE file for details.

## Contributing

Issues and pull requests welcome!
