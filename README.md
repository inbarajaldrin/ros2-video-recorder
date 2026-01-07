# ROS2 Video Recorder MCP Server

A standalone Model Context Protocol (MCP) server that records video directly from ROS2 image topics. This server uses `rclpy` and OpenCV to subscribe to camera topics and save video files, allowing AI assistants and other MCP clients to control video recording. Also includes a CLI tool for manual control and a Python library for programmatic use.

## Features

- **Direct ROS2 Integration**: Subscribes directly to ROS2 image topics using rclpy
- **Standalone Recording**: No external ROS2 packages required - uses OpenCV for video encoding
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
- `fps` (integer, default: 30): Frame rate for recording (1-120)
- `image_height` (integer, default: 720): Image height in pixels
- `image_width` (integer, default: 1280): Image width in pixels
- `overlay_timestamp` (boolean, default: false): Overlay timestamp on frames
- `folder_path` (string, optional): Directory to save videos
- `video_codec` (string, default: "mp4v"): Video codec (mp4v, XVID, etc.)
- `file_prefix` (string): Filename prefix
- `file_postfix` (string): Filename postfix
- `file_type` (string, default: "mp4"): Video file extension

## CLI Usage

### Start Recording
```bash
# Source ROS2 first
source /opt/ros/humble/setup.bash

# Basic recording
uv run ros2-recorder-cli start --camera-topic /camera/image_raw --fps 30

# With timestamp overlay
uv run ros2-recorder-cli start --camera-topic /camera/image_raw --fps 30 --timestamp

# Full options
uv run ros2-recorder-cli start \
  --camera-topic /camera/image_raw \
  --fps 30 \
  --width 1920 \
  --height 1080 \
  --timestamp \
  --output ~/videos \
  --prefix "robot1_" \
  --postfix "_test"
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

### Check Status
```bash
uv run ros2-recorder-cli status
```

## Output Location

### Default Behavior
- **No environment variables**: Videos saved to `./videos/`
- **With `MCP_CLIENT_OUTPUT_DIR`**: Videos saved to `{MCP_CLIENT_OUTPUT_DIR}/videos/`

### Filename Format
- Default: `YYYY-MM-DD_HH-MM-SS.mp4`
- With prefix/postfix: `{prefix}YYYY-MM-DD_HH-MM-SS{postfix}.mp4`
- Example: `robot1_2026-01-06_14-30-45_test.mp4`

### Advanced Configuration

## Using as a Python Library

```python
import asyncio
from ros2_video_recorder_mcp import VideoRecorderManager

async def main():
    manager = VideoRecorderManager()

    # Start recording
    result = await manager.start_recording(
        camera_topic="/camera/image_raw",
        fps=30,
        overlay_timestamp=True
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

## Troubleshooting

### Import Error: "Failed to import rclpy"
Make sure you're using Python 3.10 (ROS2 Humble requirement) and have sourced ROS2:
```bash
source /opt/ros/humble/setup.bash
python3 --version  # Should show Python 3.10.x
```

### No video output
Check:
1. Camera topic exists: `ros2 topic list | grep camera`
2. Images are being published: `ros2 topic echo /camera/image_raw --no-arr`
3. Output folder has write permissions

### "Recording already in progress"
Stop the current recording first:
```bash
uv run ros2-recorder-cli stop
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
2. **VideoRecorderManager**: Manages the recorder node lifecycle and state
3. **MCP Server**: Exposes manager functions as MCP tools
4. **CLI**: Provides command-line access to the manager

The recorder runs in a background thread, continuously processing incoming ROS2 images and writing them to a video file using OpenCV's `VideoWriter`.

## License

MIT License - See LICENSE file for details.

## Contributing

Issues and pull requests welcome!
