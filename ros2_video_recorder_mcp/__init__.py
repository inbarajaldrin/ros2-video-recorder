"""ROS2 Video Recorder MCP Server"""

__version__ = "0.1.0"

from .video_manager import VideoRecorderManager
from .image_manager import CameraCaptureManager, CaptureResult

__all__ = ["VideoRecorderManager", "CameraCaptureManager", "CaptureResult"]
