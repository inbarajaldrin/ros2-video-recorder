#!/usr/bin/env python3
"""
MCP Server for ROS2 Video Recorder

This server provides tools to control video recording from ROS2 image topics.
It directly subscribes to camera topics using rclpy and OpenCV for recording,
making it a fully standalone solution that requires only the ROS2 environment.
"""

import os
import atexit
from mcp.server.fastmcp import FastMCP
from .recorder_manager import VideoRecorderManager


# Output directories - use MCP_CLIENT_OUTPUT_DIR if set, otherwise use relative paths
BASE_OUTPUT_DIR = os.getenv("MCP_CLIENT_OUTPUT_DIR", "").strip()
if BASE_OUTPUT_DIR:
    VIDEOS_DIR = os.path.join(BASE_OUTPUT_DIR, "videos")
else:
    VIDEOS_DIR = "videos"

# Global manager instance with default folder path
manager = VideoRecorderManager(default_folder_path=VIDEOS_DIR)

# Initialize MCP server
mcp = FastMCP("ros2-video-recorder")


def cleanup_on_shutdown():
    """Cleanup function to run when the server shuts down"""
    import asyncio
    try:
        asyncio.run(manager.cleanup())
    except Exception:
        pass


# Register cleanup function to run on exit
atexit.register(cleanup_on_shutdown)


@mcp.tool(description="Start recording video from a ROS2 image topic. Records continuously until stopped (default) or for a specified duration. Videos are saved with timestamps in the filename.")
async def start_recording(
    camera_topic: str = "/camera_input",
    fps: int = 30,
    image_height: int = 720,
    image_width: int = 1280,
    overlay_timestamp: bool = False,
    folder_path: str = None,
    video_length: int = 0,
    auto_fps: bool = False,
    auto_resolution: bool = False,
    video_codec: str = "mp4v",
    file_prefix: str = "",
    file_postfix: str = "",
    file_type: str = "mp4"
) -> str:
    """
    Start recording video from a ROS2 image topic.

    Args:
        camera_topic: ROS2 topic name for camera images (e.g., /camera/image_raw)
        fps: Frame rate for recording (frames per second, 1-120)
        image_height: Image height in pixels
        image_width: Image width in pixels
        overlay_timestamp: Whether to overlay timestamp on video frames
        folder_path: Directory to save recorded videos (optional)
        video_length: Length of each video segment in seconds (0 = continuous recording)
        auto_fps: Auto-detect topic framerate (overrides fps parameter)
        auto_resolution: Auto-detect image resolution from topic (overrides image_height and image_width)
        video_codec: Video codec to use (e.g., mp4v, h264)
        file_prefix: Prefix for output filenames
        file_postfix: Postfix for output filenames
        file_type: Video file extension (e.g., mp4, avi)

    Returns:
        Status message indicating success or failure
    """
    return await manager.start_recording(
        camera_topic=camera_topic,
        fps=fps,
        image_height=image_height,
        image_width=image_width,
        overlay_timestamp=overlay_timestamp,
        folder_path=folder_path,
        video_length=video_length,
        auto_fps=auto_fps,
        auto_resolution=auto_resolution,
        video_codec=video_codec,
        file_prefix=file_prefix,
        file_postfix=file_postfix,
        file_type=file_type
    )


@mcp.tool(description="Stop the current video recording session gracefully")
async def stop_recording() -> str:
    """
    Stop the current recording.

    Returns:
        Status message indicating success or failure, including the path to the saved video
    """
    return await manager.stop_recording()


@mcp.tool(description="Check if video recording is currently active")
async def get_recording_status() -> str:
    """
    Get current recording status.

    Returns:
        Status message with details about the recording
    """
    return await manager.get_status()


def main():
    """Main entry point for the MCP server (called by the script entrypoint)"""
    mcp.run()


if __name__ == "__main__":
    main()

