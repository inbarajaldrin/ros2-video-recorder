#!/usr/bin/env python3
"""
MCP Server for ROS2 Video Recorder

This server provides tools to control video recording from ROS2 image topics.
It directly subscribes to camera topics using rclpy and OpenCV for recording,
making it a fully standalone solution that requires only the ROS2 environment.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator
from dataclasses import dataclass
from mcp.server.fastmcp import FastMCP, Context
from .recorder_manager import VideoRecorderManager


# Output directories - use MCP_CLIENT_OUTPUT_DIR if set, otherwise use relative paths
BASE_OUTPUT_DIR = os.getenv("MCP_CLIENT_OUTPUT_DIR", "").strip()
if BASE_OUTPUT_DIR:
    VIDEOS_DIR = os.path.join(BASE_OUTPUT_DIR, "videos")
else:
    VIDEOS_DIR = "videos"


@dataclass
class AppContext:
    """Application context holding the video recorder manager"""
    manager: VideoRecorderManager


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """
    Lifespan context manager for proper resource initialization and cleanup.

    This ensures the VideoRecorderManager is properly cleaned up when the MCP server
    disconnects, not just on process exit. This fixes issues with server refresh
    where stale ROS 2 state would persist.
    """
    # Initialize manager on server startup
    manager = VideoRecorderManager(default_folder_path=VIDEOS_DIR)
    try:
        yield AppContext(manager=manager)
    finally:
        # Cleanup on server shutdown (including refresh/reconnect)
        await manager.cleanup()


# Initialize MCP server with lifespan
mcp = FastMCP("ros2-video-recorder", lifespan=app_lifespan)


@mcp.tool(description="Start recording video from a ROS2 image topic. Records continuously until stopped (default) or for a specified duration. Videos are saved with timestamps in the filename.")
async def start_recording(
    ctx: Context,
    camera_topic: str,
    fps: int = None,
    image_height: int = None,
    image_width: int = None,
    overlay_timestamp: bool = False,
    folder_path: str = None,
    video_length: int = 0,
    auto_fps: bool = True,
    auto_resolution: bool = True,
    video_codec: str = "mp4v",
    file_prefix: str = "",
    file_postfix: str = "",
    file_type: str = "mp4"
) -> str:
    """
    Start recording video from a ROS2 image topic.

    Args:
        camera_topic: ROS2 topic name for camera images (e.g., /camera/image_raw) (required)
        fps: Frame rate for recording (frames per second, 1-120). If None, auto-detects from topic (default: None)
        image_height: Image height in pixels. If None, auto-detects from first frame (default: None)
        image_width: Image width in pixels. If None, auto-detects from first frame (default: None)
        overlay_timestamp: Whether to overlay timestamp on video frames
        folder_path: Directory to save recorded videos (optional)
        video_length: Length of each video segment in seconds (0 = continuous recording)
        auto_fps: Auto-detect topic framerate (default: True). Disabled if fps is explicitly set
        auto_resolution: Auto-detect image resolution from topic (default: True). Disabled if width/height are explicitly set
        video_codec: Video codec to use (e.g., mp4v, h264)
        file_prefix: Prefix for output filenames
        file_postfix: Postfix for output filenames
        file_type: Video file extension (e.g., mp4, avi)

    Returns:
        Status message indicating success or failure
    """
    manager = ctx.request_context.lifespan_context.manager
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
async def stop_recording(ctx: Context) -> str:
    """
    Stop the current recording.

    Returns:
        Status message indicating success or failure, including the path to the saved video
    """
    manager = ctx.request_context.lifespan_context.manager
    return await manager.stop_recording()


@mcp.tool(description="Check if video recording is currently active")
async def get_recording_status(ctx: Context) -> str:
    """
    Get current recording status.

    Returns:
        Status message with details about the recording
    """
    manager = ctx.request_context.lifespan_context.manager
    return await manager.get_status()


def main():
    """Main entry point for the MCP server (called by the script entrypoint)"""
    mcp.run()


if __name__ == "__main__":
    main()

