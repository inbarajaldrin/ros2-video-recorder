#!/usr/bin/env python3
"""
MCP Server for ROS2 Video Recorder

This server provides tools to control video recording from ROS2 image topics.
It directly subscribes to camera topics using rclpy and OpenCV for recording,
making it a fully standalone solution that requires only the ROS2 environment.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Union, List
from dataclasses import dataclass
from mcp.server.fastmcp import FastMCP, Context
from fastmcp.utilities.types import Image as MCPImage
from .video_manager import VideoRecorderManager
from .image_manager import CameraCaptureManager, numpy_to_bytes_jpeg


# Output directories - use MCP_CLIENT_OUTPUT_DIR if set, otherwise use relative paths
BASE_OUTPUT_DIR = os.getenv("MCP_CLIENT_OUTPUT_DIR", "").strip()
if BASE_OUTPUT_DIR:
    VIDEOS_DIR = os.path.join(BASE_OUTPUT_DIR, "videos")
    SCREENSHOTS_DIR = os.path.join(BASE_OUTPUT_DIR, "screenshots")
else:
    VIDEOS_DIR = "videos"
    SCREENSHOTS_DIR = "screenshots"


@dataclass
class AppContext:
    """Application context holding the video recorder and camera capture managers"""
    video_manager: VideoRecorderManager
    image_manager: CameraCaptureManager


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """
    Lifespan context manager for proper resource initialization and cleanup.

    This ensures the managers are properly cleaned up when the MCP server
    disconnects, not just on process exit. This fixes issues with server refresh
    where stale ROS 2 state would persist.
    """
    # Initialize managers on server startup
    video_manager = VideoRecorderManager(default_folder_path=VIDEOS_DIR)
    image_manager = CameraCaptureManager(screenshots_dir=SCREENSHOTS_DIR)
    try:
        yield AppContext(video_manager=video_manager, image_manager=image_manager)
    finally:
        # Cleanup on server shutdown (including refresh/reconnect)
        await video_manager.cleanup()
        await image_manager.cleanup()


# Initialize MCP server with lifespan
mcp = FastMCP("ros2-video-recorder", lifespan=app_lifespan)


@mcp.tool(description="Start recording video from one or more ROS2 image topics. Records continuously until stopped (default) or for a specified duration. Videos are saved with timestamps in the filename. Can record multiple topics in parallel by passing a list.")
async def start_recording(
    ctx: Context,
    camera_topic: Union[str, List[str]],
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
    Start recording video from one or more ROS2 image topics.

    Args:
        camera_topic: ROS2 topic name(s) for camera images. Can be a single string (e.g., "/camera/image_raw") or a list of strings (e.g., ["/camera1/image_raw", "/camera2/image_raw"]) to record multiple topics in parallel. (required)
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
        Status message indicating success or failure for all topics
    """
    video_manager = ctx.request_context.lifespan_context.video_manager
    return await video_manager.start_recording(
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


@mcp.tool(description="Stop the current video recording session(s) gracefully. Can stop a specific topic or all recordings.")
async def stop_recording(ctx: Context, camera_topic: str = None) -> str:
    """
    Stop the current recording(s).

    Args:
        camera_topic: Optional. If provided, stop only this topic's recording. If None, stop all active recordings.

    Returns:
        Status message indicating success or failure, including the path(s) to the saved video(s)
    """
    video_manager = ctx.request_context.lifespan_context.video_manager
    return await video_manager.stop_recording(camera_topic=camera_topic)


@mcp.tool(description="Check if video recording is currently active")
async def get_recording_status(ctx: Context) -> str:
    """
    Get current recording status.

    Returns:
        Status message with details about the recording
    """
    video_manager = ctx.request_context.lifespan_context.video_manager
    return await video_manager.get_status()


@mcp.tool(description="Capture a single image from a ROS2 camera topic. Returns the image so the agent can see and analyze it, along with status information.")
async def capture_camera_image(ctx: Context, topic_name: str, timeout: int = 10) -> list:
    """
    Capture camera image from any ROS2 image topic.
    Returns a list with status info and the image that the agent can see.

    Args:
        topic_name: The ROS2 topic to subscribe to (e.g., "/camera/image_raw")
        timeout: Timeout in seconds for image capture (default: 10)

    Returns:
        List containing status JSON and optionally the captured image
    """
    import json
    from datetime import datetime
    from .image_manager import CaptureResult

    video_manager = ctx.request_context.lifespan_context.video_manager
    image_manager = ctx.request_context.lifespan_context.image_manager

    # If there's an active video recording on this topic, grab the frame directly
    # instead of creating a competing subscription that would time out
    frame_rgb = video_manager.get_latest_frame(topic_name)
    if frame_rgb is not None:
        timestamp = datetime.now().isoformat()
        # Save screenshot backup
        save_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_clean = topic_name.replace("/", "_").strip("_")
        filename = f"{save_timestamp}_{topic_clean}.jpg"
        save_path = os.path.join(image_manager.screenshots_dir, filename)

        import cv2
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, frame_bgr)

        result = CaptureResult(
            success=True,
            image_data=frame_rgb,
            message=f"Image captured from {topic_name} (from active recording)",
            topic=topic_name,
            timestamp=timestamp,
            saved_path=save_path,
            width=frame_rgb.shape[1],
            height=frame_rgb.shape[0],
            is_fallback=False
        )
    else:
        result = await image_manager.capture_image(topic_name, timeout)

    # Build status response
    status = {
        "timestamp": result.timestamp,
        "topic": result.topic,
        "status": "success" if result.success else ("timeout" if "Timeout" in result.message else "error"),
        "message": result.message
    }

    if result.saved_path:
        status["saved_to"] = result.saved_path
    if result.width and result.height:
        status["resolution"] = f"{result.width}x{result.height}"
    if result.is_fallback:
        status["is_fallback"] = True

    # If we have image data, return it along with status
    if result.success and result.image_data is not None:
        # Convert numpy image to JPEG bytes for MCP Image
        jpeg_bytes = numpy_to_bytes_jpeg(result.image_data)

        # Use FastMCP's Image helper for proper MCP format
        mcp_image = MCPImage(data=jpeg_bytes)

        # Return list with status text and image content
        return [
            json.dumps(status, indent=2),
            mcp_image.to_image_content(mime_type="image/jpeg")
        ]
    else:
        # No image available, return only status
        return [json.dumps(status, indent=2)]


def main():
    """Main entry point for the MCP server (called by the script entrypoint)"""
    mcp.run()


if __name__ == "__main__":
    main()

