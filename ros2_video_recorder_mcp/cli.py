#!/usr/bin/env python3
"""
Standalone CLI tool for controlling ROS2 video recorder

This can be used independently of the MCP server for testing
or direct control of the video recorder.

Examples:
    # Start recording with defaults
    python -m ros2_video_recorder_mcp.cli start

    # Start recording with custom parameters
    python -m ros2_video_recorder_mcp.cli start --camera-topic /camera/image_raw --fps 30 --timestamp

    # Record for 10 seconds then stop
    python -m ros2_video_recorder_mcp.cli record --duration 10 --camera-topic /camera_input

    # Stop recording
    python -m ros2_video_recorder_mcp.cli stop

    # Check status
    python -m ros2_video_recorder_mcp.cli status
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
from .recorder_manager import VideoRecorderManager


def get_project_root() -> Path:
    """Get the project root directory (where pyproject.toml is located)"""
    # Start from the current file and walk up to find pyproject.toml
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback to current working directory
    return Path.cwd()


def get_default_videos_folder() -> str:
    """Get the default videos folder path relative to project root"""
    project_root = get_project_root()
    videos_folder = project_root / "videos"
    # Create the folder if it doesn't exist
    videos_folder.mkdir(exist_ok=True)
    return str(videos_folder)


async def start_recording_cmd(args):
    """Start recording command"""
    # Use default folder path if not specified (relative to project root)
    default_folder = args.output if args.output else get_default_videos_folder()
    manager = VideoRecorderManager(default_folder_path=default_folder)

    result = await manager.start_recording(
        camera_topic=args.camera_topic,
        fps=args.fps,
        image_height=args.height,
        image_width=args.width,
        overlay_timestamp=args.timestamp,
        folder_path=args.output,
        video_length=args.length,
        auto_fps=args.auto_fps,
        auto_resolution=args.auto_resolution,
        video_codec=args.codec,
        file_prefix=args.prefix,
        file_postfix=args.postfix,
        file_type=args.format
    )

    print(result)


async def stop_recording_cmd(args):
    """Stop recording command"""
    # Use default folder path to find videos (relative to project root)
    manager = VideoRecorderManager(default_folder_path=get_default_videos_folder())
    result = await manager.stop_recording()
    print(result)


async def status_cmd(args):
    """Get status command"""
    manager = VideoRecorderManager()
    result = await manager.get_status()
    print(result)


async def record_cmd(args):
    """Record for a specific duration then stop"""
    # Use default folder path if not specified (relative to project root)
    default_folder = args.output if args.output else get_default_videos_folder()
    manager = VideoRecorderManager(default_folder_path=default_folder)

    print("Starting recording...")
    result = await manager.start_recording(
        camera_topic=args.camera_topic,
        fps=args.fps,
        image_height=args.height,
        image_width=args.width,
        overlay_timestamp=args.timestamp,
        folder_path=args.output,
        video_length=0,  # Continuous for the duration
        auto_fps=args.auto_fps,
        auto_resolution=args.auto_resolution,
        video_codec=args.codec,
        file_prefix=args.prefix,
        file_postfix=args.postfix,
        file_type=args.format
    )
    print(result)

    if "Error" in result:
        return

    print(f"\nRecording for {args.duration} seconds...")
    await asyncio.sleep(args.duration)

    print("\nStopping recording...")
    result = await manager.stop_recording()
    print(result)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Control ROS2 video recorder from the command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start recording")
    start_parser.add_argument(
        "--camera-topic", "-t",
        default="/camera_input",
        help="ROS2 camera topic (default: /camera_input)"
    )
    start_parser.add_argument(
        "--fps", "-f",
        type=int,
        default=30,
        help="Frame rate (default: 30)"
    )
    start_parser.add_argument(
        "--width", "-w",
        type=int,
        default=1280,
        help="Image width (default: 1280)"
    )
    start_parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Image height (default: 720)"
    )
    start_parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Overlay timestamp on frames"
    )
    start_parser.add_argument(
        "--output", "-o",
        help="Output folder path"
    )
    start_parser.add_argument(
        "--length", "-l",
        type=int,
        default=0,
        help="Video segment length in seconds (0 = continuous)"
    )
    start_parser.add_argument(
        "--auto-fps",
        action="store_true",
        help="Auto-detect topic framerate"
    )
    start_parser.add_argument(
        "--auto-resolution",
        action="store_true",
        help="Auto-detect image resolution from topic"
    )
    start_parser.add_argument(
        "--codec",
        default="mp4v",
        help="Video codec (default: mp4v)"
    )
    start_parser.add_argument(
        "--prefix",
        default="",
        help="Filename prefix"
    )
    start_parser.add_argument(
        "--postfix",
        default="",
        help="Filename postfix"
    )
    start_parser.add_argument(
        "--format",
        default="mp4",
        help="Video file format (default: mp4)"
    )
    start_parser.set_defaults(func=start_recording_cmd)

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop recording")
    stop_parser.set_defaults(func=stop_recording_cmd)

    # Status command
    status_parser = subparsers.add_parser("status", help="Check recording status")
    status_parser.set_defaults(func=status_cmd)

    # Record command (start, wait, stop)
    record_parser = subparsers.add_parser(
        "record",
        help="Record for a specific duration then stop"
    )
    record_parser.add_argument(
        "--duration", "-d",
        type=int,
        required=True,
        help="Recording duration in seconds"
    )
    record_parser.add_argument(
        "--camera-topic", "-t",
        default="/camera_input",
        help="ROS2 camera topic (default: /camera_input)"
    )
    record_parser.add_argument(
        "--fps", "-f",
        type=int,
        default=30,
        help="Frame rate (default: 30)"
    )
    record_parser.add_argument(
        "--width", "-w",
        type=int,
        default=1280,
        help="Image width (default: 1280)"
    )
    record_parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Image height (default: 720)"
    )
    record_parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Overlay timestamp on frames"
    )
    record_parser.add_argument(
        "--output", "-o",
        help="Output folder path"
    )
    record_parser.add_argument(
        "--auto-fps",
        action="store_true",
        help="Auto-detect topic framerate"
    )
    record_parser.add_argument(
        "--auto-resolution",
        action="store_true",
        help="Auto-detect image resolution from topic"
    )
    record_parser.add_argument(
        "--codec",
        default="mp4v",
        help="Video codec (default: mp4v)"
    )
    record_parser.add_argument(
        "--prefix",
        default="",
        help="Filename prefix"
    )
    record_parser.add_argument(
        "--postfix",
        default="",
        help="Filename postfix"
    )
    record_parser.add_argument(
        "--format",
        default="mp4",
        help="Video file format (default: mp4)"
    )
    record_parser.set_defaults(func=record_cmd)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Run the async command
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
