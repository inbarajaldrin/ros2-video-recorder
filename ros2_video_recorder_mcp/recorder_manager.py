#!/usr/bin/env python3
"""
VideoRecorderManager - Standalone ROS2 video recorder using rclpy and OpenCV

This module directly subscribes to ROS2 image topics and records video frames
without requiring an external video recorder service. It uses rclpy to interact
with the ROS2 network and OpenCV to encode and save video files.
"""

import asyncio
import threading
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

# Import ROS2 dependencies at module level
# These are needed for the class definition, not just type checking
try:
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
except ImportError as e:
    raise ImportError(
        "Failed to import rclpy. Make sure you are using the correct Python version "
        "(ROS2 Humble requires Python 3.10). "
        "Set your .venv to use Python 3.10 or use ROS2's Python directly."
    ) from e


class VideoRecorderNode(Node):
    """ROS2 node that subscribes to camera topics and records video"""

    def __init__(self, camera_topic: str, fps: int, width: int, height: int,
                 overlay_timestamp: bool, output_path: str, video_codec: str):
        """
        Initialize the video recorder node

        Args:
            camera_topic: ROS2 image topic to subscribe to
            fps: Frames per second for recording
            width: Video width in pixels
            height: Video height in pixels
            overlay_timestamp: Whether to overlay timestamp on frames
            output_path: Path to save the video file
            video_codec: Codec to use (e.g., 'mp4v', 'XVID')
        """
        super().__init__('video_recorder_node')

        self.camera_topic = camera_topic
        self.fps = fps
        self.width = width
        self.height = height
        self.overlay_timestamp = overlay_timestamp
        self.output_path = output_path
        self.video_codec = video_codec

        self.bridge = CvBridge()
        self.video_writer = None
        self.frame_count = 0
        self.is_recording = False
        self.lock = threading.Lock()

        # Subscribe to the camera topic
        self.subscription = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10
        )

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*video_codec)
        self.video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        if not self.video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")

        self.is_recording = True
        self.get_logger().info(f"Recording started: {output_path}")

    def image_callback(self, msg: Image):
        """Callback for image subscription"""
        if not self.is_recording:
            return

        try:
            # Convert ROS2 image to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Resize if necessary
            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                frame = cv2.resize(frame, (self.width, self.height))

            # Overlay timestamp if requested
            if self.overlay_timestamp:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                cv2.putText(
                    frame,
                    timestamp,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # Write frame to video
            with self.lock:
                if self.video_writer.isOpened():
                    self.video_writer.write(frame)
                    self.frame_count += 1

        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")

    def stop_recording(self):
        """Stop recording and release resources"""
        self.is_recording = False
        if self.video_writer and self.video_writer.isOpened():
            with self.lock:
                self.video_writer.release()
        self.get_logger().info(f"Recording stopped. Frames recorded: {self.frame_count}")


class VideoRecorderManager:
    """Manages ROS2 video recording using a background ROS2 node"""

    def __init__(self, default_folder_path: Optional[str] = None):
        """
        Initialize VideoRecorderManager

        Args:
            default_folder_path: Default folder path for videos
        """
        self.default_folder_path = default_folder_path or "videos"
        Path(self.default_folder_path).mkdir(parents=True, exist_ok=True)

        self.rclpy_initialized = False
        self.executor = None
        self.executor_thread = None
        self.recorder_node = None
        self.current_output_path = None

        # State file for tracking
        self.state_dir = Path("/tmp/ros2_video_recorder")
        self.state_dir.mkdir(exist_ok=True)
        self.state_file = self.state_dir / "recorder_state.json"

    def _initialize_rclpy(self):
        """Initialize rclpy if not already done"""
        if not self.rclpy_initialized:
            try:
                # Import and initialize rclpy here (lazy import)
                import rclpy
                rclpy.init()
                self.rclpy_initialized = True
            except RuntimeError:
                # rclpy already initialized
                self.rclpy_initialized = True
            except ImportError as e:
                raise ImportError(
                    "Failed to import rclpy. Make sure you are using the correct Python version "
                    "(ROS2 Humble requires Python 3.10, not 3.13). "
                    "Set your .venv to use Python 3.10 or use ROS2's Python directly."
                ) from e

    def _start_executor_thread(self):
        """Start the rclpy executor in a background thread"""
        if self.executor is None:
            from rclpy.executors import SingleThreadedExecutor
            self.executor = SingleThreadedExecutor()
            self.executor_thread = threading.Thread(
                target=self._executor_spin,
                daemon=True
            )
            self.executor_thread.start()

    def _executor_spin(self):
        """Run the executor spin loop"""
        while self.rclpy_initialized and self.recorder_node is not None:
            try:
                self.executor.spin_once(timeout_sec=0.1)
            except Exception as e:
                print(f"Executor error: {e}")
                break

    def _generate_filename(self, file_prefix: str = "", file_postfix: str = "",
                          file_type: str = "mp4") -> str:
        """Generate a timestamped video filename"""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"{file_prefix}{timestamp}{file_postfix}.{file_type}"
        return filename

    def _save_state(self, params: dict):
        """Save recording state to file"""
        state = {
            "recording": True,
            "params": params,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f)

    def _load_state(self) -> Optional[dict]:
        """Load recording state from file"""
        if not self.state_file.exists():
            return None
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None

    def _clear_state(self):
        """Clear the state file"""
        if self.state_file.exists():
            self.state_file.unlink()

    def is_recording(self) -> bool:
        """Check if recording is currently active"""
        state = self._load_state()
        return bool(state and state.get("recording")) and self.recorder_node is not None

    async def start_recording(
        self,
        camera_topic: str = "/camera_input",
        fps: int = 30,
        image_height: int = 720,
        image_width: int = 1280,
        overlay_timestamp: bool = False,
        folder_path: Optional[str] = None,
        video_length: int = 0,
        auto_fps: bool = False,
        auto_resolution: bool = False,
        video_codec: str = "mp4v",
        file_prefix: str = "",
        file_postfix: str = "",
        file_type: str = "mp4"
    ) -> str:
        """
        Start video recording from a ROS2 camera topic

        Args:
            camera_topic: ROS2 topic name for camera images
            fps: Frame rate for recording (1-120)
            image_height: Image height in pixels
            image_width: Image width in pixels
            overlay_timestamp: Whether to overlay timestamp on frames
            folder_path: Directory to save videos
            video_length: Length of video segment in seconds (0 = continuous)
            auto_fps: Auto-detect topic framerate (not yet implemented)
            auto_resolution: Auto-detect image resolution (not yet implemented)
            video_codec: Video codec to use (default: mp4v)
            file_prefix: Prefix for output filename
            file_postfix: Postfix for output filename
            file_type: Video file extension

        Returns:
            Status message indicating success or failure
        """

        if self.is_recording():
            return "Error: Recording is already in progress. Stop current recording first."

        try:
            # Initialize ROS2
            self._initialize_rclpy()

            # Use provided or default folder path
            if folder_path is None:
                folder_path = self.default_folder_path
            Path(folder_path).mkdir(parents=True, exist_ok=True)

            # Generate output filename
            filename = self._generate_filename(file_prefix, file_postfix, file_type)
            self.current_output_path = str(Path(folder_path) / filename)

            # Create and start the recorder node
            self.recorder_node = VideoRecorderNode(
                camera_topic=camera_topic,
                fps=fps,
                width=image_width,
                height=image_height,
                overlay_timestamp=overlay_timestamp,
                output_path=self.current_output_path,
                video_codec=video_codec
            )

            # Start executor thread if not already running
            self._start_executor_thread()

            # Add node to executor
            if self.executor:
                self.executor.add_node(self.recorder_node)

            # Give the subscription a moment to connect
            await asyncio.sleep(0.5)

            # Save state
            params = {
                "camera_topic": camera_topic,
                "fps": fps,
                "image_width": image_width,
                "image_height": image_height,
                "overlay_timestamp": overlay_timestamp,
                "video_length": video_length,
                "folder_path": folder_path,
                "file_prefix": file_prefix,
                "file_postfix": file_postfix,
                "file_type": file_type,
                "output_path": self.current_output_path,
            }
            self._save_state(params)

            result = "Recording started successfully!\n"
            result += f"Output file: {self.current_output_path}\n"
            result += f"Camera topic: {camera_topic}\n"
            result += f"FPS: {fps}\n"
            result += f"Resolution: {image_width}x{image_height}\n"
            result += f"Overlay timestamp: {overlay_timestamp}\n"
            result += f"Video length: {'continuous' if video_length == 0 else f'{video_length} seconds'}\n"

            return result

        except Exception as e:
            self._clear_state()
            self.recorder_node = None
            return f"Error: Failed to start recording. {str(e)}"

    async def stop_recording(self) -> str:
        """
        Stop the current recording

        Returns:
            Status message indicating success or failure
        """

        if not self.is_recording():
            return "No recording in progress."

        try:
            if self.recorder_node:
                self.recorder_node.stop_recording()

            # Give video writer time to flush
            await asyncio.sleep(0.5)

            # Clean up
            if self.executor and self.recorder_node:
                self.executor.remove_node(self.recorder_node)
            self.recorder_node = None

            # Load state to get output path
            state = self._load_state()
            output_path = None
            if state:
                output_path = state.get("params", {}).get("output_path")

            self._clear_state()

            result = "Recording stopped successfully.\n"
            if output_path and Path(output_path).exists():
                file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
                result += f"Video saved: {output_path}\n"
                result += f"File size: {file_size:.2f} MB\n"
            else:
                result += f"Output: {self.current_output_path}\n"

            return result

        except Exception as e:
            self._clear_state()
            self.recorder_node = None
            return f"Error stopping recording: {str(e)}"

    async def get_status(self) -> str:
        """
        Get current recording status

        Returns:
            Status message with details
        """
        if self.is_recording():
            state = self._load_state()
            if state:
                params = state.get("params", {})
                result = "Status: Recording in progress\n"
                result += f"Camera topic: {params.get('camera_topic', 'unknown')}\n"
                result += f"FPS: {params.get('fps', 'unknown')}\n"
                result += f"Resolution: {params.get('image_width', '?')}x{params.get('image_height', '?')}\n"
                if self.recorder_node:
                    result += f"Frames recorded: {self.recorder_node.frame_count}\n"
                return result
            return "Status: Recording in progress"
        else:
            return "Status: Not recording"

    async def cleanup(self):
        """Cleanup when shutting down"""
        if self.is_recording():
            await self.stop_recording()

        # Shutdown ROS2
        if self.executor_thread:
            self.executor_thread.join(timeout=2.0)
        if self.rclpy_initialized:
            try:
                import rclpy
                rclpy.shutdown()
            except Exception:
                pass


# Standalone usage example
async def main():
    """Example of using VideoRecorderManager standalone"""
    manager = VideoRecorderManager()

    print("Starting recording from /camera/image_raw...")
    result = await manager.start_recording(
        camera_topic="/camera/image_raw",
        fps=30,
        overlay_timestamp=True
    )
    print(result)

    # Record for 10 seconds
    print("Recording for 10 seconds...")
    for i in range(10):
        status = await manager.get_status()
        print(f"  {i+1}s: {status.splitlines()[0]}")
        await asyncio.sleep(1)

    print("\nStopping recording...")
    result = await manager.stop_recording()
    print(result)

    await manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
