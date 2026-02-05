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
from typing import Optional, Callable, Tuple, Union, List
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

    def __init__(self, camera_topic: str, fps: Optional[int], width: Optional[int], height: Optional[int],
                 overlay_timestamp: bool, output_path: str, video_codec: str,
                 auto_fps: bool = False, auto_resolution: bool = False,
                 on_values_detected: Optional[Callable[[], None]] = None):
        """
        Initialize the video recorder node

        Args:
            camera_topic: ROS2 image topic to subscribe to
            fps: Frames per second for recording (None if auto)
            width: Video width in pixels (None if auto)
            height: Video height in pixels (None if auto)
            overlay_timestamp: Whether to overlay timestamp on frames
            output_path: Path to save the video file
            video_codec: Codec to use (e.g., 'mp4v', 'XVID')
            auto_fps: Auto-detect FPS from topic
            auto_resolution: Auto-detect resolution from first frame
            on_values_detected: Optional callback when values are auto-detected
        """
        # Generate unique node name from topic to avoid conflicts with multiple recorders
        # Replace slashes and special chars with underscores for valid ROS2 node name
        topic_suffix = camera_topic.strip('/').replace('/', '_').replace('-', '_')
        node_name = f'video_recorder_{topic_suffix}'
        super().__init__(node_name)
        # Set logger level to WARN to suppress INFO messages
        self.get_logger().set_level(30)  # 30 = WARN level (suppresses INFO)

        self.camera_topic = camera_topic
        self.fps = fps
        self.width = width
        self.height = height
        self.overlay_timestamp = overlay_timestamp
        self.output_path = output_path
        self.video_codec = video_codec
        self.auto_fps = auto_fps
        self.auto_resolution = auto_resolution
        self.on_values_detected = on_values_detected

        self.bridge = CvBridge()
        self.video_writer = None
        self.frame_count = 0
        self.is_recording = False
        self.lock = threading.Lock()
        self.video_writer_initialized = False
        self.recording_start_time = None  # Track when recording actually starts (when first frame is written)
        self.first_frame_written = False  # Track if first frame has been written

        # For auto FPS detection
        self.frame_timestamps = []
        self.last_frame_time = None
        
        # Event to signal when values are detected (for async waiting)
        self.values_detected_event = None  # Will be set by manager if needed
        self.event_loop = None  # Will be set by manager if needed
        # Event to signal when first frame is written (for async waiting)
        self.first_frame_written_event = None  # Will be set by manager if needed

        # Subscribe to the camera topic
        self.subscription = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10
        )

        # Initialize video writer if all parameters are provided
        if fps is not None and width is not None and height is not None:
            fourcc = cv2.VideoWriter_fourcc(*video_codec)
            self.video_writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            if not self.video_writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {output_path}")

            self.video_writer_initialized = True
            self.is_recording = True
            self.get_logger().info(f"Recording started: {output_path}")
        else:
            # Will initialize on first frame
            self.is_recording = True
            self.get_logger().info(f"Recording started (auto-detect mode): {output_path}")

    def image_callback(self, msg: Image):
        """Callback for image subscription"""
        if not self.is_recording:
            return

        try:
            # Convert ROS2 image to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Auto-detect resolution from first frame if needed
            if not self.video_writer_initialized:
                if self.auto_resolution or self.height is None or self.width is None:
                    self.height = frame.shape[0]
                    self.width = frame.shape[1]
                    self.get_logger().info(f"Auto-detected resolution: {self.width}x{self.height}")

                # Auto-detect FPS if needed (will calculate from first few frames)
                fps_ready = False
                if self.auto_fps and self.fps is None:
                    # We need to detect FPS - collect frame intervals
                    current_time = datetime.now().timestamp()
                    if self.last_frame_time is not None:
                        frame_interval = current_time - self.last_frame_time
                        if frame_interval > 0:
                            self.frame_timestamps.append(frame_interval)
                            # Use average of last 10 intervals for FPS calculation
                            if len(self.frame_timestamps) >= 10:
                                avg_interval = sum(self.frame_timestamps[-10:]) / len(self.frame_timestamps[-10:])
                                self.fps = int(round(1.0 / avg_interval))
                                self.get_logger().info(f"Auto-detected FPS: {self.fps}")
                                fps_ready = True
                    self.last_frame_time = current_time

                    # Only initialize VideoWriter after FPS is detected
                    # This ensures we use the actual camera FPS, not a default
                    if not fps_ready:
                        # Skip this frame - we're still detecting FPS
                        return
                else:
                    # FPS is explicitly set or auto_fps is disabled - we can proceed
                    if self.fps is None:
                        self.fps = 30  # Default fallback
                    fps_ready = True

                # Initialize video writer now that we have all parameters (including detected FPS)
                fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
                self.video_writer = cv2.VideoWriter(
                    self.output_path,
                    fourcc,
                    self.fps,
                    (self.width, self.height)
                )

                if not self.video_writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer for {self.output_path}")

                self.video_writer_initialized = True
                # Don't set recording_start_time here - wait for first frame to be written
                self.get_logger().info(f"Video writer initialized: {self.width}x{self.height} @ {self.fps} fps")
                
                # Signal that values are detected (thread-safe)
                if self.values_detected_event and self.event_loop:
                    self.event_loop.call_soon_threadsafe(self.values_detected_event.set)
                
                # Notify callback if values were auto-detected
                if self.on_values_detected:
                    self.on_values_detected()

            # Resize if necessary (only if not auto-detected)
            if not self.auto_resolution and (frame.shape[0] != self.height or frame.shape[1] != self.width):
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
                if self.video_writer and self.video_writer.isOpened():
                    # Record start time when first frame is actually written
                    if not self.first_frame_written:
                        self.recording_start_time = datetime.now()
                        self.first_frame_written = True
                        # Signal that first frame has been written (thread-safe)
                        if self.first_frame_written_event and self.event_loop:
                            self.event_loop.call_soon_threadsafe(self.first_frame_written_event.set)
                    self.video_writer.write(frame)
                    self.frame_count += 1

        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")

    def stop_recording(self):
        """Stop recording and release resources"""
        with self.lock:
            # Set is_recording to False inside the lock to prevent race condition
            # where a frame could be mid-write when we call release()
            self.is_recording = False
            if self.video_writer and self.video_writer.isOpened():
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
        # Support multiple recorder nodes - dict keyed by camera topic
        self.recorder_nodes = {}
        # Track output paths for each topic
        self.output_paths = {}

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
        if self.executor is None or not self.executor_thread.is_alive():
            from rclpy.executors import SingleThreadedExecutor
            self.executor = SingleThreadedExecutor()
            self.executor_thread = threading.Thread(
                target=self._executor_spin,
                daemon=True
            )
            self.executor_thread.start()

    def _executor_spin(self):
        """Run the executor spin loop"""
        while self.rclpy_initialized and len(self.recorder_nodes) > 0:
            try:
                self.executor.spin_once(timeout_sec=0.1)
            except Exception as e:
                # Silently handle executor errors (usually happens during shutdown)
                break

    def _generate_filename(self, file_prefix: str = "", file_postfix: str = "",
                          file_type: str = "mp4") -> str:
        """Generate a timestamped video filename"""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"{file_prefix}{timestamp}{file_postfix}.{file_type}"
        return filename

    def _save_state(self, camera_topic: str, params: dict):
        """Save recording state to file for a specific topic"""
        # Load existing state
        state = self._load_state() or {"recordings": {}}

        # Add or update this recording
        state["recordings"][camera_topic] = {
            "recording": True,
            "params": params,
            "timestamp": datetime.now().isoformat(),
            "start_time": datetime.now().timestamp()  # Track actual start time for duration calculation
        }

        with open(self.state_file, "w") as f:
            json.dump(state, f)

    def _update_state_with_detected_values(self, camera_topic: str):
        """Update state file with auto-detected values from recorder node"""
        recorder_node = self.recorder_nodes.get(camera_topic)
        if recorder_node and recorder_node.video_writer_initialized:
            state = self._load_state()
            if state and camera_topic in state.get("recordings", {}):
                params = state["recordings"][camera_topic].get("params", {})
                params["fps"] = recorder_node.fps
                params["image_width"] = recorder_node.width
                params["image_height"] = recorder_node.height
                self._save_state(camera_topic, params)

    def _load_state(self) -> Optional[dict]:
        """Load recording state from file"""
        if not self.state_file.exists():
            return None
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None

    def _clear_state(self, camera_topic: Optional[str] = None):
        """Clear the state file or remove a specific recording from state

        Args:
            camera_topic: If provided, remove only this topic's state. If None, clear all.
        """
        if camera_topic is None:
            # Clear entire state file
            if self.state_file.exists():
                self.state_file.unlink()
        else:
            # Remove specific recording from state
            state = self._load_state()
            if state and "recordings" in state and camera_topic in state["recordings"]:
                del state["recordings"][camera_topic]
                # If no recordings left, delete the file
                if not state["recordings"]:
                    if self.state_file.exists():
                        self.state_file.unlink()
                else:
                    # Otherwise save the updated state
                    with open(self.state_file, "w") as f:
                        json.dump(state, f)

    def is_recording(self, camera_topic: Optional[str] = None) -> bool:
        """Check if recording is currently active

        Args:
            camera_topic: If provided, check if this specific topic is recording.
                         If None, check if any recording is active.

        Returns:
            True if recording(s) active, False otherwise
        """
        state = self._load_state()
        if not state or "recordings" not in state:
            return False

        if camera_topic is not None:
            # Check specific topic
            return (camera_topic in state["recordings"] and
                    state["recordings"][camera_topic].get("recording", False) and
                    camera_topic in self.recorder_nodes)
        else:
            # Check if any recording is active
            return (any(rec.get("recording", False) for rec in state["recordings"].values()) and
                    len(self.recorder_nodes) > 0)

    def _get_video_metadata(self, video_path: Path) -> Tuple[Optional[float], Optional[int], Optional[float]]:
        """
        Read actual video file metadata to get FPS, frame count, and duration.
        This is the source of truth - what's actually in the file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (duration_seconds, frame_count, fps) or (None, None, None) if error
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None, None, None
            
            # Get metadata from the actual video file
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            cap.release()
            
            # Calculate duration from file metadata
            if fps > 0 and frame_count > 0:
                duration_seconds = frame_count / fps
                return duration_seconds, frame_count, fps
            else:
                return None, frame_count if frame_count > 0 else None, fps if fps > 0 else None
                
        except Exception:
            # Silently fail - return None values
            return None, None, None

    async def _wait_for_video_complete(self, video_path: str, expected_frames: int, timeout: float = 10.0) -> bool:
        """
        Wait for the video file to be fully written and finalized.

        This is critical for MP4 files where the moov atom (index) is written at the end.
        The function waits until:
        1. The file size stabilizes (no more writes)
        2. The file can be opened and read by OpenCV
        3. OS buffers are flushed to disk

        Args:
            video_path: Path to the video file
            expected_frames: Number of frames we expect (from recorder tracking)
            timeout: Maximum time to wait in seconds

        Returns:
            True if file appears complete, False if timeout or error
        """
        import os
        import time

        start_time = time.time()
        last_size = -1
        stable_count = 0

        while time.time() - start_time < timeout:
            try:
                # Check current file size
                current_size = Path(video_path).stat().st_size

                # File size hasn't changed - might be stable
                if current_size == last_size and current_size > 0:
                    stable_count += 1

                    # After 3 consecutive checks with same size, verify file is readable
                    if stable_count >= 3:
                        # Force OS to flush buffers
                        try:
                            with open(video_path, 'r+b') as f:
                                os.fsync(f.fileno())
                        except Exception:
                            pass

                        # Verify file is readable by OpenCV
                        file_duration, file_frame_count, file_fps = self._get_video_metadata(Path(video_path))

                        if file_frame_count is not None and file_frame_count > 0:
                            # File is readable and has frames - consider it complete
                            # Note: frame count may differ slightly from expected due to timing
                            return True
                else:
                    # Size changed - reset stability counter
                    stable_count = 0
                    last_size = current_size

            except Exception:
                pass

            # Small delay between checks
            await asyncio.sleep(0.1)

        # Timeout - do final fsync and return
        try:
            with open(video_path, 'r+b') as f:
                os.fsync(f.fileno())
        except Exception:
            pass

        return False

    async def _start_single_recording(
        self,
        camera_topic: str,
        fps: Optional[int],
        image_height: Optional[int],
        image_width: Optional[int],
        overlay_timestamp: bool,
        folder_path: Optional[str],
        video_length: int,
        auto_fps: bool,
        auto_resolution: bool,
        video_codec: str,
        file_prefix: str,
        file_postfix: str,
        file_type: str
    ) -> str:
        """
        Internal method to start recording for a single camera topic.

        Returns:
            Status message indicating success or failure
        """
        # Check if this specific topic is already being recorded
        if self.is_recording(camera_topic):
            return f"Error: Recording is already in progress for topic '{camera_topic}'. Stop it first before starting a new recording on the same topic."

        try:
            # Initialize ROS2
            self._initialize_rclpy()

            # Only use auto modes if values are not explicitly set
            use_auto_fps = auto_fps and fps is None
            use_auto_resolution = auto_resolution and (image_width is None or image_height is None)

            # Use provided or default folder path
            if folder_path is None:
                folder_path = self.default_folder_path
            Path(folder_path).mkdir(parents=True, exist_ok=True)

            # Generate output filename
            filename = self._generate_filename(file_prefix, file_postfix, file_type)
            output_path = str(Path(folder_path) / filename)
            self.output_paths[camera_topic] = output_path

            # Create events for signaling when values are detected and when first frame is written
            values_detected_event = None
            first_frame_written_event = asyncio.Event()  # Always create this - we need to wait for first frame
            if use_auto_fps or use_auto_resolution:
                values_detected_event = asyncio.Event()
            event_loop = asyncio.get_event_loop()

            # Create callback that passes camera_topic to update function
            def on_values_detected():
                self._update_state_with_detected_values(camera_topic)

            # Create and start the recorder node
            recorder_node = VideoRecorderNode(
                camera_topic=camera_topic,
                fps=fps,
                width=image_width,
                height=image_height,
                overlay_timestamp=overlay_timestamp,
                output_path=output_path,
                video_codec=video_codec,
                auto_fps=use_auto_fps,
                auto_resolution=use_auto_resolution,
                on_values_detected=on_values_detected
            )

            # Set events and loop in node for signaling
            recorder_node.first_frame_written_event = first_frame_written_event
            recorder_node.event_loop = event_loop
            if values_detected_event:
                recorder_node.values_detected_event = values_detected_event

            # Store the recorder node
            self.recorder_nodes[camera_topic] = recorder_node

            # Start executor thread if not already running
            self._start_executor_thread()

            # Add node to executor
            if self.executor:
                self.executor.add_node(recorder_node)

            # Give executor a moment to start processing (allows first message to arrive)
            await asyncio.sleep(0.05)

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
                "output_path": output_path,
            }
            self._save_state(camera_topic, params)

            # Wait for first frame to be received and values to be detected (if auto-detection is enabled)
            values_detected = False
            if use_auto_fps or use_auto_resolution:
                # Check if already detected (might have happened during the sleep above)
                if recorder_node and recorder_node.video_writer_initialized:
                    values_detected = True
                    self._update_state_with_detected_values(camera_topic)
                elif not values_detected_event.is_set():
                    try:
                        # Wait for event with 5 second timeout
                        await asyncio.wait_for(values_detected_event.wait(), timeout=5.0)
                        values_detected = True
                        self._update_state_with_detected_values(camera_topic)
                    except asyncio.TimeoutError:
                        # Timeout - check if detection happened anyway (event might have been set)
                        if recorder_node and recorder_node.video_writer_initialized:
                            values_detected = True
                            self._update_state_with_detected_values(camera_topic)
                else:
                    # Event was already set
                    values_detected = True
                    self._update_state_with_detected_values(camera_topic)
            else:
                # Just give subscription time to connect
                await asyncio.sleep(0.4)

            # Wait for first frame to be written before returning
            # This ensures recording has actually started and no operation data is lost
            if not recorder_node.first_frame_written:
                try:
                    # Wait for first frame with timeout (should happen quickly after VideoWriter is initialized)
                    await asyncio.wait_for(first_frame_written_event.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    # Timeout - check if first frame was written anyway
                    if not recorder_node.first_frame_written:
                        # Clean up this failed recording
                        if self.executor:
                            self.executor.remove_node(recorder_node)
                        del self.recorder_nodes[camera_topic]
                        del self.output_paths[camera_topic]
                        self._clear_state(camera_topic)
                        return "Error: Timeout waiting for first frame. Check that the camera topic is publishing images."

            result = "Recording started successfully!\n"
            result += f"Output file: {output_path}\n"
            result += f"Camera topic: {camera_topic}\n"

            # Show actual FPS (detected or set)
            if values_detected and recorder_node:
                result += f"FPS: {recorder_node.fps}\n"
            elif fps is not None:
                result += f"FPS: {fps}\n"
            else:
                result += f"FPS: Auto-detect (detecting...)\n"

            # Show actual resolution (detected or set)
            if values_detected and recorder_node:
                result += f"Resolution: {recorder_node.width}x{recorder_node.height}\n"
            elif image_width is not None and image_height is not None:
                result += f"Resolution: {image_width}x{image_height}\n"
            else:
                result += f"Resolution: Auto-detect (detecting...)\n"

            result += f"Overlay timestamp: {overlay_timestamp}\n"
            result += f"Video length: {'continuous' if video_length == 0 else f'{video_length} seconds'}\n"

            return result

        except Exception as e:
            # Clean up this recording on error
            self._clear_state(camera_topic)
            if camera_topic in self.recorder_nodes:
                del self.recorder_nodes[camera_topic]
            if camera_topic in self.output_paths:
                del self.output_paths[camera_topic]
            return f"Error: Failed to start recording. {str(e)}"

    async def start_recording(
        self,
        camera_topic: Union[str, List[str]] = "/camera_input",
        fps: Optional[int] = None,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
        overlay_timestamp: bool = False,
        folder_path: Optional[str] = None,
        video_length: int = 0,
        auto_fps: bool = True,
        auto_resolution: bool = True,
        video_codec: str = "mp4v",
        file_prefix: str = "",
        file_postfix: str = "",
        file_type: str = "mp4"
    ) -> str:
        """
        Start video recording from one or more ROS2 camera topics

        Args:
            camera_topic: ROS2 topic name(s) for camera images. Can be a single string or a list of strings.
            fps: Frame rate for recording (1-120). If None, auto-detects from topic (default: None)
            image_height: Image height in pixels. If None, auto-detects from first frame (default: None)
            image_width: Image width in pixels. If None, auto-detects from first frame (default: None)
            overlay_timestamp: Whether to overlay timestamp on frames
            folder_path: Directory to save videos
            video_length: Length of video segment in seconds (0 = continuous)
            auto_fps: Auto-detect topic framerate (default: True). Disabled if fps is explicitly set
            auto_resolution: Auto-detect image resolution (default: True). Disabled if width/height are explicitly set
            video_codec: Video codec to use (default: mp4v)
            file_prefix: Prefix for output filename
            file_postfix: Postfix for output filename
            file_type: Video file extension

        Returns:
            Status message indicating success or failure for all topics
        """
        # Handle both single topic and list of topics
        topics = [camera_topic] if isinstance(camera_topic, str) else camera_topic

        if not topics:
            return "Error: No camera topics specified."

        # Start recording for each topic
        results = []
        for topic in topics:
            result = await self._start_single_recording(
                camera_topic=topic,
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
            results.append(result)

        # Return combined results
        if len(results) == 1:
            return results[0]
        else:
            return "\n---\n".join(results)

    async def stop_recording(self, camera_topic: Optional[str] = None) -> str:
        """
        Stop the current recording(s)

        Args:
            camera_topic: If provided, stop only this topic's recording.
                         If None, stop all active recordings.

        Returns:
            Status message indicating success or failure
        """

        if not self.is_recording():
            return "No recording in progress."

        # Determine which topics to stop
        topics_to_stop = []
        if camera_topic is not None:
            if not self.is_recording(camera_topic):
                return f"No recording in progress for topic '{camera_topic}'."
            topics_to_stop = [camera_topic]
        else:
            # Stop all recordings
            topics_to_stop = list(self.recorder_nodes.keys())

        if not topics_to_stop:
            return "No recording in progress."

        results = []

        for topic in topics_to_stop:
            try:
                recorder_node = self.recorder_nodes.get(topic)
                if not recorder_node:
                    continue

                # Capture frame count, fps, and start time before cleanup
                frame_count = recorder_node.frame_count
                fps = recorder_node.fps
                recording_start_time = recorder_node.recording_start_time

                # Load state to get output path BEFORE stopping (need for verification)
                state = self._load_state()
                output_path = None
                start_time = None
                if state and "recordings" in state and topic in state["recordings"]:
                    rec_state = state["recordings"][topic]
                    output_path = rec_state.get("params", {}).get("output_path")
                    start_time = rec_state.get("start_time")

                # Stop recording - this releases the video writer
                recorder_node.stop_recording()

                # Use node's start time if available (more accurate), otherwise use state
                if not recording_start_time and start_time:
                    recording_start_time = datetime.fromtimestamp(start_time)

                # Wait for video file to be fully written and finalized
                # MP4 files need the moov atom written at the end, which may take time
                if output_path and Path(output_path).exists():
                    await self._wait_for_video_complete(output_path, frame_count, timeout=10.0)

                # Calculate actual stop time from video file duration (most accurate)
                # This ensures stop time matches the actual video content, not when stop_recording() was called
                stop_time = None
                if output_path and Path(output_path).exists():
                    file_duration, file_frame_count, file_fps = self._get_video_metadata(Path(output_path))
                    if recording_start_time and file_duration:
                        # Calculate stop time based on start time + actual video duration
                        from datetime import timedelta
                        stop_time = recording_start_time + timedelta(seconds=file_duration)

                # Fallback to current time if we can't calculate from video
                if stop_time is None:
                    stop_time = datetime.now()

                # Clean up
                if self.executor and recorder_node:
                    self.executor.remove_node(recorder_node)
                del self.recorder_nodes[topic]
                if topic in self.output_paths:
                    del self.output_paths[topic]

                self._clear_state(topic)

                result = f"Recording stopped successfully for topic '{topic}'.\n"
                if output_path and Path(output_path).exists():
                    file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
                    result += f"Video saved: {output_path}\n"

                    # Read actual metadata from the video file (source of truth)
                    file_duration, file_frame_count, file_fps = self._get_video_metadata(Path(output_path))

                    # Use file metadata if available, otherwise fall back to tracked values
                    if file_frame_count is not None:
                        result += f"Frame count: {file_frame_count}\n"
                    elif frame_count > 0:
                        result += f"Frame count: {frame_count} (tracked, file metadata unavailable)\n"

                    if file_fps is not None:
                        result += f"FPS: {file_fps:.2f}\n"
                    elif fps is not None:
                        result += f"FPS: {fps} (tracked, file metadata unavailable)\n"

                    # Calculate duration from file metadata (most accurate)
                    if file_duration is not None:
                        duration_seconds = file_duration
                        minutes = int(duration_seconds // 60)
                        seconds_remainder = duration_seconds % 60
                        if minutes > 0:
                            if seconds_remainder == int(seconds_remainder):
                                result += f"Duration: {minutes}m {int(seconds_remainder)}s\n"
                            else:
                                result += f"Duration: {minutes}m {seconds_remainder:.1f}s\n"
                        else:
                            if seconds_remainder == int(seconds_remainder):
                                result += f"Duration: {int(seconds_remainder)}s\n"
                            else:
                                result += f"Duration: {seconds_remainder:.1f}s\n"
                    # Fallback to calculated duration if file metadata unavailable
                    elif fps and fps > 0 and frame_count > 0:
                        duration_seconds = frame_count / fps
                        minutes = int(duration_seconds // 60)
                        seconds = int(duration_seconds % 60)
                        if minutes > 0:
                            result += f"Duration: {minutes}m {seconds}s (calculated from tracked values)\n"
                        else:
                            result += f"Duration: {seconds}s (calculated from tracked values)\n"

                    result += f"File size: {file_size:.2f} MB\n"

                    # Add start and end timestamps
                    if recording_start_time:
                        result += f"Recording started: {recording_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    result += f"Recording stopped: {stop_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                else:
                    result += f"Output: {output_path or 'unknown'}\n"
                    if frame_count > 0:
                        result += f"Frame count: {frame_count}\n"
                    if recording_start_time:
                        result += f"Recording started: {recording_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    result += f"Recording stopped: {stop_time.strftime('%Y-%m-%d %H:%M:%S')}\n"

                results.append(result)

            except Exception as e:
                # Clean up this recording on error
                self._clear_state(topic)
                if topic in self.recorder_nodes:
                    del self.recorder_nodes[topic]
                if topic in self.output_paths:
                    del self.output_paths[topic]
                results.append(f"Error stopping recording for topic '{topic}': {str(e)}\n")

        return "\n".join(results)

    async def get_status(self) -> str:
        """
        Get current recording status for all active recordings

        Returns:
            Status message with details for all recordings
        """
        if not self.is_recording():
            return "Status: Not recording"

        state = self._load_state()
        if not state or "recordings" not in state or not state["recordings"]:
            return "Status: Not recording"

        results = []
        active_count = len(self.recorder_nodes)
        results.append(f"Status: {active_count} recording(s) in progress\n")

        for topic, recorder_node in self.recorder_nodes.items():
            rec_state = state["recordings"].get(topic, {})
            params = rec_state.get("params", {})

            result = f"Camera topic: {topic}\n"

            # Get actual FPS from recorder node if available (may be auto-detected)
            if recorder_node and recorder_node.video_writer_initialized:
                actual_fps = recorder_node.fps
                result += f"  FPS: {actual_fps}\n"
            else:
                result += f"  FPS: {params.get('fps', 'Auto-detect (not yet initialized)')}\n"

            # Get actual resolution from recorder node if available (may be auto-detected)
            if recorder_node and recorder_node.video_writer_initialized:
                actual_width = recorder_node.width
                actual_height = recorder_node.height
                result += f"  Resolution: {actual_width}x{actual_height}\n"
            else:
                width = params.get('image_width', '?')
                height = params.get('image_height', '?')
                if width == '?' or height == '?':
                    result += f"  Resolution: Auto-detect (not yet initialized)\n"
                else:
                    result += f"  Resolution: {width}x{height}\n"

            if recorder_node:
                result += f"  Frames recorded: {recorder_node.frame_count}\n"

            result += f"  Output: {params.get('output_path', 'unknown')}\n"

            results.append(result)

        return "\n".join(results)

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
