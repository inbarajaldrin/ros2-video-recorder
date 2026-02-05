#!/usr/bin/env python3
"""
CameraCaptureManager - Standalone ROS2 camera capture using rclpy and OpenCV

This module captures single frames from ROS2 image topics and returns them
for analysis. It uses rclpy to interact with the ROS2 network and OpenCV
for image processing.
"""

import asyncio
import threading
import cv2
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Any
from dataclasses import dataclass

# Import ROS2 dependencies
try:
    from rclpy.node import Node
    from sensor_msgs.msg import Image as RosImage
    from cv_bridge import CvBridge
except ImportError as e:
    raise ImportError(
        "Failed to import rclpy. Make sure you are using the correct Python version "
        "(ROS2 Humble requires Python 3.10)."
    ) from e


@dataclass
class CaptureResult:
    """Result of a camera capture operation"""
    success: bool
    image_data: Optional[np.ndarray] = None  # RGB format
    message: str = ""
    topic: str = ""
    timestamp: str = ""
    saved_path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    is_fallback: bool = False


class CameraCaptureNode(Node):
    """ROS2 node that captures a single frame from a camera topic"""

    def __init__(self, camera_topic: str):
        """
        Initialize the camera capture node

        Args:
            camera_topic: ROS2 image topic to subscribe to
        """
        # Generate unique node name from topic
        topic_suffix = camera_topic.strip('/').replace('/', '_').replace('-', '_')
        node_name = f'camera_capture_{topic_suffix}'
        super().__init__(node_name)
        # Suppress INFO messages
        self.get_logger().set_level(30)  # WARN level

        self.camera_topic = camera_topic
        self.bridge = CvBridge()
        self.captured_frame = None
        self.frame_captured_event = None
        self.event_loop = None
        self.lock = threading.Lock()

        # Subscribe to the camera topic
        self.subscription = self.create_subscription(
            RosImage,
            camera_topic,
            self.image_callback,
            10
        )

    def image_callback(self, msg: RosImage):
        """Callback for image subscription - captures first frame and stops"""
        with self.lock:
            if self.captured_frame is not None:
                # Already captured a frame
                return

            try:
                # Convert ROS2 image to OpenCV format (BGR)
                frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                # Convert to RGB for return
                self.captured_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Signal that frame is captured
                if self.frame_captured_event and self.event_loop:
                    self.event_loop.call_soon_threadsafe(self.frame_captured_event.set)

            except Exception as e:
                self.get_logger().error(f"Error capturing frame: {e}")


class CameraCaptureManager:
    """Manages ROS2 camera capture operations"""

    def __init__(self, screenshots_dir: Optional[str] = None):
        """
        Initialize CameraCaptureManager

        Args:
            screenshots_dir: Directory to save screenshot backups
        """
        self.screenshots_dir = screenshots_dir or "screenshots"
        Path(self.screenshots_dir).mkdir(parents=True, exist_ok=True)

        self.rclpy_initialized = False
        self.executor = None
        self.executor_thread = None
        self._executor_lock = threading.Lock()

    def _initialize_rclpy(self):
        """Initialize rclpy if not already done"""
        if not self.rclpy_initialized:
            try:
                import rclpy
                rclpy.init()
                self.rclpy_initialized = True
            except RuntimeError:
                # rclpy already initialized
                self.rclpy_initialized = True
            except ImportError as e:
                raise ImportError(
                    "Failed to import rclpy. Make sure you are using the correct Python version "
                    "(ROS2 Humble requires Python 3.10)."
                ) from e

    def _ensure_executor(self):
        """Ensure executor is running"""
        with self._executor_lock:
            if self.executor is None:
                from rclpy.executors import SingleThreadedExecutor
                self.executor = SingleThreadedExecutor()

            # Start or restart executor thread if needed
            if self.executor_thread is None or not self.executor_thread.is_alive():
                self.executor_thread = threading.Thread(
                    target=self._executor_spin,
                    daemon=True
                )
                self.executor_thread.start()

    def _executor_spin(self):
        """Run the executor spin loop"""
        while self.rclpy_initialized:
            try:
                self.executor.spin_once(timeout_sec=0.1)
            except Exception:
                break

    def _get_latest_screenshot(self) -> Optional[Tuple[str, bytes]]:
        """
        Get the latest screenshot from the screenshots directory

        Returns:
            Tuple of (filename, raw_data) or None if no screenshots exist
        """
        try:
            if not os.path.exists(self.screenshots_dir):
                return None

            files = sorted([
                f for f in os.listdir(self.screenshots_dir)
                if f.endswith('.jpg') or f.endswith('.png')
            ])

            if not files:
                return None

            latest_file = os.path.join(self.screenshots_dir, files[-1])
            with open(latest_file, 'rb') as f:
                raw_data = f.read()
            return (files[-1], raw_data)

        except Exception:
            return None

    async def capture_image(self, topic_name: str, timeout: int = 10) -> CaptureResult:
        """
        Capture a single frame from a ROS2 image topic

        Args:
            topic_name: The ROS2 topic to subscribe to
            timeout: Timeout in seconds for image capture

        Returns:
            CaptureResult with image data and metadata
        """
        timestamp = datetime.now().isoformat()
        capture_node = None

        try:
            # Initialize ROS2
            self._initialize_rclpy()
            self._ensure_executor()

            # Create capture node
            capture_node = CameraCaptureNode(topic_name)

            # Set up event for signaling
            frame_captured_event = asyncio.Event()
            capture_node.frame_captured_event = frame_captured_event
            capture_node.event_loop = asyncio.get_event_loop()

            # Add node to executor
            if self.executor:
                self.executor.add_node(capture_node)

            # Wait for frame with timeout
            try:
                await asyncio.wait_for(frame_captured_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                # Cleanup node
                if self.executor and capture_node:
                    self.executor.remove_node(capture_node)

                # Try fallback to latest screenshot
                fallback = self._get_latest_screenshot()
                if fallback:
                    filename, raw_data = fallback
                    # Decode image for dimensions
                    nparr = np.frombuffer(raw_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        return CaptureResult(
                            success=True,
                            image_data=img_rgb,
                            message=f"Timeout after {timeout}s, using latest screenshot: {filename}",
                            topic=topic_name,
                            timestamp=timestamp,
                            saved_path=os.path.join(self.screenshots_dir, filename),
                            width=img_rgb.shape[1],
                            height=img_rgb.shape[0],
                            is_fallback=True
                        )

                return CaptureResult(
                    success=False,
                    message=f"Timeout after {timeout}s waiting for image from {topic_name}",
                    topic=topic_name,
                    timestamp=timestamp
                )

            # Get captured frame
            frame_rgb = capture_node.captured_frame

            # Cleanup node
            if self.executor and capture_node:
                self.executor.remove_node(capture_node)

            if frame_rgb is None:
                return CaptureResult(
                    success=False,
                    message=f"No image data received from {topic_name}",
                    topic=topic_name,
                    timestamp=timestamp
                )

            # Save backup screenshot
            save_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            topic_clean = topic_name.replace("/", "_").strip("_")
            filename = f"{save_timestamp}_{topic_clean}.jpg"
            save_path = os.path.join(self.screenshots_dir, filename)

            # Convert RGB to BGR for saving
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, frame_bgr)

            return CaptureResult(
                success=True,
                image_data=frame_rgb,
                message=f"Image captured from {topic_name}",
                topic=topic_name,
                timestamp=timestamp,
                saved_path=save_path,
                width=frame_rgb.shape[1],
                height=frame_rgb.shape[0],
                is_fallback=False
            )

        except Exception as e:
            # Cleanup on error
            if self.executor and capture_node:
                try:
                    self.executor.remove_node(capture_node)
                except Exception:
                    pass

            return CaptureResult(
                success=False,
                message=f"Error capturing image: {str(e)}",
                topic=topic_name,
                timestamp=timestamp
            )

    async def cleanup(self):
        """Cleanup when shutting down"""
        # Shutdown ROS2
        if self.executor_thread:
            self.executor_thread.join(timeout=2.0)
        if self.rclpy_initialized:
            try:
                import rclpy
                rclpy.shutdown()
            except Exception:
                pass


def numpy_to_base64_jpeg(image_rgb: np.ndarray, quality: int = 90) -> str:
    """
    Convert numpy RGB image to base64-encoded JPEG

    Args:
        image_rgb: Image in RGB format (numpy array)
        quality: JPEG quality (0-100)

    Returns:
        Base64-encoded JPEG string
    """
    import base64
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    # Encode as JPEG
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode('.jpg', image_bgr, encode_params)
    # Convert to base64
    return base64.b64encode(buffer).decode('utf-8')


def numpy_to_bytes_jpeg(image_rgb: np.ndarray, quality: int = 90) -> bytes:
    """
    Convert numpy RGB image to JPEG bytes

    Args:
        image_rgb: Image in RGB format (numpy array)
        quality: JPEG quality (0-100)

    Returns:
        JPEG image as bytes
    """
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    # Encode as JPEG
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode('.jpg', image_bgr, encode_params)
    return buffer.tobytes()
