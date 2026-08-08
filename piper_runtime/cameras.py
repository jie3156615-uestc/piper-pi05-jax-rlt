"""Dual RealSense color acquisition using the training camera mapping."""

import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


class FrameValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraSpec:
    serial: str
    width: int = 640
    height: int = 480
    fps: int = 30


CAMERA_SPECS = {
    "camera1": CameraSpec(serial="347522072112"),
    "camera2": CameraSpec(serial="260622272544"),
}


def camera_specs_from_env() -> Dict[str, CameraSpec]:
    return {
        "camera1": CameraSpec(serial=os.environ.get("PIPER_CAMERA1_SERIAL", CAMERA_SPECS["camera1"].serial)),
        "camera2": CameraSpec(serial=os.environ.get("PIPER_CAMERA2_SERIAL", CAMERA_SPECS["camera2"].serial)),
    }


def validate_frame(name: str, frame: np.ndarray) -> np.ndarray:
    spec = camera_specs_from_env()[name]
    if not isinstance(frame, np.ndarray):
        raise FrameValidationError("%s did not return a numpy frame" % name)
    if frame.shape != (spec.height, spec.width, 3):
        raise FrameValidationError("%s has shape %r" % (name, frame.shape))
    if frame.dtype != np.uint8:
        raise FrameValidationError("%s has dtype %s" % (name, frame.dtype))
    if not np.any(frame):
        raise FrameValidationError("%s returned an empty black frame" % name)
    return frame


class DualRealSenseReader:
    def __init__(self, rs_module=None):
        if rs_module is None:
            import pyrealsense2 as rs_module
        self.rs = rs_module
        self._pipelines: Dict[str, object] = {}
        self._specs = camera_specs_from_env()

    def start(self) -> None:
        if self._pipelines:
            return
        try:
            self._specs = camera_specs_from_env()
            for name, spec in self._specs.items():
                pipeline = self.rs.pipeline()
                config = self.rs.config()
                config.enable_device(spec.serial)
                config.enable_stream(
                    self.rs.stream.color,
                    spec.width,
                    spec.height,
                    self.rs.format.rgb8,
                    spec.fps,
                )
                pipeline.start(config)
                self._pipelines[name] = pipeline
        except Exception:
            self.stop()
            raise

    def read(self, timeout_ms: int = 5000, warmup_frames: int = 60) -> Dict[str, np.ndarray]:
        if set(self._pipelines) != set(self._specs):
            raise FrameValidationError("both camera pipelines must be started")
        images: Dict[str, np.ndarray] = {}
        for name, pipeline in self._pipelines.items():
            color = None
            for _ in range(warmup_frames):
                frames = pipeline.wait_for_frames(timeout_ms)
                color = frames.get_color_frame()
                if not color:
                    raise FrameValidationError("%s returned no color frame" % name)
            images[name] = validate_frame(name, np.asanyarray(color.get_data()))
        return images

    def stop(self) -> None:
        for pipeline in reversed(list(self._pipelines.values())):
            try:
                pipeline.stop()
            except Exception:
                pass
        self._pipelines.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop()
