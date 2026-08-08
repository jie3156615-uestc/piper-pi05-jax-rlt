import numpy as np
import pytest

from piper_runtime.cameras import CAMERA_SPECS, FrameValidationError, validate_frame


def test_training_camera_mapping_is_exact():
    assert CAMERA_SPECS["camera1"].serial == "347522072112"
    assert CAMERA_SPECS["camera2"].serial == "260622272544"


def test_validate_frame_accepts_training_shape():
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    assert validate_frame("camera1", frame) is frame


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((240, 320, 3), dtype=np.uint8),
        np.zeros((480, 640, 3), dtype=np.float32),
    ],
)
def test_validate_frame_rejects_invalid_or_empty_frame(frame):
    with pytest.raises(FrameValidationError):
        validate_frame("camera1", frame)


def test_camera_specs_can_be_overridden_by_environment(monkeypatch):
    from piper_runtime.cameras import camera_specs_from_env

    monkeypatch.setenv("PIPER_CAMERA2_SERIAL", "315122272094")
    specs = camera_specs_from_env()

    assert specs["camera1"].serial == "347522072112"
    assert specs["camera2"].serial == "315122272094"
