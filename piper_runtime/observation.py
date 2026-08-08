"""Training-compatible observations and policy response validation."""

from typing import Dict

import numpy as np

from piper_runtime.cameras import CAMERA_SPECS, validate_frame


DEFAULT_PROMPT = 'Put the green block into the box.'


class PolicyResponseError(RuntimeError):
    pass


def build_observation(images: Dict[str, np.ndarray], state: np.ndarray, prompt: str = DEFAULT_PROMPT) -> dict:
    if set(images) != set(CAMERA_SPECS):
        raise ValueError("images must contain exactly camera1 and camera2")
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (7,) or not np.all(np.isfinite(state)):
        raise ValueError("state must be a finite seven-element vector")
    return {
        "observation/image": validate_frame("camera1", images["camera1"]),
        "observation/wrist_image": validate_frame("camera2", images["camera2"]),
        "observation/state": state.copy(),
        "prompt": str(prompt),
    }


def absolute_actions_to_delta(response: dict, state_snapshot: np.ndarray) -> np.ndarray:
    try:
        actions = np.asarray(response["actions"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PolicyResponseError("response has no valid actions") from exc
    state = np.asarray(state_snapshot, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] != 50 or actions.shape[1] < 7:
        raise PolicyResponseError("actions must have shape (50, >=7), got %r" % (actions.shape,))
    if state.shape != (7,):
        raise PolicyResponseError("state snapshot must have shape (7,)")
    selected = actions[:, :7].astype(np.float32, copy=True)
    if not np.all(np.isfinite(selected)) or not np.all(np.isfinite(state)):
        raise PolicyResponseError("policy output or state is non-finite")
    selected[:, :6] -= state[None, :6]
    return selected
