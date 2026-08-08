import dataclasses
import math
from typing import Literal

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


ROBOSUITE_ACTION_DIM = 7
ROBOSUITE_STATE_DIM = 8


@dataclasses.dataclass(frozen=True)
class RobosuiteTaskSpec:
    env_name: str
    prompt: str
    max_steps: int
    camera_name: str = "agentview"
    wrist_camera_name: str = "robot0_eye_in_hand"
    critical_predicate: str = "near_object"
    critical_threshold: float = 0.08


TASK_SPECS: dict[str, RobosuiteTaskSpec] = {
    "Lift": RobosuiteTaskSpec(
        env_name="Lift",
        prompt="lift the cube",
        max_steps=250,
        critical_predicate="near_object",
        critical_threshold=0.08,
    ),
    "PickPlaceCan": RobosuiteTaskSpec(
        env_name="PickPlaceCan",
        prompt="pick up the can and place it in the target bin",
        max_steps=400,
        critical_predicate="near_object",
        critical_threshold=0.10,
    ),
    "Door": RobosuiteTaskSpec(
        env_name="Door",
        prompt="open the door",
        max_steps=300,
        critical_predicate="near_handle",
        critical_threshold=0.10,
    ),
    "NutAssembly": RobosuiteTaskSpec(
        env_name="NutAssembly",
        prompt="place the nut onto the peg",
        max_steps=500,
        critical_predicate="near_object",
        critical_threshold=0.10,
    ),
    "Stack": RobosuiteTaskSpec(
        env_name="Stack",
        prompt="stack the red block on the green block",
        max_steps=400,
        critical_predicate="near_object",
        critical_threshold=0.10,
    ),
}


def make_robosuite_example() -> dict:
    return {
        "observation/state": np.random.rand(ROBOSUITE_STATE_DIM).astype(np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "lift the cube",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).clip(0, 255).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return np.ascontiguousarray(image)


@dataclasses.dataclass(frozen=True)
class RobosuiteInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class RobosuiteOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ROBOSUITE_ACTION_DIM], dtype=np.float32)}


def quat2axisangle(quat) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(float(quat[3]))) / den).astype(np.float32)


def proprio_state(obs: dict) -> np.ndarray:
    gripper = np.asarray(obs.get("robot0_gripper_qpos", np.zeros(2)), dtype=np.float32)
    if gripper.shape[0] < 2:
        gripper = np.pad(gripper, (0, 2 - gripper.shape[0]))
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat2axisangle(obs["robot0_eef_quat"]),
            gripper[:2],
        ]
    ).astype(np.float32)


def make_policy_input(
    obs: dict,
    task: str | RobosuiteTaskSpec,
    *,
    camera_name: str | None = None,
    wrist_camera_name: str | None = None,
) -> dict:
    spec = get_task_spec(task)
    camera_name = camera_name or spec.camera_name
    wrist_camera_name = wrist_camera_name or spec.wrist_camera_name
    return {
        "observation/image": _parse_image(obs[f"{camera_name}_image"]),
        "observation/wrist_image": _parse_image(obs[f"{wrist_camera_name}_image"]),
        "observation/state": proprio_state(obs),
        "prompt": spec.prompt,
    }


def get_task_spec(task: str | RobosuiteTaskSpec) -> RobosuiteTaskSpec:
    if isinstance(task, RobosuiteTaskSpec):
        return task
    if task not in TASK_SPECS:
        raise ValueError(f"Unknown robosuite task {task!r}. Available: {sorted(TASK_SPECS)}")
    return TASK_SPECS[task]


def make_env(
    task: str | RobosuiteTaskSpec,
    *,
    seed: int,
    image_size: int = 256,
    control_freq: int = 20,
    hard_reset: bool = False,
):
    import robosuite as suite
    from robosuite.controllers import load_controller_config

    spec = get_task_spec(task)
    controller = load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        env_name=spec.env_name,
        robots="Panda",
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=[spec.camera_name, spec.wrist_camera_name],
        camera_heights=image_size,
        camera_widths=image_size,
        control_freq=control_freq,
        horizon=spec.max_steps,
        ignore_done=True,
        hard_reset=hard_reset,
    )
    env.seed(seed)
    return env


def is_success(env) -> bool:
    checker = getattr(env, "_check_success", None)
    if checker is None:
        return False
    return bool(checker())


def action_bounds(env) -> tuple[np.ndarray, np.ndarray]:
    low, high = env.action_spec
    return np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)


def clip_action(env, action: np.ndarray) -> np.ndarray:
    low, high = action_bounds(env)
    return np.clip(np.asarray(action, dtype=np.float32), low, high)


def _first_distance(obs: dict, suffix: str = "_pos") -> float | None:
    candidates = []
    for key, value in obs.items():
        if key.startswith("gripper_to_") and key.endswith(suffix):
            arr = np.asarray(value, dtype=np.float32)
            candidates.append(float(np.linalg.norm(arr)))
    if not candidates:
        return None
    return min(candidates)


def critical_phase_active(
    obs: dict,
    env,
    *,
    mode: Literal["full", "window", "predicate"],
    step: int,
    start_step: int = 0,
    end_step: int | None = None,
    predicate: str = "near_object",
    threshold: float = 0.10,
) -> bool:
    if mode == "full":
        return True
    if mode == "window":
        return step >= start_step and (end_step is None or step < end_step)
    if mode != "predicate":
        raise ValueError(f"Unknown critical phase mode: {mode}")

    if predicate in {"near_object", "near_handle", "near_target"}:
        distance = _first_distance(obs)
        return distance is not None and distance < threshold
    if predicate == "object_grasped":
        gripper = getattr(env, "robots", [None])[0].gripper if getattr(env, "robots", None) else None
        objects = getattr(env, "objects", [])
        check_grasp = getattr(env, "_check_grasp", None)
        if gripper is not None and check_grasp is not None:
            return any(bool(check_grasp(gripper, obj)) for obj in objects)
        return False
    if predicate == "success":
        return is_success(env)
    raise ValueError(f"Unknown critical predicate: {predicate}")
