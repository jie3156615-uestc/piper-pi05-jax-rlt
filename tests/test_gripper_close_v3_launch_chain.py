#!/usr/bin/env python3
"""Static, side-effect-free audit of the gripper-close v3 shell launch chain."""

from __future__ import annotations

import ast
import os
from pathlib import Path


ROOT = Path(
    os.environ.get("GRIPPER_V3_SCRIPT_ROOT", Path(__file__).resolve().parent)
).resolve()


def _resolve_deploy_root() -> Path:
    """Locate the exact runtime/workspace paired with the launch scripts.

    On the 5090 the test is installed at the self-contained staging root, so
    ``ROOT`` is authoritative.  Local development overlays can select their
    candidate explicitly.  The two fallback candidates only make the checked-in
    static test convenient to run before deployment.
    """

    override = os.environ.get("GRIPPER_V3_DEPLOY_ROOT")
    if override:
        return Path(override).resolve()
    candidates = (
        ROOT,
        ROOT.parent,
        ROOT.parent / ".codex_projection_fix",
        ROOT.parent / "remote_piper_runtime",
    )
    for candidate in candidates:
        if (candidate / "piper_runtime" / "rlt_actor_protocol.py").is_file():
            return candidate.resolve()
    raise AssertionError(
        "cannot locate the gripper-v3 runtime; set GRIPPER_V3_DEPLOY_ROOT "
        "to the self-contained staging root"
    )


DEPLOY_ROOT = _resolve_deploy_root()
HOOK = ROOT / "run_rlt_online_update_hook_gripper_close_v3.sh"
LAUNCHER = Path(
    os.environ.get(
        "GRIPPER_V3_LAUNCHER",
        ROOT / "run_rlt_lineage_gripper_close_v3_online.sh",
    )
).resolve()
SESSION = Path(
    os.environ.get(
        "GRIPPER_V3_SESSION_WRAPPER",
        (
            LAUNCHER.parent / "run_rlt_online_session_gripper_close_v3.sh"
            if (
                LAUNCHER.parent / "run_rlt_online_session_gripper_close_v3.sh"
            ).is_file()
            else ROOT / "run_rlt_online_session_gripper_close_v3.sh"
        ),
    )
).resolve()
CURRENT = ROOT / "run_greenblock_gripper_close_v3_current.sh"
ACTOR_PROTOCOL = DEPLOY_ROOT / "piper_runtime" / "rlt_actor_protocol.py"
ONLINE_SESSION_RUNTIME = DEPLOY_ROOT / "piper_runtime" / "rlt_online_session.py"
ROLLOUT_RUNTIME = DEPLOY_ROOT / "piper_runtime" / "rlt_takeover_rollout.py"
SHADOW_POLICY_RUNTIME = DEPLOY_ROOT / "piper_runtime" / "rlt_shadow_policy.py"
SHADOW_SERVICE_RUNTIME = (
    DEPLOY_ROOT / "piper_runtime" / "rlt_shadow_policy_service.py"
)
ACTOR_RUNTIME = DEPLOY_ROOT / "src" / "openpi" / "rlt" / "real" / "actor_runtime.py"

RAW_SCHEMA = (
    "piper_joint_delta_v5_c10_n10_stride10_behavior_ref50_"
    "rank1_joint_r005_d1_0015_d2_001_cone15_gripper_close_knot_r005"
)
EXECUTION_SCHEMA = (
    "piper_joint_delta_v5_c10_n10_stride10_behavior_ref50_"
    "persistent_filtered_actual_r005_d1_0015_d2_001_cone15_"
    "gripper_close_assist_r005_d1_0005_d2_0003_boundary0005"
)
GOVERNOR = (
    "persistent_governor_v3_joint_r005_d1_0015_d2_001_cone15_"
    "boundary060_gripper_close_r005_d1_0005_d2_0003_boundary0005"
)
PROJECTION = (
    "rank1_joint_v1_r005_d1_0015_d2_001_cone15_"
    "scale33_min020_gripper_close_knot_r005"
)
LEGACY_PROJECTION = (
    "rank1_bump_v1_r005_d1_0015_d2_001_cone15_"
    "scale33_min020_gripper_frozen"
)


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{label}: missing {needle!r}")


def reject(text: str, needle: str, label: str) -> None:
    if needle in text:
        raise AssertionError(f"{label}: forbidden legacy token {needle!r}")


def literal_string_assignment(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError) as exc:
            raise AssertionError(
                f"{path}: {name} is not a literal string assignment"
            ) from exc
        if not isinstance(value, str):
            raise AssertionError(f"{path}: {name} is not a string")
        return value
    raise AssertionError(f"{path}: missing assignment {name}")


def name_mapping_assignment(path: Path, name: str) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            raise AssertionError(f"{path}: {name} is not a dict literal")
        result: dict[str, str] = {}
        for key, value in zip(node.value.keys, node.value.values):
            if not isinstance(key, ast.Name) or not isinstance(value, ast.Name):
                raise AssertionError(
                    f"{path}: {name} must map named schemas to named profiles"
                )
            result[key.id] = value.id
        return result
    raise AssertionError(f"{path}: missing assignment {name}")


def audit_runtime_projection_chain(launcher: str) -> None:
    runtime_paths = (
        ACTOR_PROTOCOL,
        ONLINE_SESSION_RUNTIME,
        ROLLOUT_RUNTIME,
        SHADOW_POLICY_RUNTIME,
        SHADOW_SERVICE_RUNTIME,
        ACTOR_RUNTIME,
    )
    for path in runtime_paths:
        if not path.is_file():
            raise AssertionError(
                f"self-contained gripper-v3 runtime is incomplete: {path}"
            )

    protocol = ACTOR_PROTOCOL.read_text(encoding="utf-8")
    online_session = ONLINE_SESSION_RUNTIME.read_text(encoding="utf-8")
    rollout = ROLLOUT_RUNTIME.read_text(encoding="utf-8")
    shadow_policy = SHADOW_POLICY_RUNTIME.read_text(encoding="utf-8")
    shadow_service = SHADOW_SERVICE_RUNTIME.read_text(encoding="utf-8")
    actor_runtime = ACTOR_RUNTIME.read_text(encoding="utf-8")

    actual_projection = literal_string_assignment(
        ACTOR_PROTOCOL, "RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE"
    )
    if actual_projection != PROJECTION:
        raise AssertionError(
            "runtime gripper-v3 projection differs from launcher contract: "
            f"{actual_projection!r} != {PROJECTION!r}"
        )
    actual_legacy_projection = literal_string_assignment(
        ACTOR_PROTOCOL, "ACTOR_PROJECTION_PROFILE"
    )
    if actual_legacy_projection != LEGACY_PROJECTION:
        raise AssertionError(
            "legacy projection identity changed unexpectedly: "
            f"{actual_legacy_projection!r}"
        )

    for needle in (
        "RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE",
        "ACTOR_PROJECTION_PROFILE_BY_ACTION_SCHEMA",
    ):
        require(protocol, needle, "runtime actor protocol")
    schema_projection_map = name_mapping_assignment(
        ACTOR_PROTOCOL, "ACTOR_PROJECTION_PROFILE_BY_ACTION_SCHEMA"
    )
    expected_mapping = {
        "ACTION_SCHEMA_FINGERPRINT": "ACTOR_PROJECTION_PROFILE",
        "RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT": (
            "RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE"
        ),
    }
    if schema_projection_map != expected_mapping:
        raise AssertionError(
            "runtime schema/projection map is not the exact legacy+v3 contract: "
            f"{schema_projection_map!r}"
        )

    for text, label in (
        (online_session, "runtime online session"),
        (rollout, "runtime rollout"),
    ):
        require(
            text,
            "RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE",
            label,
        )
        require(text, "expected_projection_profile = (", label)
        require(
            text,
            "if self.actor_projection_profile != expected_projection_profile:",
            label,
        )
        reject(
            text,
            "if self.actor_projection_profile != ACTOR_PROJECTION_PROFILE:",
            label,
        )

    for needle in (
        "ACTOR_PROJECTION_PROFILE_BY_ACTION_SCHEMA",
        "expected_projection = ACTOR_PROJECTION_PROFILE_BY_ACTION_SCHEMA.get(",
        "self.actor_action_schema_fingerprint",
        "self.config.actor_projection_profile",
    ):
        require(shadow_policy, needle, "runtime shadow policy")
    for forbidden in (
        '"actor_projection_profile": ACTOR_PROJECTION_PROFILE',
        '"a_actor_projection_profile"] = ACTOR_PROJECTION_PROFILE',
        '"action_schema_fingerprint": ACTION_SCHEMA_FINGERPRINT',
        '"a_actor_action_schema_fingerprint"] = ACTION_SCHEMA_FINGERPRINT',
    ):
        reject(shadow_policy, forbidden, "runtime shadow policy metadata")

    for needle in (
        "_resolve_actor_wire_contract(",
        '"output_actor_projection_profile"',
        "ACTOR_PROJECTION_PROFILE_BY_ACTION_SCHEMA.get(",
        '"PIPER_RLT_EXPECTED_PROJECTION_PROFILE"',
        "launcher/loaded Actor projection mismatch",
        "actor_projection_profile=actor_projection_profile",
    ):
        require(shadow_service, needle, "runtime shadow service")

    for needle in (
        "self.output_action_schema_fingerprint",
        "self.output_actor_projection_profile",
        "RANK1_GRIPPER_CLOSE_PROJECTION_PROFILE",
        '"actor_output_projection_profile"',
    ):
        require(actor_runtime, needle, "checkpoint Actor runtime")

    for needle in (
        "POLICY_IMPORT_AUDIT=",
        "from piper_runtime.rlt_online_session import SessionRuntimeConfig",
        "from piper_runtime.rlt_takeover_rollout import TakeoverRuntimeConfig",
        "SessionRuntimeConfig(**contract_kwargs).validate()",
        "TakeoverRuntimeConfig(**contract_kwargs).validate()",
        'print("session_and_rollout_contract=validated")',
        'EXPECTED_PROJECTION="${PIPER_RLT_EXPECTED_PROJECTION_PROFILE:?}"',
        "RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE",
        'Environment="PIPER_RLT_EXPECTED_PROJECTION_PROFILE=$EXPECTED_PROJECTION"',
    ):
        require(launcher, needle, "generated shadow-service projection audit")
    if launcher.index("POLICY_IMPORT_AUDIT=") > launcher.index(
        'if [[ "$DRY_RUN" == "1" ]]'
    ):
        raise AssertionError(
            "launcher dry-run exits before constructing and validating the "
            "session/rollout v3 projection contract"
        )


def audit_ros_runtime_import_preflight(launcher: str, session: str) -> None:
    """Ensure dry-run covers the exact Python used for the live ROS session."""

    for needle in (
        'PIKA_PYTHON="${PIPER_RLT_PIKA_PYTHON:-$HOME/venvs/pika/bin/python}"',
        "ROS_RUNTIME_IMPORT_AUDIT=",
        '"$PIKA_PYTHON" -',
        "import rospkg",
        "import catkin_pkg",
        "import rospy",
        "from sensor_msgs.msg import JointState",
        "from piper_runtime.ros_command_io import require_ros_modules",
        "resolved_rospy, resolved_joint_state = require_ros_modules()",
        "SessionRuntimeConfig(**contract_kwargs).validate()",
        "TakeoverRuntimeConfig(**contract_kwargs).validate()",
        'print("ros_session_and_rollout_import=validated")',
        'export PIPER_RLT_ROS_PYTHON="$PIKA_PYTHON"',
    ):
        require(launcher, needle, "ROS runtime import preflight")

    dry_run_index = launcher.index('if [[ "$DRY_RUN" == "1" ]]')
    if launcher.index("ROS_RUNTIME_IMPORT_AUDIT=") > dry_run_index:
        raise AssertionError(
            "launcher dry-run exits before importing the live ROS dependency "
            "chain"
        )
    if launcher.index(
        '[[ -x "$PIKA_PYTHON" ]]'
    ) > dry_run_index:
        raise AssertionError(
            "launcher dry-run exits before checking its live ROS interpreter"
        )
    if launcher.index(
        'print("ros_session_and_rollout_import=validated")'
    ) > dry_run_index:
        raise AssertionError(
            "launcher dry-run exits before validating Session/Takeover under "
            "the live ROS interpreter"
        )

    for needle in (
        'ROS_PYTHON="${PIPER_RLT_ROS_PYTHON:-$HOME/venvs/pika/bin/python}"',
        '[[ -x "$ROS_PYTHON" ]]',
        'printf \'%q \' "$ROS_PYTHON" "${ARGS[@]}" "$@"',
        'exec "$ROS_PYTHON" "${ARGS[@]}" "$@"',
    ):
        require(session, needle, "live ROS session interpreter")
    reject(
        session,
        'exec "$PROJECT_PYTHON" "${ARGS[@]}" "$@"',
        "live ROS session interpreter",
    )


def main() -> None:
    paths = (HOOK, SESSION, LAUNCHER, CURRENT)
    for path in paths:
        if not path.is_file():
            raise AssertionError(f"missing launch-chain file: {path}")
    hook, session, launcher, current = (
        path.read_text(encoding="utf-8") for path in paths
    )

    for label, text in zip(
        ("hook", "session", "launcher"), (hook, session, launcher)
    ):
        require(text, RAW_SCHEMA, label)
        require(text, EXECUTION_SCHEMA, label)
        require(text, GOVERNOR, label)
        require(text, "close_only_persistent_v1", label)
        require(text, ".venv/bin/python", label)
        reject(text, "gripper_absolute_frozen_residual", label)
        reject(text, "rank1_bump_v1", label)

    hook_mapping = {
        "RLT_BETA_HUMAN_GRIPPER_BC": "--beta-human-gripper-bc",
        "RLT_HUMAN_GRIPPER_BC_SCALE_M": "--human-gripper-bc-scale-m",
        "RLT_HUMAN_GRIPPER_Q_FILTER_MODE": "--human-gripper-q-filter-mode",
        "RLT_HUMAN_GRIPPER_Q_FILTER_MARGIN": "--human-gripper-q-filter-margin",
        "RLT_GRIPPER_RESIDUAL_MODE": "--gripper-residual-mode",
        "RLT_GRIPPER_RESIDUAL_MAX": "--gripper-residual-max",
        "RLT_GRIPPER_RESIDUAL_D1_MAX_M": "--gripper-residual-d1-max-m",
        "RLT_GRIPPER_RESIDUAL_D2_MAX_M": "--gripper-residual-d2-max-m",
        "RLT_GRIPPER_MAX_BOUNDARY_JUMP_M": "--gripper-max-boundary-jump-m",
        "RLT_GRIPPER_COMMAND_MIN_M": "--gripper-command-min-m",
        "RLT_GRIPPER_COMMAND_MAX_M": "--gripper-command-max-m",
        "RLT_GRIPPER_RELEASE_REFERENCE_M": "--gripper-release-reference-m",
        "RLT_GRIPPER_RELEASE_DELTA_M": "--gripper-release-delta-m",
    }
    for config_key, flag in hook_mapping.items():
        require(hook, f'require_config_value {config_key} "', "hook guard")
        require(hook, f'{flag} "${config_key}"', "hook updater mapping")
    reject(hook, "\n  --freeze-gripper-residual", "hook updater mapping")
    require(
        hook,
        '--min-admitted-human-episodes "$RLT_MIN_ADMITTED_HUMAN_EPISODES"',
        "hook admitted-human guard",
    )
    require(
        hook,
        'require_config_value RLT_MIN_SUCCESS_HUMAN_EPISODES "0"',
        "hook reward-independent human guard",
    )

    session_mapping = {
        "ACTOR_GRIPPER_RESIDUAL_MODE": "--actor-gripper-residual-mode",
        "ACTOR_GRIPPER_RESIDUAL_MAX_CLOSE_M": (
            "--actor-gripper-residual-max-close-m"
        ),
        "ACTOR_GRIPPER_RESIDUAL_D1_MAX_M": (
            "--actor-gripper-residual-d1-max-m"
        ),
        "ACTOR_GRIPPER_RESIDUAL_D2_MAX_M": (
            "--actor-gripper-residual-d2-max-m"
        ),
        "ACTOR_GRIPPER_MAX_BOUNDARY_JUMP_M": (
            "--actor-gripper-max-boundary-jump-m"
        ),
        "ACTOR_GRIPPER_COMMAND_MIN_M": "--actor-gripper-command-min-m",
        "ACTOR_GRIPPER_COMMAND_MAX_M": "--actor-gripper-command-max-m",
        "ACTOR_GRIPPER_RELEASE_REFERENCE_M": (
            "--actor-gripper-release-reference-m"
        ),
        "ACTOR_GRIPPER_RELEASE_DELTA_M": "--actor-gripper-release-delta-m",
    }
    for env_key, flag in session_mapping.items():
        require(session, f'require_value {env_key} "${env_key}"', "session guard")
        require(session, f'{flag} "${env_key}"', "session runtime mapping")

    launcher_mapping = {
        "RLT_GRIPPER_RESIDUAL_MODE": "ACTOR_GRIPPER_RESIDUAL_MODE",
        "RLT_GRIPPER_RESIDUAL_MAX": "ACTOR_GRIPPER_RESIDUAL_MAX_CLOSE_M",
        "RLT_GRIPPER_RESIDUAL_D1_MAX_M": "ACTOR_GRIPPER_RESIDUAL_D1_MAX_M",
        "RLT_GRIPPER_RESIDUAL_D2_MAX_M": "ACTOR_GRIPPER_RESIDUAL_D2_MAX_M",
        "RLT_GRIPPER_MAX_BOUNDARY_JUMP_M": (
            "ACTOR_GRIPPER_MAX_BOUNDARY_JUMP_M"
        ),
        "RLT_GRIPPER_COMMAND_MIN_M": "ACTOR_GRIPPER_COMMAND_MIN_M",
        "RLT_GRIPPER_COMMAND_MAX_M": "ACTOR_GRIPPER_COMMAND_MAX_M",
        "RLT_GRIPPER_RELEASE_REFERENCE_M": (
            "ACTOR_GRIPPER_RELEASE_REFERENCE_M"
        ),
        "RLT_GRIPPER_RELEASE_DELTA_M": "ACTOR_GRIPPER_RELEASE_DELTA_M",
    }
    for config_key, env_key in launcher_mapping.items():
        require(
            launcher,
            f'export {env_key}="${config_key}"',
            "launcher session mapping",
        )
    require(
        launcher,
        'ONLINE_SESSION_SCRIPT="$SCRIPT_DIR/'
        'run_rlt_online_session_gripper_close_v3.sh"',
        "launcher explicit session",
    )
    require(
        launcher,
        'UPDATE_HOOK_SCRIPT="$SCRIPT_DIR/'
        'run_rlt_online_update_hook_gripper_close_v3.sh"',
        "launcher explicit hook",
    )
    require(
        launcher,
        'V3_NATIVE_SERVICE="rlt-native-sdk-command-gripper-v3.service"',
        "launcher native unit",
    )
    require(
        launcher,
        'V3_SHADOW_SERVICE="openpi-rlt-shadow-policy-gripper-v3.service"',
        "launcher policy unit",
    )
    require(
        hook,
        'EXPECTED_SHADOW_SERVICE="openpi-rlt-shadow-policy-gripper-v3.service"',
        "hook promotion unit",
    )
    for selected_path in (
        "WorkingDirectory=$RUNTIME",
        "WorkingDirectory=$WORKSPACE",
        "ExecStart=$PIKA_PYTHON -u -m piper_runtime.native_sdk_command_bridge",
        "PIPER_RLT_PIKA_PYTHON:-$HOME/venvs/pika/bin/python",
        "JAX_COMPILATION_CACHE_DIR=$RUNTIME/jax_cache_gripper_v3",
        "PIPER_RLT_SELECTED_ACTOR_FILE=$RLT_SELECTED_ACTOR_FILE",
        "PIPER_RLT_EXPECTED_RUNTIME=$RUNTIME",
        "PIPER_RLT_EXPECTED_RAW_SCHEMA=$EXPECTED_RAW_SCHEMA",
        "StandardOutput=append:$RUNTIME/logs/",
    ):
        require(launcher, selected_path, "generated v3 units")
    require(
        launcher,
        "import piper_runtime.rlt_shadow_policy_service as service",
        "policy import audit",
    )
    require(
        launcher,
        "RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT",
        "policy raw-schema audit",
    )
    reject(
        launcher,
        "/home/cwzk/piper_jax_inference_v1",
        "generated v3 units",
    )
    reject(
        launcher,
        "/home/cwzk/openpi_jax_piper_lora_v1_20260707",
        "generated v3 units",
    )
    for old_command in (
        'systemctl --user restart rlt-native-sdk-command.service',
        'systemctl --user start rlt-native-sdk-command.service',
        'systemctl --user restart openpi-rlt-shadow-policy.service',
        'systemctl --user stop openpi-piper-policy.service',
        "Conflicts=rlt-native-sdk-command.service",
        "Conflicts=openpi-piper-policy.service",
    ):
        reject(launcher, old_command, "v3 service commands")
    reject(launcher, "run_greenblock_rlt_online_beta40_v4.sh", "launcher")
    reject(launcher, 'run_rlt_online_session.sh"', "launcher")
    reject(launcher, 'run_rlt_online_update_hook.sh"', "launcher")

    for flag in (
        "--lineage",
        "--state-dir",
        "--latest-episode",
        "--actor-live-max-chunks",
        "--max-episodes",
    ):
        require(current, flag, "current parser")
        require(launcher, flag, "launcher parser")
    require(
        current,
        "greenblock_rlt_gripper_close_v3_from_v2_ep407_20260727",
        "current default lineage",
    )
    require(
        current,
        ".online_rlt_persistent_gripper_v3",
        "current default state",
    )
    for text, label in ((current, "current"), (launcher, "launcher")):
        require(text, '-d "$SCRIPT_DIR/src/openpi"', label)
        require(text, '-d "$SCRIPT_DIR/piper_runtime"', label)
        require(text, 'DEFAULT_WORKSPACE="$SCRIPT_DIR"', label)
        require(text, 'DEFAULT_RUNTIME="$SCRIPT_DIR"', label)
    for text, label in (
        (hook, "hook"),
        (session, "session"),
        (launcher, "launcher"),
    ):
        require(text, "critic_min_advantage_v1", label)
        reject(text, "success-human gripper", label)
        reject(text, "success-human intervention", label)
        reject(text, "failure-human critic", label)

    audit_runtime_projection_chain(launcher)
    audit_ros_runtime_import_preflight(launcher, session)

    print(
        "GRIPPER_CLOSE_V3_STATIC_CHAIN_PASS "
        "raw=v5 execution=v5 governor=v3 "
        f"projection=v3 runtime={DEPLOY_ROOT} "
        "gripper=close_only[5,0.5,0.3,0.5]mm "
        "objective=admitted-human-Q-filter beta_human_gripper_bc=1"
    )


if __name__ == "__main__":
    main()
