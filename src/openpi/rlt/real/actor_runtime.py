"""Small inference-only adapter for the Piper Actor shadow service."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np

from openpi.rlt.real.agent_jax import RealRLTLearner
from openpi.rlt.real.config import LEGACY_ACTOR_EXECUTION_PROFILE
from openpi.rlt.real.config import GRIPPER_RESIDUAL_CLOSE_ASSIST
from openpi.rlt.real.config import PERSISTENT_ACTION_SCHEMA_FINGERPRINT
from openpi.rlt.real.config import PERSISTENT_ACTOR_EXECUTION_PROFILE
from openpi.rlt.real.config import PERSISTENT_EXECUTION_FILTER_PROFILE
from openpi.rlt.real.config import PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
from openpi.rlt.real.config import PERSISTENT_GRIPPER_CLOSE_GOVERNOR_PROFILE
from openpi.rlt.real.config import PERSISTENT_GOVERNOR_PROFILE
from openpi.rlt.real.config import RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT
from openpi.rlt.real.config import RANK1_GRIPPER_CLOSE_PROJECTION_PROFILE
from openpi.rlt.real.config import RANK1_ACTOR_OUTPUT_SCHEMA_FINGERPRINT


ACTION_SCHEMA_FINGERPRINT = RANK1_ACTOR_OUTPUT_SCHEMA_FINGERPRINT
ACTOR_PROJECTION_PROFILE = (
    "rank1_bump_v1_r005_d1_0015_d2_001_cone15_"
    "scale33_min020_gripper_frozen"
)
BASE_CHECKPOINT_FINGERPRINT = (
    "full20k_step20000_metadata_sha256_"
    "14d9cac129ec7ce91f2e5aab3f5bfac06172c8fb70709f01850fb8e8215870e5"
)
RL_TOKEN_FINGERPRINT = "2f2e1e6bbcae8f08217ec7ba0b88088bfa44627be495035319deb84e06052b49"
PHASE_CLASSIFIER_FINGERPRINT = "8c5b443edd3f399529680ef5e4c5dffcdee4af2ae2f224f6da4152237c9a50dc"
FRESH_ZERO_ACTOR_INITIALIZATION = "fresh_zero_random_v1"
FRESH_ZERO_LINEAGE_MANIFEST_FORMAT = (
    "openpi_piper_gripper_v3_fresh_zero_lineage_v1"
)
EXPECTED_CHECKPOINT_FINGERPRINTS = {
    "base_checkpoint": BASE_CHECKPOINT_FINGERPRINT,
    "rl_token": RL_TOKEN_FINGERPRINT,
    "phase_classifier": PHASE_CLASSIFIER_FINGERPRINT,
    "action_schema": ACTION_SCHEMA_FINGERPRINT,
}
EXPECTED_COMMON_CHECKPOINT_FINGERPRINTS = {
    key: value
    for key, value in EXPECTED_CHECKPOINT_FINGERPRINTS.items()
    if key != "action_schema"
}


def _deployment_float(name: str, checkpoint_value: float) -> float:
    """Read an optional inference-only projection envelope override."""

    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(checkpoint_value)
    value = float(raw)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive; got {raw!r}")
    return value


class JaxCheckpointShadowActor:
    """Expose a saved :class:`RealRLTLearner` through the shadow protocol.

    Inputs and output use the real-RLT learning coordinate contract:
    ``joint delta[0:6] + absolute gripper[6]`` with shape ``(10, 7)``.
    """

    def __init__(self, checkpoint: str | Path) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.learner = RealRLTLearner.load_checkpoint(
            self.checkpoint,
            expected_fingerprints=EXPECTED_COMMON_CHECKPOINT_FINGERPRINTS,
        )
        if self.learner.config.chunk_length != 10 or self.learner.config.action_dim != 7:
            raise ValueError("Piper Actor shadow requires a C=10, action_dim=7 learner checkpoint")
        cfg = self.learner.config
        checkpoint_schema = self.learner.fingerprints.get("action_schema")
        if checkpoint_schema == ACTION_SCHEMA_FINGERPRINT:
            if cfg.actor_execution_profile != LEGACY_ACTOR_EXECUTION_PROFILE:
                raise ValueError(
                    "v3 Actor checkpoint schema requires the legacy rank1 execution profile"
                )
            checkpoint_kind = "legacy_v3_rank1_source"
        elif checkpoint_schema == PERSISTENT_ACTION_SCHEMA_FINGERPRINT:
            if cfg.actor_execution_profile != PERSISTENT_ACTOR_EXECUTION_PROFILE:
                raise ValueError(
                    "v4 persistent checkpoint schema requires "
                    f"{PERSISTENT_ACTOR_EXECUTION_PROFILE!r}"
                )
            if (
                self.learner.fingerprints.get("execution_filter_profile")
                != PERSISTENT_EXECUTION_FILTER_PROFILE
            ):
                raise ValueError("persistent checkpoint execution filter fingerprint mismatch")
            if self.learner.fingerprints.get("actor_governor") != PERSISTENT_GOVERNOR_PROFILE:
                raise ValueError("persistent checkpoint Actor governor fingerprint mismatch")
            if (
                self.learner.fingerprints.get("warm_start_actor_source_schema")
                != ACTION_SCHEMA_FINGERPRINT
            ):
                raise ValueError(
                    "persistent checkpoint must record its v3 Actor-head source schema"
                )
            checkpoint_kind = "persistent_v4_checkpoint_with_v3_rank1_output"
        elif checkpoint_schema == PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT:
            if cfg.actor_execution_profile != PERSISTENT_ACTOR_EXECUTION_PROFILE:
                raise ValueError(
                    "gripper-close checkpoint requires persistent C10 execution"
                )
            if cfg.gripper_residual_mode != GRIPPER_RESIDUAL_CLOSE_ASSIST:
                raise ValueError(
                    "gripper-close checkpoint mode/fingerprint mismatch"
                )
            if cfg.freeze_gripper_residual:
                raise ValueError(
                    "gripper-close checkpoint cannot freeze the gripper residual"
                )
            if (
                self.learner.fingerprints.get("execution_filter_profile")
                != PERSISTENT_EXECUTION_FILTER_PROFILE
            ):
                raise ValueError(
                    "gripper-close checkpoint execution filter fingerprint mismatch"
                )
            if (
                self.learner.fingerprints.get("actor_governor")
                != PERSISTENT_GRIPPER_CLOSE_GOVERNOR_PROFILE
            ):
                raise ValueError(
                    "gripper-close checkpoint governor fingerprint mismatch"
                )
            warm_start_source = self.learner.fingerprints.get(
                "warm_start_actor_source_schema"
            )
            fresh_zero_initialization = self.learner.fingerprints.get(
                "actor_initialization"
            )
            claims_warm_start = warm_start_source == ACTION_SCHEMA_FINGERPRINT
            claims_fresh_zero = (
                fresh_zero_initialization == FRESH_ZERO_ACTOR_INITIALIZATION
            )
            if claims_warm_start and claims_fresh_zero:
                raise ValueError(
                    "gripper-close checkpoint has conflicting warm-start and "
                    "fresh-zero Actor provenance"
                )
            if claims_warm_start:
                checkpoint_kind = (
                    "persistent_v5_gripper_close_with_rank1_output"
                )
            elif claims_fresh_zero:
                warm_start_fields = sorted(
                    key
                    for key, value in self.learner.fingerprints.items()
                    if key.startswith("warm_start_") and value not in (None, "")
                )
                if warm_start_fields:
                    raise ValueError(
                        "fresh-zero gripper-close checkpoint contains forbidden "
                        f"warm-start provenance: {warm_start_fields}"
                    )
                manifest_format = self.learner.fingerprints.get(
                    "fresh_zero_lineage_manifest_format"
                )
                manifest_sha256 = self.learner.fingerprints.get(
                    "fresh_zero_lineage_manifest_sha256", ""
                )
                if (
                    manifest_format != FRESH_ZERO_LINEAGE_MANIFEST_FORMAT
                    or len(manifest_sha256) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in manifest_sha256
                    )
                ):
                    raise ValueError(
                        "fresh-zero gripper-close checkpoint has invalid lineage "
                        "manifest provenance"
                    )
                if self.learner.update_step <= cfg.actor_start_step:
                    raise ValueError(
                        "fresh-zero gripper-close checkpoint has no audited Actor "
                        "updates after Critic burn-in"
                    )
                checkpoint_kind = (
                    "persistent_v5_gripper_close_fresh_zero_rank1_output"
                )
            else:
                raise ValueError(
                    "gripper-close checkpoint must record either its frozen v3 "
                    "Actor source or explicit fresh-zero initialization provenance"
                )
        else:
            raise ValueError(
                "unsupported Actor checkpoint action schema fingerprint: "
                f"{checkpoint_schema!r}"
            )
        # The checkpoint provenance can be v3 or v4, but the unchanged Actor
        # head always responds in the v3 rank1 protocol consumed by the online
        # governor.  Never label the policy response with the replay schema.
        self.checkpoint_action_schema_fingerprint = str(checkpoint_schema)
        self.output_action_schema_fingerprint = (
            RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT
            if checkpoint_schema
            == PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
            else ACTION_SCHEMA_FINGERPRINT
        )
        self.output_actor_projection_profile = (
            RANK1_GRIPPER_CLOSE_PROJECTION_PROFILE
            if checkpoint_schema
            == PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
            else ACTOR_PROJECTION_PROFILE
        )
        self.checkpoint_kind = checkpoint_kind
        if cfg.actor_residual_parameterization != "rank1_bump":
            raise ValueError(
                "legacy full-chunk Actor checkpoints are read-only and cannot be loaded by the real-robot service"
            )
        if (
            checkpoint_schema != PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
            and not cfg.freeze_gripper_residual
        ):
            raise ValueError(
                "frozen real-robot Actor checkpoint must freeze the gripper residual"
            )
        expected_limits = {
            "actor_residual_max_rad": 0.005,
            "actor_residual_d1_max_rad": 0.0015,
            "actor_residual_d2_max_rad": 0.001,
            "actor_direction_cone_deg": 15.0,
        }
        mismatches = {
            name: (float(expected), float(getattr(cfg, name)))
            for name, expected in expected_limits.items()
            if not np.isclose(float(getattr(cfg, name)), float(expected), rtol=0.0, atol=1e-12)
        }
        if mismatches:
            raise ValueError(f"real-robot Actor projection contract mismatch: {mismatches}")
        if checkpoint_schema == PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT:
            expected_gripper_limits = {
                "actor_gripper_residual_max_close_m": 0.005,
                "actor_gripper_residual_d1_max_m": 0.0005,
                "actor_gripper_residual_d2_max_m": 0.0003,
                "actor_gripper_max_boundary_jump_m": 0.0005,
                "gripper_command_min_m": 0.0,
                "gripper_command_max_m": 0.08,
            }
            gripper_mismatches = {
                name: (float(expected), float(getattr(cfg, name)))
                for name, expected in expected_gripper_limits.items()
                if not np.isclose(
                    float(getattr(cfg, name)),
                    float(expected),
                    rtol=0.0,
                    atol=1e-12,
                )
            }
            if gripper_mismatches:
                raise ValueError(
                    "gripper-close Actor contract mismatch: "
                    f"{gripper_mismatches}"
                )

        # Preserve the checkpoint fingerprint as training provenance.  This
        # separately logged deployment envelope rescales the learned Actor
        # direction without rewriting historical checkpoint metadata.
        deployment_limits = {
            "actor_residual_max_rad": _deployment_float(
                "PIPER_RLT_DEPLOY_RESIDUAL_MAX_RAD", cfg.actor_residual_max_rad
            ),
            "actor_residual_d1_max_rad": _deployment_float(
                "PIPER_RLT_DEPLOY_RESIDUAL_D1_MAX_RAD", cfg.actor_residual_d1_max_rad
            ),
            "actor_residual_d2_max_rad": _deployment_float(
                "PIPER_RLT_DEPLOY_RESIDUAL_D2_MAX_RAD", cfg.actor_residual_d2_max_rad
            ),
        }
        self.checkpoint_projection_limits = {
            name: float(getattr(cfg, name)) for name in deployment_limits
        }
        self.deployment_projection_limits = deployment_limits
        self.learner.config = dataclasses.replace(cfg, **deployment_limits)

    def provenance_metadata(self) -> dict[str, str]:
        return {
            "actor_checkpoint_action_schema_fingerprint": (
                self.checkpoint_action_schema_fingerprint
            ),
            "actor_output_protocol_action_schema_fingerprint": (
                self.output_action_schema_fingerprint
            ),
            "actor_output_projection_profile": (
                self.output_actor_projection_profile
            ),
            "actor_checkpoint_kind": self.checkpoint_kind,
        }

    def predict(self, *, z_rl: np.ndarray, state: np.ndarray, a_ref: np.ndarray) -> np.ndarray:
        z_rl = np.asarray(z_rl, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        a_ref = np.asarray(a_ref, dtype=np.float32)
        expected_z = self.learner.normalization.z_rl.mean.shape
        if z_rl.shape != expected_z:
            raise ValueError(f"z_rl must have shape {expected_z}, got {z_rl.shape}")
        if state.shape != (self.learner.config.state_dim,):
            raise ValueError(f"state must have shape ({self.learner.config.state_dim},), got {state.shape}")
        expected_ref = (self.learner.config.chunk_length, self.learner.config.action_dim)
        if a_ref.shape != expected_ref:
            raise ValueError(f"a_ref must have shape {expected_ref}, got {a_ref.shape}")
        if not all(np.all(np.isfinite(value)) for value in (z_rl, state, a_ref)):
            raise ValueError("Actor shadow inputs must be finite")
        action = self.learner.act(z_rl, state, a_ref, use_target=False)[0]
        if action.shape != expected_ref or not np.all(np.isfinite(action)):
            raise RuntimeError(f"Actor checkpoint returned an invalid action chunk: {action.shape}")
        return action.astype(np.float32, copy=False)


def create_shadow_actor(checkpoint: Path) -> JaxCheckpointShadowActor:
    """Factory consumed by ``piper_runtime.rlt_shadow_policy_service``."""

    return JaxCheckpointShadowActor(checkpoint)
