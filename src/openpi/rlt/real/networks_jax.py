from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.rlt.real.config import GRIPPER_RESIDUAL_CLOSE_ASSIST
from openpi.rlt.real.config import RealRLTConfig

_BINOMIAL5 = (1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0)
_RANK1_BUMP_C10 = (0.0, 0.2, 0.5, 0.8, 1.0, 1.0, 0.8, 0.5, 0.2, 0.0)
_DIRECTION_SCALE_COUNT = 33
_REFERENCE_MOTION_EPS_RAD = 1e-3
_NUMERIC_TOLERANCE = 1e-9
_PERSISTENT_C10_BLEND = (
    0.0,
    0.011532794797541024,
    0.07641111619163743,
    0.2098765432098765,
    0.3966874968246709,
    0.603312503175329,
    0.790123456790123,
    0.9235888838083626,
    0.9884672052024577,
    1.0,
)


def straight_through_positive(value: jnp.ndarray) -> jnp.ndarray:
    """Return ``max(value, 0)`` while keeping an identity backward gradient.

    The old frozen Actor starts with an exactly-zero gripper logit.  A regular
    ReLU-like projection can strand that logit on the negative side, where the
    success-only gripper imitation loss can no longer recover it.  This keeps
    the physical forward value strictly one-sided while allowing that loss to
    move a migrated zero/negative logit back toward a closing command.
    """

    projected = jnp.maximum(value, 0.0)
    return value + jax.lax.stop_gradient(projected - value)


def rank1_bump_window(*, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    """Return the fixed non-negative C10 residual envelope.

    Its endpoint values are exactly zero.  Its largest first and second
    differences are 0.3 and 0.2, so a 0.005 rad direction hard-bounds residual
    d1/d2 at 0.0015/0.001 rad respectively.
    """

    return jnp.asarray(_RANK1_BUMP_C10, dtype=dtype)


def rank1_direction_limit(residual_limit: jnp.ndarray, cfg: RealRLTConfig) -> jnp.ndarray:
    """Collapse per-step replay limits to one safe direction limit per joint."""

    residual_limit = jnp.asarray(residual_limit)
    expected = (len(_RANK1_BUMP_C10), cfg.action_dim)
    if residual_limit.shape != expected:
        raise ValueError(f"residual_limit must have shape {expected}, got {residual_limit.shape}")
    if cfg.action_dim != 7:
        raise ValueError(f"rank1_bump requires Piper action_dim=7, got {cfg.action_dim}")

    window = rank1_bump_window(dtype=residual_limit.dtype)
    # At a zero endpoint the residual is identically zero, so that row imposes
    # no direction constraint.  Every non-zero row is respected exactly.
    per_step_direction_limit = jnp.where(
        window[:, None] > 0.0,
        residual_limit / jnp.maximum(window[:, None], jnp.finfo(residual_limit.dtype).tiny),
        jnp.inf,
    )
    replay_limit = jnp.min(per_step_direction_limit, axis=0)
    max_d1_coefficient = jnp.max(jnp.abs(jnp.diff(window)))
    max_d2_coefficient = jnp.max(jnp.abs(jnp.diff(window, n=2)))
    configured_joint_limit = jnp.minimum(
        jnp.asarray(cfg.actor_residual_max_rad, dtype=residual_limit.dtype),
        jnp.minimum(
            jnp.asarray(cfg.actor_residual_d1_max_rad, dtype=residual_limit.dtype) / max_d1_coefficient,
            jnp.asarray(cfg.actor_residual_d2_max_rad, dtype=residual_limit.dtype) / max_d2_coefficient,
        ),
    )
    direction_limit = jnp.minimum(replay_limit, configured_joint_limit)
    if cfg.freeze_gripper_residual:
        # Frozen-v2 compatibility: Pi0.5 remains the sole gripper owner.
        return direction_limit.at[-1].set(0.0)
    # Dim 6 is metres, not radians.  It therefore gets an independent limit
    # and never inherits the arm-joint d1/d2 cap above.
    gripper_limit = jnp.minimum(
        replay_limit[-1],
        jnp.asarray(
            cfg.actor_gripper_residual_max_close_m,
            dtype=residual_limit.dtype,
        ),
    )
    return direction_limit.at[-1].set(gripper_limit)


def rank1_bump_residual(direction: jnp.ndarray) -> jnp.ndarray:
    """Expand one per-sample action direction into the fixed rank-one C10 chunk."""

    if direction.ndim != 2:
        raise ValueError(f"direction must have shape (batch, action), got {direction.shape}")
    window = rank1_bump_window(dtype=direction.dtype)
    return window[None, :, None] * direction[:, None, :]


def rank1_direction_from_residual(residual: jnp.ndarray) -> jnp.ndarray:
    """Least-squares direction for a C10 residual (exact for rank1_bump chunks)."""

    if residual.ndim != 3 or residual.shape[1] != len(_RANK1_BUMP_C10):
        raise ValueError(f"residual must have shape (batch, 10, action), got {residual.shape}")
    window = rank1_bump_window(dtype=residual.dtype)
    return jnp.sum(residual * window[None, :, None], axis=1) / jnp.sum(jnp.square(window))


def persistent_c10_blend_window(*, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    """Runtime-identical quintic samples for one physical C10 Actor plan."""

    return jnp.asarray(_PERSISTENT_C10_BLEND, dtype=dtype)


def persistent_governed_residual(
    a_ref: jnp.ndarray,
    canonical_direction: jnp.ndarray,
    carry_in: jnp.ndarray,
    previous_carry: jnp.ndarray,
    boundary_anchor: jnp.ndarray,
    *,
    residual_max_rad: float | jnp.ndarray,
    d1_max_rad: float | jnp.ndarray,
    d2_max_rad: float | jnp.ndarray,
    direction_cone_deg: float | jnp.ndarray,
    boundary_jump_max_rad: float | jnp.ndarray,
    direction_static_threshold_rad: float | jnp.ndarray,
    projection_scale_steps: int = _DIRECTION_SCALE_COUNT,
    gripper_residual_mode: str = "frozen",
    gripper_residual_max_close_m: float | jnp.ndarray = 0.005,
    gripper_d1_max_m: float | jnp.ndarray = 0.0005,
    gripper_d2_max_m: float | jnp.ndarray = 0.0003,
    gripper_boundary_jump_max_m: float | jnp.ndarray = 0.0005,
    gripper_command_min_m: float | jnp.ndarray = 0.0,
    gripper_command_max_m: float | jnp.ndarray = 0.08,
    gripper_release_reference_m: float | jnp.ndarray = 0.05,
    gripper_release_delta_m: float | jnp.ndarray = 0.002,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Project a raw rank1 direction exactly like the online governor.

    The selected scale is discrete, but the returned residual is reconstructed
    from the selected scale and the original direction.  Gradients therefore
    flow through the direction inside a selected projection bin, matching the
    runtime/training contract.
    """

    if a_ref.ndim != 3 or a_ref.shape[1:] != (len(_PERSISTENT_C10_BLEND), 7):
        raise ValueError(f"a_ref must have shape (batch, 10, 7), got {a_ref.shape}")
    batch_size = a_ref.shape[0]
    expected_vector = (batch_size, 7)
    for name, value in (
        ("canonical_direction", canonical_direction),
        ("carry_in", carry_in),
        ("previous_carry", previous_carry),
        ("boundary_anchor", boundary_anchor),
    ):
        if value.shape != expected_vector:
            raise ValueError(f"{name} must have shape {expected_vector}, got {value.shape}")
    if projection_scale_steps < 2:
        raise ValueError("projection_scale_steps must be at least 2")

    dtype = a_ref.dtype
    blend = persistent_c10_blend_window(dtype=dtype)
    scales = jnp.linspace(0.0, 1.0, projection_scale_steps, dtype=dtype)
    carry_joint = carry_in[..., :6]
    direction_joint = canonical_direction[..., :6]
    target = (
        carry_joint[:, None, :]
        + scales[None, :, None] * (direction_joint - carry_joint)[:, None, :]
    )
    residual_joint = (
        carry_joint[:, None, None, :]
        + blend[None, None, :, None] * (target - carry_joint[:, None, :])[:, :, None, :]
    )

    previous_joint = previous_carry[..., :6]
    history = jnp.concatenate(
        [
            jnp.broadcast_to(previous_joint[:, None, None, :], (batch_size, projection_scale_steps, 1, 6)),
            jnp.broadcast_to(carry_joint[:, None, None, :], (batch_size, projection_scale_steps, 1, 6)),
            residual_joint,
            residual_joint[:, :, -1:, :],
        ],
        axis=2,
    )
    residual_d1 = jnp.diff(history, axis=2)
    residual_d2 = jnp.diff(residual_d1, axis=2)
    residual_abs_max = jnp.max(
        jnp.abs(
            jnp.concatenate(
                [
                    jnp.broadcast_to(carry_joint[:, None, None, :], (batch_size, projection_scale_steps, 1, 6)),
                    residual_joint,
                ],
                axis=2,
            )
        ),
        axis=(2, 3),
    )
    d1_abs_max = jnp.max(jnp.abs(residual_d1), axis=(2, 3))
    d2_abs_max = jnp.max(jnp.abs(residual_d2), axis=(2, 3))

    safe_joint = a_ref[:, None, :, :6] + residual_joint
    reference_boundary_anchor = boundary_anchor[:, :6] - carry_joint
    ref_steps = jnp.diff(
        jnp.concatenate([reference_boundary_anchor[:, None, :], a_ref[..., :6]], axis=1),
        axis=1,
    )
    actor_steps = jnp.diff(
        jnp.concatenate(
            [
                jnp.broadcast_to(
                    boundary_anchor[:, None, None, :6],
                    (batch_size, projection_scale_steps, 1, 6),
                ),
                safe_joint,
            ],
            axis=2,
        ),
        axis=2,
    )
    boundary_jump = jnp.max(jnp.abs(actor_steps[:, :, 0, :]), axis=-1)
    ref_norm = jnp.linalg.norm(ref_steps, axis=-1)
    actor_norm = jnp.linalg.norm(actor_steps, axis=-1)
    dot = jnp.sum(ref_steps[:, None, :, :] * actor_steps, axis=-1)
    tolerance = jnp.asarray(_NUMERIC_TOLERANCE, dtype=dtype)
    cone_cosine = jnp.cos(jnp.deg2rad(jnp.asarray(direction_cone_deg, dtype=dtype)))
    moving = ref_norm >= jnp.asarray(direction_static_threshold_rad, dtype=dtype)
    moving_ok = (dot >= -tolerance) & (
        dot + tolerance >= cone_cosine * ref_norm[:, None, :] * actor_norm
    )
    static_ok = (
        actor_norm
        <= jnp.asarray(direction_static_threshold_rad, dtype=dtype) + tolerance
    )
    direction_ok = jnp.all(jnp.where(moving[:, None, :], moving_ok, static_ok), axis=-1)
    valid = (
        (residual_abs_max <= jnp.asarray(residual_max_rad, dtype=dtype) + tolerance)
        & (d1_abs_max <= jnp.asarray(d1_max_rad, dtype=dtype) + tolerance)
        & (d2_abs_max <= jnp.asarray(d2_max_rad, dtype=dtype) + tolerance)
        & (boundary_jump <= jnp.asarray(boundary_jump_max_rad, dtype=dtype) + tolerance)
        & direction_ok
    )
    selected_scale = jnp.max(
        jnp.where(valid, scales[None, :], -jnp.ones_like(scales)[None, :]),
        axis=1,
    )
    approved = selected_scale >= 0.0
    selected_scale = jnp.maximum(selected_scale, 0.0)
    selected_target = carry_joint + selected_scale[:, None] * (
        direction_joint - carry_joint
    )
    selected_joint_residual = carry_joint[:, None, :] + blend[None, :, None] * (
        selected_target - carry_joint
    )[:, None, :]
    selected_joint_residual = jnp.where(
        approved[:, None, None],
        selected_joint_residual,
        jnp.zeros_like(selected_joint_residual),
    )
    if gripper_residual_mode == GRIPPER_RESIDUAL_CLOSE_ASSIST:
        carry_gripper = carry_in[..., 6]
        previous_gripper = previous_carry[..., 6]
        reference_gripper = a_ref[..., 6]
        # ``boundary_anchor`` is the actually published action.  Removing the
        # committed carry recovers the counterfactual Pi0.5/base anchor.
        base_boundary_gripper = boundary_anchor[..., 6] - carry_gripper
        opening_delta = reference_gripper[:, -1] - base_boundary_gripper
        release_intent = (
            jnp.max(reference_gripper, axis=1)
            >= jnp.asarray(gripper_release_reference_m, dtype=dtype)
        ) | (
            opening_delta
            >= jnp.asarray(gripper_release_delta_m, dtype=dtype)
        )
        raw_gripper_target = jnp.clip(
            canonical_direction[..., 6],
            -jnp.asarray(gripper_residual_max_close_m, dtype=dtype),
            0.0,
        )
        desired_gripper = jnp.where(release_intent, 0.0, raw_gripper_target)
        gripper_target_candidates = (
            carry_gripper[:, None]
            + scales[None, :] * (desired_gripper - carry_gripper)[:, None]
        )
        gripper_candidates = (
            carry_gripper[:, None, None]
            + blend[None, None, :]
            * (gripper_target_candidates - carry_gripper[:, None])[:, :, None]
        )
        gripper_history = jnp.concatenate(
            [
                jnp.broadcast_to(
                    previous_gripper[:, None, None],
                    (batch_size, projection_scale_steps, 1),
                ),
                jnp.broadcast_to(
                    carry_gripper[:, None, None],
                    (batch_size, projection_scale_steps, 1),
                ),
                gripper_candidates,
                gripper_candidates[:, :, -1:],
            ],
            axis=2,
        )
        gripper_d1 = jnp.diff(gripper_history, axis=2)
        gripper_d2 = jnp.diff(gripper_d1, axis=2)
        gripper_boundary_jump = jnp.abs(
            gripper_candidates[:, :, 0] - carry_gripper[:, None]
        )
        gripper_command = reference_gripper[:, None, :] + gripper_candidates
        gripper_valid = (
            (
                jnp.max(jnp.abs(gripper_candidates), axis=2)
                <= jnp.asarray(gripper_residual_max_close_m, dtype=dtype)
                + tolerance
            )
            & (
                jnp.max(jnp.abs(gripper_d1), axis=2)
                <= jnp.asarray(gripper_d1_max_m, dtype=dtype) + tolerance
            )
            & (
                jnp.max(jnp.abs(gripper_d2), axis=2)
                <= jnp.asarray(gripper_d2_max_m, dtype=dtype) + tolerance
            )
            & (
                gripper_boundary_jump
                <= jnp.asarray(gripper_boundary_jump_max_m, dtype=dtype)
                + tolerance
            )
            & (
                jnp.min(gripper_command, axis=2)
                >= jnp.asarray(gripper_command_min_m, dtype=dtype) - tolerance
            )
            & (
                jnp.max(gripper_command, axis=2)
                <= jnp.asarray(gripper_command_max_m, dtype=dtype) + tolerance
            )
        )
        gripper_scale = jnp.max(
            jnp.where(
                gripper_valid,
                scales[None, :],
                -jnp.ones_like(scales)[None, :],
            ),
            axis=1,
        )
        gripper_scale = jnp.maximum(gripper_scale, 0.0)
        selected_gripper_target = (
            carry_gripper
            + gripper_scale * (desired_gripper - carry_gripper)
        )
        selected_gripper_residual = (
            carry_gripper[:, None]
            + blend[None, :]
            * (selected_gripper_target - carry_gripper)[:, None]
        )
    else:
        selected_gripper_residual = jnp.zeros(
            (batch_size, len(_PERSISTENT_C10_BLEND)),
            dtype=dtype,
        )
    selected_residual = jnp.concatenate(
        [selected_joint_residual, selected_gripper_residual[..., None]],
        axis=-1,
    )
    return selected_residual, selected_scale, approved


def persistent_filter_residual(
    planned_residual: jnp.ndarray,
    carry_in: jnp.ndarray,
    filter_alpha: jnp.ndarray,
    *,
    freeze_gripper_residual: bool = True,
) -> jnp.ndarray:
    """Apply the runtime's stateful residual low-pass over one C10 plan."""

    if planned_residual.ndim != 3 or planned_residual.shape[1:] != (10, 7):
        raise ValueError(
            f"planned_residual must have shape (batch, 10, 7), got {planned_residual.shape}"
        )
    batch_size = planned_residual.shape[0]
    if carry_in.shape != (batch_size, 7):
        raise ValueError(f"carry_in must have shape {(batch_size, 7)}, got {carry_in.shape}")
    if filter_alpha.shape != (batch_size, 10):
        raise ValueError(
            f"filter_alpha must have shape {(batch_size, 10)}, got {filter_alpha.shape}"
        )

    def step(previous: jnp.ndarray, inputs: tuple[jnp.ndarray, jnp.ndarray]):
        planned, alpha = inputs
        current = (1.0 - alpha[:, None]) * previous + alpha[:, None] * planned
        if freeze_gripper_residual:
            current = current.at[:, 6].set(0.0)
        else:
            # Piper's native smoothing filters only joints[:6].  The gripper
            # path is already rate-limited by the persistent quintic knot.
            current = current.at[:, 6].set(planned[:, 6])
        return current, current

    _, filtered_time_major = jax.lax.scan(
        step,
        carry_in,
        (
            jnp.swapaxes(planned_residual, 0, 1),
            jnp.swapaxes(filter_alpha, 0, 1),
        ),
    )
    return jnp.swapaxes(filtered_time_major, 0, 1)


def persistent_filtered_candidate_action(
    a_base_filtered: jnp.ndarray,
    a_ref: jnp.ndarray,
    canonical_direction: jnp.ndarray,
    carry_in: jnp.ndarray,
    previous_carry: jnp.ndarray,
    boundary_anchor: jnp.ndarray,
    filter_alpha: jnp.ndarray,
    cfg: RealRLTConfig,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Map a raw Actor knot to the exact counterfactual executed C10 action."""

    if a_base_filtered.shape != a_ref.shape:
        raise ValueError(
            "a_base_filtered and a_ref must share shape, got "
            f"{a_base_filtered.shape} and {a_ref.shape}"
        )
    planned, projection_scale, approved = persistent_governed_residual(
        a_ref,
        canonical_direction,
        carry_in,
        previous_carry,
        boundary_anchor,
        residual_max_rad=cfg.actor_residual_max_rad,
        d1_max_rad=cfg.actor_residual_d1_max_rad,
        d2_max_rad=cfg.actor_residual_d2_max_rad,
        direction_cone_deg=cfg.actor_direction_cone_deg,
        boundary_jump_max_rad=cfg.actor_max_boundary_jump_rad,
        direction_static_threshold_rad=cfg.actor_direction_static_threshold_rad,
        projection_scale_steps=cfg.actor_projection_scale_steps,
        gripper_residual_mode=cfg.gripper_residual_mode,
        gripper_residual_max_close_m=cfg.actor_gripper_residual_max_close_m,
        gripper_d1_max_m=cfg.actor_gripper_residual_d1_max_m,
        gripper_d2_max_m=cfg.actor_gripper_residual_d2_max_m,
        gripper_boundary_jump_max_m=cfg.actor_gripper_max_boundary_jump_m,
        gripper_command_min_m=cfg.gripper_command_min_m,
        gripper_command_max_m=cfg.gripper_command_max_m,
        gripper_release_reference_m=cfg.gripper_release_reference_m,
        gripper_release_delta_m=cfg.gripper_release_delta_m,
    )
    filtered = persistent_filter_residual(
        planned,
        carry_in,
        filter_alpha,
        freeze_gripper_residual=cfg.freeze_gripper_residual,
    )
    return a_base_filtered + filtered, {
        "planned_residual": planned,
        "filtered_residual": filtered,
        "projection_scale": projection_scale,
        "projection_approved": approved,
    }


def apply_direction_cone_scale(
    a_ref: jnp.ndarray,
    residual: jnp.ndarray,
    *,
    cone_deg: float | jnp.ndarray,
    motion_epsilon_rad: float = _REFERENCE_MOTION_EPS_RAD,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Scale a whole rank-one residual into the reference-velocity cone.

    ``a_ref`` stores joint deltas from the state at the beginning of the chunk.
    Prepending zero therefore recovers all ten reference step velocities.  One
    of 33 scales in [0, 1] is selected, and the largest scale satisfying every
    step is used.  A single scale preserves the rank1_bump direction contract.
    The selection is piecewise differentiable: within a selected scale the
    actor still receives gradients through its direction.
    """

    if a_ref.ndim != 3 or residual.ndim != 3 or a_ref.shape != residual.shape:
        raise ValueError(f"a_ref and residual must share shape (batch, chunk, action), got {a_ref.shape} and {residual.shape}")
    if a_ref.shape[1:] != (len(_RANK1_BUMP_C10), 7):
        raise ValueError(f"direction cone requires shape (batch, 10, 7), got {a_ref.shape}")
    if motion_epsilon_rad <= 0.0:
        raise ValueError("motion_epsilon_rad must be positive")

    ref_joint = a_ref[..., :6]
    residual_joint = residual[..., :6]
    origin = jnp.zeros_like(ref_joint[:, :1, :])
    ref_velocity = jnp.diff(jnp.concatenate([origin, ref_joint], axis=1), axis=1)

    scales = jnp.linspace(0.0, 1.0, _DIRECTION_SCALE_COUNT, dtype=residual.dtype)
    candidate_joint = ref_joint[:, None, :, :] + scales[None, :, None, None] * residual_joint[:, None, :, :]
    candidate_origin = jnp.zeros_like(candidate_joint[:, :, :1, :])
    candidate_velocity = jnp.diff(jnp.concatenate([candidate_origin, candidate_joint], axis=2), axis=2)

    ref_norm = jnp.linalg.norm(ref_velocity, axis=-1)
    candidate_norm = jnp.linalg.norm(candidate_velocity, axis=-1)
    dot = jnp.sum(ref_velocity[:, None, :, :] * candidate_velocity, axis=-1)
    cosine_threshold = jnp.cos(jnp.deg2rad(jnp.asarray(cone_deg, dtype=residual.dtype)))
    tolerance = jnp.asarray(1e-12, dtype=residual.dtype)
    moving_ok = (dot >= -tolerance) & (
        dot + tolerance >= cosine_threshold * ref_norm[:, None, :] * candidate_norm
    )
    static_ok = candidate_norm <= jnp.asarray(motion_epsilon_rad, dtype=residual.dtype) + tolerance
    moving = ref_norm >= jnp.asarray(motion_epsilon_rad, dtype=residual.dtype)
    step_ok = jnp.where(moving[:, None, :], moving_ok, static_ok)
    scale_ok = jnp.all(step_ok, axis=-1)
    selected_scale = jnp.max(jnp.where(scale_ok, scales[None, :], -jnp.ones_like(scales)[None, :]), axis=-1)
    selected_scale = jnp.maximum(selected_scale, 0.0)
    scaled = residual.at[..., :6].set(
        residual[..., :6] * selected_scale[:, None, None]
    )
    return scaled, selected_scale


def residual_temporal_metrics(residual: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Hard-max diagnostics for the six joint residual trajectories."""

    if residual.ndim != 3 or residual.shape[1] != len(_RANK1_BUMP_C10) or residual.shape[2] < 6:
        raise ValueError(f"residual must have shape (batch, 10, action>=6), got {residual.shape}")
    joint_residual = residual[..., :6]
    d1 = jnp.diff(joint_residual, axis=1)
    d2 = jnp.diff(joint_residual, n=2, axis=1)
    direction = rank1_direction_from_residual(residual)[..., :6]
    reconstruction = rank1_bump_residual(
        jnp.pad(direction, ((0, 0), (0, residual.shape[-1] - 6)))
    )[..., :6]
    return {
        "residual_d1_abs_mean": jnp.mean(jnp.abs(d1)),
        "residual_d1_abs_max": jnp.max(jnp.abs(d1)),
        "residual_d2_abs_mean": jnp.mean(jnp.abs(d2)),
        "residual_d2_abs_max": jnp.max(jnp.abs(d2)),
        "residual_endpoint_abs_max": jnp.max(jnp.abs(joint_residual[:, (0, -1), :])),
        "residual_rank1_error_abs_max": jnp.max(jnp.abs(joint_residual - reconstruction)),
    }


def smooth_residual_chunk(residual: jnp.ndarray) -> jnp.ndarray:
    """Apply a fixed differentiable low-pass to the residual chunk.

    Replicated edges preserve constant corrections.  The SFT reference itself
    is deliberately left untouched, so the actor still predicts a residual
    around the original reference action.
    """

    if residual.ndim != 3:
        raise ValueError(f"residual must have shape (batch, chunk, action), got {residual.shape}")
    padded = jnp.pad(residual, ((0, 0), (2, 2), (0, 0)), mode="edge")
    chunk_length = residual.shape[1]
    return sum(weight * padded[:, offset : offset + chunk_length, :] for offset, weight in enumerate(_BINOMIAL5))


class ModalityProjector(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.output_dim)(x)
        x = nn.LayerNorm()(x)
        return nn.tanh(x)


class ResidualActor(nn.Module):
    cfg: RealRLTConfig

    @nn.compact
    def __call__(
        self,
        z_rl: jnp.ndarray,
        robot_state: jnp.ndarray,
        ref_input: jnp.ndarray,
        a_ref: jnp.ndarray,
        actor_limit: jnp.ndarray,
    ) -> jnp.ndarray:
        z_h = ModalityProjector(self.cfg.projection_dim)(z_rl)
        state_h = ModalityProjector(self.cfg.projection_dim)(robot_state)
        ref_h = ModalityProjector(self.cfg.projection_dim)(ref_input)
        h = jnp.concatenate([z_h, state_h, ref_h], axis=-1)
        h = nn.Dense(self.cfg.hidden_dim)(h)
        h = nn.relu(h)
        h = nn.Dense(self.cfg.hidden_dim)(h)
        h = nn.relu(h)
        if self.cfg.actor_residual_parameterization == "legacy_full_chunk":
            delta = nn.Dense(
                self.cfg.chunk_length * self.cfg.action_dim,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
            )(h)
            delta = delta.reshape((-1, self.cfg.chunk_length, self.cfg.action_dim))
            residual_limit = actor_limit.reshape((1, self.cfg.chunk_length, self.cfg.action_dim))
            residual = smooth_residual_chunk(jnp.tanh(delta) * residual_limit)
        else:
            direction_logits = nn.Dense(
                self.cfg.action_dim,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
            )(h)
            direction_limit = actor_limit.reshape((1, self.cfg.action_dim))
            direction = jnp.tanh(direction_logits) * direction_limit
            if self.cfg.gripper_residual_mode == GRIPPER_RESIDUAL_CLOSE_ASSIST:
                # Old frozen checkpoints have an exactly-zero seventh logit,
                # so migration begins at exact Pi0.5 pass-through.  This
                # one-sided map can only make the absolute gripper target
                # smaller (tighter), never open it.
                close_fraction = straight_through_positive(
                    jnp.tanh(direction_logits[..., -1])
                )
                direction = direction.at[..., -1].set(
                    -close_fraction * direction_limit[..., -1]
                )
            residual = rank1_bump_residual(direction)
            residual, _ = apply_direction_cone_scale(
                a_ref.reshape((-1, self.cfg.chunk_length, self.cfg.action_dim)),
                residual,
                cone_deg=self.cfg.actor_direction_cone_deg,
            )
        return a_ref.reshape((-1, self.cfg.chunk_length, self.cfg.action_dim)) + residual


class QHead(nn.Module):
    cfg: RealRLTConfig

    @nn.compact
    def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(self.cfg.hidden_dim)(h)
        h = nn.relu(h)
        h = nn.Dense(self.cfg.hidden_dim)(h)
        h = nn.relu(h)
        q = nn.Dense(1)(h)
        return jnp.squeeze(q, axis=-1)


class TwinCritic(nn.Module):
    cfg: RealRLTConfig

    @nn.compact
    def __call__(
        self,
        z_rl: jnp.ndarray,
        robot_state: jnp.ndarray,
        a_ref: jnp.ndarray,
        candidate_action: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        z_h = ModalityProjector(self.cfg.projection_dim)(z_rl)
        state_h = ModalityProjector(self.cfg.projection_dim)(robot_state)
        ref_h = ModalityProjector(self.cfg.projection_dim)(a_ref)
        action_h = ModalityProjector(self.cfg.projection_dim)(candidate_action)
        h = jnp.concatenate([z_h, state_h, ref_h, action_h], axis=-1)
        return QHead(self.cfg)(h), QHead(self.cfg)(h)
