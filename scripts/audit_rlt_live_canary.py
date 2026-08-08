from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def audit_canary_rows(
    rows: list[dict[str, Any]],
    *,
    expected_checkpoint: str | None,
    chunk_length: int = 10,
    max_live_chunks: int = 1,
    max_residual_rad: float = 0.02,
    max_exec_joint_step_deg: float = 0.5,
) -> dict[str, Any]:
    if chunk_length < 1:
        raise ValueError("chunk_length must be positive")
    if max_live_chunks < 1:
        raise ValueError("max_live_chunks must be positive")
    if max_residual_rad < 0 or max_exec_joint_step_deg < 0:
        raise ValueError("audit limits must be non-negative")
    violations: list[str] = []
    rlt_rows = [row for row in rows if row.get("source") == "rlt"]
    max_rlt_rows = int(chunk_length) * int(max_live_chunks)
    if not rlt_rows:
        violations.append("no RLT command was executed; canary is inconclusive")
    if len(rlt_rows) > max_rlt_rows:
        violations.append(f"RLT exposure exceeded {max_rlt_rows} rows: {len(rlt_rows)}")

    offsets: list[int | None] = []
    plan_ids: list[str | None] = []
    residual_max = 0.0
    policy_latencies: list[float] = []
    for row in rlt_rows:
        metadata = row.get("policy_metadata") or {}
        if row.get("gate_active") is not True:
            violations.append(f"t={row.get('t')}: RLT selected outside the frozen phase gate")
        if metadata.get("actor_shadow_ready") is not True:
            violations.append(f"t={row.get('t')}: RLT selected without actor_shadow_ready")
        observation_t = row.get("policy_observation_t", metadata.get("policy_observation_t"))
        enter_t = row.get("gate_enter_t", metadata.get("phase_gate_enter_t"))
        if observation_t is None or enter_t is None or int(observation_t) < int(enter_t):
            violations.append(f"t={row.get('t')}: Actor observation predates phase entry")

        checkpoint = str(
            row.get("behavior_actor_checkpoint")
            or metadata.get("behavior_actor_checkpoint")
            or ""
        )
        if expected_checkpoint and expected_checkpoint not in checkpoint:
            violations.append(f"t={row.get('t')}: unexpected Actor checkpoint {checkpoint!r}")

        plan_id = row.get("policy_plan_id", metadata.get("policy_plan_id"))
        raw_offset = row.get("plan_offset", metadata.get("plan_offset"))
        offsets.append(None if raw_offset is None else int(raw_offset))
        plan_ids.append(None if plan_id is None else str(plan_id))

        a_ref = np.asarray(row.get("a_ref"), dtype=np.float32)
        a_actor = np.asarray(row.get("a_actor"), dtype=np.float32)
        if a_ref.shape != (chunk_length, 7) or a_actor.shape != (chunk_length, 7):
            violations.append(f"t={row.get('t')}: invalid a_ref/a_actor chunk shape")
        else:
            residual = a_actor[0] - a_ref[0]
            residual_max = max(residual_max, float(np.max(np.abs(residual[:6]))))
            if np.any(np.abs(residual[:6]) > max_residual_rad + 1e-6):
                violations.append(f"t={row.get('t')}: Actor joint residual exceeds configured limit")
            if abs(float(residual[6])) > 1e-6:
                violations.append(f"t={row.get('t')}: Actor modified the frozen gripper residual")

        latency = metadata.get("policy_inference_latency_s")
        if latency is not None and math.isfinite(float(latency)):
            policy_latencies.append(float(latency))

    if rlt_rows:
        expected_offsets = list(range(chunk_length))
        if len(rlt_rows) != chunk_length or offsets != expected_offsets:
            violations.append(
                f"canary must execute one complete ordered C={chunk_length} chunk; offsets={offsets}"
            )
        if any(plan_id is None for plan_id in plan_ids) or len(set(plan_ids)) != 1:
            violations.append(f"Actor plan changed inside the canary chunk: {plan_ids}")
        completed = max(
            int((row.get("policy_metadata") or {}).get("actor_live_chunks_completed", 0))
            for row in rlt_rows
        )
        if completed < 1:
            violations.append("runtime did not report a completed Actor canary chunk")

    max_exec_step = 0.0
    cap_rad = math.radians(float(max_exec_joint_step_deg))
    for previous, current in zip(rows, rows[1:]):
        if int(current.get("t", -2)) != int(previous.get("t", -1)) + 1:
            continue
        if "rlt" not in {previous.get("source"), current.get("source")}:
            continue
        previous_exec = np.asarray(previous.get("a_exec"), dtype=np.float32)
        current_exec = np.asarray(current.get("a_exec"), dtype=np.float32)
        if previous_exec.shape != (7,) or current_exec.shape != (7,):
            violations.append("missing finite a_exec around the Actor segment")
            continue
        step = float(np.max(np.abs(current_exec[:6] - previous_exec[:6])))
        max_exec_step = max(max_exec_step, step)
        if step > cap_rad + 1e-5:
            violations.append(
                f"t={current.get('t')}: filtered command step {step:.6f} rad exceeds "
                f"{cap_rad:.6f} rad"
            )

    source_counts = dict(collections.Counter(str(row.get("source")) for row in rows))
    safety_counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        safety_counts.update(str(reason) for reason in (row.get("policy_metadata") or {}).get("safety_reasons", []))

    return {
        "format": "piper_rlt_live_canary_audit_v1",
        "passed": not violations,
        "violations": violations,
        "rows": len(rows),
        "rlt_rows": len(rlt_rows),
        "source_counts": source_counts,
        "actor_plan_ids": sorted({value for value in plan_ids if value is not None}),
        "actor_plan_offsets": offsets,
        "actor_residual_abs_max_rad": residual_max,
        "filtered_exec_joint_step_abs_max_rad": max_exec_step,
        "policy_inference_latency_s_max": max(policy_latencies, default=None),
        "safety_reason_counts": dict(safety_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed audit of a one-chunk Piper RLT live canary")
    parser.add_argument("--episode-jsonl", type=Path, required=True)
    parser.add_argument("--expected-checkpoint")
    parser.add_argument("--chunk-length", type=int, default=10)
    parser.add_argument("--max-live-chunks", type=int, default=1)
    parser.add_argument("--max-residual-rad", type=float, default=0.02)
    parser.add_argument("--max-exec-joint-step-deg", type=float, default=0.5)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.episode_jsonl.expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = audit_canary_rows(
        rows,
        expected_checkpoint=args.expected_checkpoint,
        chunk_length=args.chunk_length,
        max_live_chunks=args.max_live_chunks,
        max_residual_rad=args.max_residual_rad,
        max_exec_joint_step_deg=args.max_exec_joint_step_deg,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        output_path = args.output_json.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
