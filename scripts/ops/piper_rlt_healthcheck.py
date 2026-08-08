#!/usr/bin/env python3
"""Read-only operational health check for the Piper pi0.5/RLT host."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    summary: str
    details: Dict[str, Any]


def _run(command: List[str], *, timeout: float = 8.0) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, capture_output=True, check=False, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(command, 127, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(command, 124, exc.stdout or "", exc.stderr or "timed out")


def _check_port(port: int) -> Check:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            pass
    except OSError as exc:
        return Check(f"port_{port}", "fail", f"127.0.0.1:{port} is not accepting connections", {"error": str(exc)})
    return Check(f"port_{port}", "ok", f"127.0.0.1:{port} is accepting connections", {})


def _check_can() -> Check:
    result = _run(["ip", "-details", "link", "show", "can0"])
    output = (result.stdout + result.stderr).strip()
    up = bool(re.search(r"<[^>]*\bUP\b", output))
    bitrate = bool(re.search(r"\bbitrate\s+1000000\b", output))
    status = "ok" if result.returncode == 0 and up and bitrate else "fail"
    return Check(
        "can0",
        status,
        "UP at 1,000,000 bit/s" if status == "ok" else "can0 is missing, down, or has the wrong bitrate",
        {"returncode": result.returncode, "up": up, "bitrate_1000000": bitrate, "output": output[-800:]},
    )


def _check_realsense(minimum: int) -> Check:
    result = _run(["rs-enumerate-devices", "-s"], timeout=12.0)
    output = (result.stdout + result.stderr).strip()
    serials = sorted(set(re.findall(r"\b\d{8,}\b", output)))
    expected = sorted(
        filter(None, os.environ.get("RLT_EXPECTED_REALSENSE_SERIALS", "").replace(",", " ").split())
    )
    missing = sorted(set(expected).difference(serials))
    ok = result.returncode == 0 and len(serials) >= minimum and not missing
    return Check(
        "realsense",
        "ok" if ok else "fail",
        f"detected {len(serials)} RealSense device(s)" if ok else f"expected at least {minimum}; detected {len(serials)}",
        {
            "serials": serials,
            "expected_serials": expected,
            "missing_expected_serials": missing,
            "returncode": result.returncode,
            "output": output[-800:],
        },
    )


def _check_sense() -> Check:
    configured = os.environ.get(
        "RLT_SENSE_BY_PATH", "/dev/serial/by-path/pci-0000:00:14.0-usb-0:1.4:1.0-port0"
    )
    path = Path(configured)
    if not path.exists():
        return Check("sense", "fail", f"Sense path does not exist: {path}", {})
    resolved = path.resolve()
    return Check("sense", "ok", f"{path} -> {resolved}", {"configured": str(path), "resolved": str(resolved)})


def _check_services(names: List[str]) -> List[Check]:
    checks = []
    for name in names:
        result = _run(["systemctl", "--user", "is-active", name])
        active = result.returncode == 0 and result.stdout.strip() == "active"
        checks.append(
            Check(
                f"service:{name}",
                "ok" if active else "fail",
                result.stdout.strip() or result.stderr.strip() or "inactive",
                {"returncode": result.returncode},
            )
        )
    return checks


def _latest_json(root: Path, pattern: str) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    if not root.exists():
        return None, None
    candidates = [path for path in root.glob(pattern) if path.is_file()]
    if not candidates:
        return None, None
    path = max(candidates, key=lambda item: item.stat().st_mtime_ns)
    try:
        return path, json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return path, None


def _find_key(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find_key(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_key(child, key)
            if found is not None:
                return found
    return None


def _check_learning_state(session_root: Path) -> List[Check]:
    checks: List[Check] = []
    selector_env = os.environ.get("PIPER_RLT_SELECTED_ACTOR_FILE")
    selector_candidates = ([Path(selector_env).expanduser()] if selector_env else []) + list(
        session_root.glob(".online_rlt*/selected_actor_checkpoint.txt")
    )
    selector = next((path for path in selector_candidates if path.is_file()), None)
    if selector is None:
        checks.append(Check("selected_actor", "warn", "selected Actor file was not found", {}))
    else:
        selected = selector.read_text(encoding="utf-8").strip()
        match = re.search(r"step_(\d+)", selected)
        checks.append(
            Check(
                "selected_actor",
                "ok",
                selected or "NONE",
                {"path": str(selector), "step": None if match is None else int(match.group(1))},
            )
        )

    reports: List[Tuple[Path, Dict[str, Any]]] = []
    if session_root.exists():
        for path in session_root.glob("episode_*/report.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if value.get("outcome") == "episode_done" and value.get("terminal_reward") in {0, 0.0, 1, 1.0}:
                reports.append((path, value))
    reports.sort(key=lambda item: item[0].name)
    rewards = [float(value["terminal_reward"]) for _, value in reports]
    recent = rewards[-20:]
    summary = (
        f"{sum(rewards):.0f}/{len(rewards)} total successes; {sum(recent):.0f}/{len(recent)} recent"
        if rewards
        else "no completed rewarded episodes found"
    )
    checks.append(
        Check(
            "episode_metrics",
            "ok" if rewards else "warn",
            summary,
            {
                "episodes": len(rewards),
                "successes": int(sum(rewards)),
                "success_rate": None if not rewards else sum(rewards) / len(rewards),
                "recent_window": len(recent),
                "recent_success_rate": None if not recent else sum(recent) / len(recent),
            },
        )
    )

    validation_path, validation = _latest_json(session_root, ".online_rlt*/learner/validation_*.json")
    if validation_path is None:
        checks.append(Check("learner_validation", "warn", "no learner validation report found", {}))
    elif validation is None:
        checks.append(Check("learner_validation", "fail", f"invalid JSON: {validation_path}", {}))
    else:
        keys = (
            "validation_td_error_abs_mean",
            "actor_q_advantage_abs_p95",
            "reward1_reward0_exec_q_gap",
            "success_failure_exec_q_gap",
        )
        metrics = {key: _find_key(validation, key) for key in keys}
        checks.append(
            Check(
                "learner_validation",
                "ok",
                f"latest validation: {validation_path.name}",
                {"path": str(validation_path), **metrics},
            )
        )

    training_path, training = _latest_json(session_root, ".online_rlt*/learner/training_summary.json")
    if training_path is None:
        checks.append(Check("learner_training", "warn", "no learner training summary found", {}))
    elif training is None:
        checks.append(Check("learner_training", "fail", f"invalid JSON: {training_path}", {}))
    else:
        final_metrics = training.get("final_metrics", {})
        metric_keys = (
            "critic_loss",
            "critic_q1_loss",
            "critic_q2_loss",
            "q1_mean",
            "q2_mean",
            "critic_reward1_q_mean",
            "critic_reward0_q_mean",
            "critic_reward1_reward0_q_gap",
            "td_error_abs",
            "actor_loss",
        )
        metrics = {key: final_metrics.get(key) for key in metric_keys}
        checks.append(
            Check(
                "learner_training",
                "ok",
                f"step {training.get('final_step')} ({training.get('additional_steps')} new update(s))",
                {"path": str(training_path), **metrics},
            )
        )
    return checks


def _check_gpu() -> Check:
    result = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    output = result.stdout.strip()
    if result.returncode != 0 or not output:
        return Check("gpu", "fail", result.stderr.strip() or "nvidia-smi returned no GPU", {})
    fields = [item.strip() for item in output.splitlines()[0].split(",")]
    details = {
        "name": fields[0],
        "memory_total_mib": int(fields[1]),
        "memory_used_mib": int(fields[2]),
        "memory_free_mib": int(fields[3]),
        "temperature_c": int(fields[4]),
    }
    status = "warn" if details["memory_free_mib"] < 4096 or details["temperature_c"] >= 80 else "ok"
    return Check("gpu", status, f"{details['memory_free_mib']} MiB free, {details['temperature_c']} C", details)


def _check_disk(path: Path, *, min_free_gib: float, min_free_percent: float) -> Check:
    usage = shutil.disk_usage(path)
    free_gib = usage.free / 1024**3
    free_percent = 100.0 * usage.free / usage.total
    status = "ok" if free_gib >= min_free_gib and free_percent >= min_free_percent else "fail"
    return Check(
        "disk",
        status,
        f"{free_gib:.1f} GiB free ({free_percent:.1f}%) at {path}",
        {"path": str(path), "free_gib": free_gib, "free_percent": free_percent},
    )


def collect(
    profile: str, session_root: Path, *, min_disk_gib: float, min_disk_percent: float
) -> Dict[str, Any]:
    disk_path = session_root if session_root.exists() else Path.home()
    checks = [
        _check_disk(disk_path, min_free_gib=min_disk_gib, min_free_percent=min_disk_percent),
        _check_gpu(),
    ]
    checks.extend(_check_learning_state(session_root))
    if profile == "rlt":
        checks.extend([_check_can(), _check_realsense(2), _check_sense(), _check_port(11311), _check_port(8001)])
        services = os.environ.get(
            "RLT_HEALTH_SERVICES",
            "rlt-roscore.service,rlt-piper-controller.service,rlt-takeover-sources.service,"
            "rlt-teleop-controller.service,openpi-rlt-shadow-policy-gripper-v3.service",
        )
        checks.extend(_check_services([item.strip() for item in services.split(",") if item.strip()]))
    elif profile == "inference":
        checks.extend([_check_can(), _check_realsense(2), _check_port(8000)])
    counts = {status: sum(check.status == status for check in checks) for status in ("ok", "warn", "fail")}
    return {
        "format": "piper_rlt_health_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "profile": profile,
        "overall": "fail" if counts["fail"] else "warn" if counts["warn"] else "ok",
        "counts": counts,
        "checks": [asdict(check) for check in checks],
    }


def _atomic_json(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("offline", "inference", "rlt"), default="rlt")
    parser.add_argument(
        "--session-root",
        type=Path,
        default=Path(
            os.environ.get("RLT_SESSION_ROOT", "~/rlt_online_sessions/greenblock_rlt_fresh_v3_20260729")
        ).expanduser(),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--min-disk-gib", type=float, default=20.0)
    parser.add_argument("--min-disk-percent", type=float, default=10.0)
    args = parser.parse_args()
    report = collect(
        args.profile,
        args.session_root.expanduser(),
        min_disk_gib=args.min_disk_gib,
        min_disk_percent=args.min_disk_percent,
    )
    if args.output:
        _atomic_json(args.output.expanduser(), report)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"Piper RLT health: {report['overall'].upper()} ({report['counts']})")
        for check in report["checks"]:
            print(f"[{check['status'].upper():4}] {check['name']}: {check['summary']}")
    raise SystemExit(2 if report["overall"] == "fail" else 1 if report["overall"] == "warn" else 0)


if __name__ == "__main__":
    main()
