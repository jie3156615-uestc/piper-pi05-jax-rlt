"""Explicitly authorized joint-6 low-amplitude round-trip test."""

import argparse
import json
import time
from pathlib import Path


UNITS_PER_DEGREE = 1000.0
OTHER_JOINT_TOLERANCE = 500
TARGET_OVERSHOOT_TOLERANCE = 500


class MotionSafetyError(RuntimeError):
    pass


def compute_joint6_target(origin, degrees=2.0):
    if degrees <= 0 or degrees > 2.0:
        raise MotionSafetyError("joint-6 test must be in (0, 2] degrees")
    target = [int(value) for value in origin]
    target[5] += round(degrees * UNITS_PER_DEGREE)
    return target


def validate_feedback(origin, current, target):
    if any(abs(int(current[index]) - int(origin[index])) > OTHER_JOINT_TOLERANCE for index in range(5)):
        raise MotionSafetyError("unexpected motion on joints 1-5")
    lower = int(origin[5]) - TARGET_OVERSHOOT_TOLERANCE
    upper = int(target[5]) + TARGET_OVERSHOOT_TOLERANCE
    if not lower <= int(current[5]) <= upper:
        raise MotionSafetyError("joint 6 exceeded the authorized range")


def raw_joints(piper):
    state = piper.GetArmJointMsgs().joint_state
    return [int(getattr(state, "joint_%d" % index)) for index in range(1, 7)]


def require_normal(piper):
    status = piper.GetArmStatus()
    if float(status.Hz) <= 0 or int(status.arm_status.err_code) != 0:
        raise MotionSafetyError("arm status is not healthy")


def send_target(piper, target):
    piper.MotionCtrl_2(0x01, 0x01, 10, 0x00)
    piper.JointCtrl(*target)


def drive_until(piper, origin, authorized_target, command_target, tolerance, timeout_s, samples):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        require_normal(piper)
        current = raw_joints(piper)
        validate_feedback(origin, current, authorized_target)
        samples.append({"t": time.time(), "joints_raw": current})
        if all(abs(current[index] - command_target[index]) <= tolerance for index in range(6)):
            return current
        send_target(piper, command_target)
        time.sleep(0.01)
    raise MotionSafetyError("motion did not reach target before timeout")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    from piper_sdk import C_PiperInterface_V2
    from piper_runtime.piper_feedback import PiperFeedbackReader

    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort(False, False, True)
    samples = []
    origin = None
    target = None
    outcome = "failed"
    error = None
    try:
        PiperFeedbackReader(piper).wait_until_healthy(timeout_s=5.0)
        require_normal(piper)
        origin = raw_joints(piper)
        target = compute_joint6_target(origin, 2.0)
        enable_deadline = time.monotonic() + 5.0
        while not piper.EnablePiper():
            if time.monotonic() >= enable_deadline:
                raise MotionSafetyError("arm enable timeout")
            time.sleep(0.01)
        drive_until(piper, origin, target, target, 200, 3.0, samples)
        hold_deadline = time.monotonic() + 0.5
        while time.monotonic() < hold_deadline:
            require_normal(piper)
            current = raw_joints(piper)
            validate_feedback(origin, current, target)
            send_target(piper, target)
            time.sleep(0.01)
        final = drive_until(piper, origin, target, origin, 200, 3.0, samples)
        outcome = "returned_to_origin"
    except Exception as exc:
        error = "%s: %s" % (type(exc).__name__, exc)
        raise
    finally:
        if origin is not None:
            for _ in range(100):
                try:
                    send_target(piper, origin)
                except Exception:
                    break
                time.sleep(0.01)
        final = raw_joints(piper) if origin is not None else None
        piper.DisconnectPort()
        report = {
            "outcome": outcome,
            "error": error,
            "origin_raw": origin,
            "target_raw": target,
            "final_raw": final,
            "joint6_command_degrees": 2.0,
            "speed_percent": 10,
            "sample_count": len(samples),
            "samples": samples,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "samples"}))


if __name__ == "__main__":
    main()
