# Piper JAX Runtime

## Policy service

```bash
systemctl --user status openpi-piper-policy.service
systemctl --user restart openpi-piper-policy.service
systemctl --user stop openpi-piper-policy.service
tail -f ~/piper_jax_inference_v1/logs/policy_server.log
```

The service binds only to `127.0.0.1:8000`, loads checkpoint `~/openpi_checkpoints/29999`, performs one JAX warm-up inference, and then starts accepting WebSocket requests. User lingering is enabled so the service can start at boot without an interactive login.

## Read-only live inference

```bash
cd ~/piper_jax_inference_v1
PYTHONPATH=. ~/venvs/pika/bin/python -m piper_runtime.live_once \
  --audit reports/live_dry_run_audit.jsonl \
  --report reports/live_dry_run_report.json
```

The command reads:

- D435 `347522072112` as `camera1`;
- D405 `260622272544` as `camera2`;
- Piper joint/gripper feedback from `can0`.

It calls the policy service and runs the response through `DeltaRunner` with `MockEmitter`. It does not initialize, enable, reset, change mode, or send arm/gripper targets. It fails if the `can0` TX packet counter changes during the run.

## Tests

```bash
cd ~/piper_jax_inference_v1
PYTHONPATH=. ~/venvs/pika/bin/pytest -q tests
```

## Before any motion test

Do not replace the mock emitter until joint and gripper units, per-joint limits, maximum delta, velocity, acceleration, watchdog behavior, emergency stop, workspace clearance, and operator authorization have all been reviewed. The first hardware test must use a separately approved low-amplitude plan.
