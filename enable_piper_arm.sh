#!/usr/bin/env bash
set -eo pipefail

source "$HOME/pika_ros/install/setup.bash"
set -u

echo "Calling /enable_srv enable_request=true ..."
enable_response="$(rosservice call /enable_srv "enable_request: true")"
printf '%s\n' "$enable_response"

# rosservice exits successfully even when the service response is False.
# /arm_status err_code=0 also does not expose per-joint driver enable, so the
# service response is the authoritative preflight gate.
if ! grep -qE '^enable_response: True$' <<<"$enable_response"; then
  echo "Piper /enable_srv did not confirm that all joint drivers are enabled." >&2
  exit 4
fi

# The Piper controller can briefly publish a non-zero status while enable is
# settling.  Do not launch policy execution until ROS reports a stable healthy
# state; runtime faults after this gate are still handled immediately.
stable_samples=0
deadline=$((SECONDS + 12))
last_status=""
while (( SECONDS < deadline )); do
  last_status="$(timeout 2 rostopic echo -n 1 /arm_status 2>/dev/null || true)"
  if grep -qE '^err_code: 0$' <<<"$last_status"; then
    stable_samples=$((stable_samples + 1))
    if (( stable_samples >= 5 )); then
      echo "Piper enable preflight: err_code=0 stable for 5 samples"
      exit 0
    fi
  else
    stable_samples=0
  fi
  sleep 0.2
done

echo "Piper did not reach a stable healthy state after enable." >&2
printf '%s\n' "$last_status" >&2
exit 4
