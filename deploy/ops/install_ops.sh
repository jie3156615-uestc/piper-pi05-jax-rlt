#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPER_RLT_REPO:-$HOME/piper_jax_inference_v1}"
USER_UNITS="$HOME/.config/systemd/user"
OPS_CONFIG="$HOME/.config/piper-rlt"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ENABLE_HEALTH_TIMER=0

if [[ "${1:-}" == "--enable-health-timer" ]]; then
  ENABLE_HEALTH_TIMER=1
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--enable-health-timer]" >&2
  exit 2
fi

command -v logrotate >/dev/null 2>&1 || {
  echo "logrotate is required: sudo apt-get install logrotate" >&2
  exit 1
}
mkdir -p "$USER_UNITS" "$OPS_CONFIG" "$HOME/.local/state/piper-rlt" "$HOME/.local/state/piper-rlt/backups/$STAMP"

for name in piper-rlt-healthcheck.service piper-rlt-healthcheck.timer piper-rlt-logrotate.service piper-rlt-logrotate.timer; do
  target="$USER_UNITS/$name"
  if [[ -e "$target" ]]; then
    cp -a "$target" "$HOME/.local/state/piper-rlt/backups/$STAMP/$name"
  fi
  install -m 0644 "$ROOT/deploy/systemd/portable/$name" "$target"
done

if [[ -e "$OPS_CONFIG/logrotate.conf" ]]; then
  cp -a "$OPS_CONFIG/logrotate.conf" "$HOME/.local/state/piper-rlt/backups/$STAMP/logrotate.conf"
fi
sed "s|@HOME@|$HOME|g" "$ROOT/deploy/logrotate/piper-rlt.conf.in" > "$OPS_CONFIG/logrotate.conf"
chmod 0644 "$OPS_CONFIG/logrotate.conf"

systemd-analyze --user verify \
  "$USER_UNITS/piper-rlt-healthcheck.service" \
  "$USER_UNITS/piper-rlt-healthcheck.timer" \
  "$USER_UNITS/piper-rlt-logrotate.service" \
  "$USER_UNITS/piper-rlt-logrotate.timer"
systemctl --user daemon-reload
systemctl --user enable --now piper-rlt-logrotate.timer
if [[ "$ENABLE_HEALTH_TIMER" == 1 ]]; then
  systemctl --user enable --now piper-rlt-healthcheck.timer
else
  systemctl --user disable --now piper-rlt-healthcheck.timer >/dev/null 2>&1 || true
fi

systemctl --user start piper-rlt-logrotate.service
echo "Installed Piper RLT operations support. Backup: $HOME/.local/state/piper-rlt/backups/$STAMP"
echo "Run: bash $ROOT/scripts/ops/run_ops_acceptance.sh rlt"
