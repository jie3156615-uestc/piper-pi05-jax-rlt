import os
import re
from pathlib import Path


ROOT = Path(os.environ.get("PIPER_INFERENCE_ROOT", "/home/cwzk/piper_jax_inference_v1"))


def _default_value(script_name: str, variable: str) -> str:
    text = (ROOT / script_name).read_text(encoding="utf-8")
    match = re.search(rf'^{variable}="\$\{{{variable}:-([^}}]+)\}}"$', text, re.MULTILINE)
    assert match is not None, f"{script_name} has no default for {variable}"
    return match.group(1)


def test_pure_inference_keeps_soft_reset_default():
    for script_name in ("run_h50_inference.sh", "run_windowed_h10_inference.sh"):
        assert _default_value(script_name, "RESET_SECONDS") == "4"


def test_pure_inference_uses_operator_label_control_instead_of_time_limits():
    for script_name in ("run_h50_inference.sh", "run_windowed_h10_inference.sh"):
        text = (ROOT / script_name).read_text(encoding="utf-8")
        assert "DURATION=" not in text
        assert "MAX_PLANS=" not in text
        assert "--duration" not in text
        assert "--max-plans" not in text
        assert "--operator-label-control" in text


def test_pure_inference_records_binary_episode_labels_and_auto_continues():
    for script_name in ("run_h50_inference.sh", "run_windowed_h10_inference.sh"):
        text = (ROOT / script_name).read_text(encoding="utf-8")
        assert "LABELS_CSV=" in text
        assert "operator_label" in text
        assert "recorded success=" in text
        assert "success? type 1/0" not in text
        assert "Reset the scene, then type n" not in text
