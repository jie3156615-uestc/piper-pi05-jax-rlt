import importlib.util
from pathlib import Path


def _load_script(path: str):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_train_script_exposes_main():
    module = _load_script("scripts/train_piper_phase_classifier.py")

    assert callable(module.main)


def test_validate_script_exposes_main():
    module = _load_script("scripts/validate_piper_phase_classifier.py")

    assert callable(module.main)


def test_train_script_accepts_report_and_auto_pos_weight():
    module = _load_script("scripts/train_piper_phase_classifier.py")

    args = module.build_parser().parse_args(
        [
            "--dataset",
            "/tmp/dataset",
            "--output",
            "/tmp/model.pt",
            "--report",
            "/tmp/report.json",
            "--pos-weight",
            "auto",
        ]
    )

    assert str(args.report) == "/tmp/report.json"
    assert args.pos_weight == "auto"


def test_validate_script_accepts_timeline_csv():
    module = _load_script("scripts/validate_piper_phase_classifier.py")

    args = module.build_parser().parse_args(
        [
            "--checkpoint",
            "/tmp/model.pt",
            "--heldout-dataset",
            "/tmp/dataset",
            "--report",
            "/tmp/report.json",
            "--timeline-csv",
            "/tmp/timeline.csv",
        ]
    )

    assert str(args.timeline_csv) == "/tmp/timeline.csv"
