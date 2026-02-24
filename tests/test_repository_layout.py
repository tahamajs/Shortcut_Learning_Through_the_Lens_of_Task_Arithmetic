from pathlib import Path


def test_expected_core_files_exist():
    root = Path(__file__).resolve().parents[1]
    expected = [
        root / "shortcut_patcher" / "src" / "train.py",
        root / "shortcut_patcher" / "src" / "task_vector.py",
        root / "shortcut_patcher" / "src" / "edit_model.py",
        root / "shortcut_patcher" / "src" / "analyze.py",
        root / "shortcut_patcher" / "src" / "visualize.py",
        root / "shortcut_patcher" / "run_pipeline.sh",
    ]
    for path in expected:
        assert path.exists(), f"Missing expected file: {path}"
