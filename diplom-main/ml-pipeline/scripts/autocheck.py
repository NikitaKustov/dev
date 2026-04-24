import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = ROOT / "ml-pipeline"


def ok(msg):
    print(f"[OK] {msg}")


def warn(msg):
    print(f"[WARN] {msg}")


def fail(msg):
    print(f"[FAIL] {msg}")


def exists_file(p: Path, required=True):
    if p.exists() and p.is_file():
        ok(f"File exists: {p}")
        return True
    (fail if required else warn)(f"Missing file: {p}")
    return False


def exists_dir(p: Path, required=True):
    if p.exists() and p.is_dir():
        ok(f"Dir exists: {p}")
        return True
    (fail if required else warn)(f"Missing dir: {p}")
    return False


def run_cmd(cmd):
    try:
        cp = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
        return True, cp.stdout.strip() or cp.stderr.strip()
    except Exception as e:
        return False, str(e)


def parse_args():
    p = argparse.ArgumentParser(description="Automatic environment and pipeline checks.")
    p.add_argument("--check-server", action="store_true", help="Run Node syntax check for server/index.js")
    p.add_argument("--check-scripts", action="store_true", help="Run py_compile on ML scripts")
    p.add_argument("--check-data", action="store_true", help="Check merged/raw/processed dataset artifacts")
    return p.parse_args()


def main():
    args = parse_args()
    errors = 0

    ok(f"Python executable: {sys.executable}")
    ok(f"Python version: {sys.version.split()[0]}")

    required_files = [
        ML_ROOT / "requirements.txt",
        ML_ROOT / "configs" / "custom_pose.yaml",
        ML_ROOT / "scripts" / "download_public_datasets.py",
        ML_ROOT / "scripts" / "merge_public_datasets.py",
        ML_ROOT / "scripts" / "prepare_dataset.py",
        ML_ROOT / "scripts" / "augment_occlusions.py",
        ML_ROOT / "scripts" / "train.py",
        ML_ROOT / "scripts" / "evaluate.py",
        ML_ROOT / "scripts" / "export_model.py",
        ML_ROOT / "inference" / "pose_infer.py",
        ROOT / "server" / "index.js",
    ]
    for f in required_files:
        if not exists_file(f, required=True):
            errors += 1

    if args.check_data:
        for d in [
            ML_ROOT / "datasets" / "raw",
            ML_ROOT / "datasets" / "processed",
            ML_ROOT / "datasets" / "public",
        ]:
            if not exists_dir(d, required=False):
                warn(f"Will be created on bootstrap: {d}")

        ann = ML_ROOT / "datasets" / "raw" / "annotations.json"
        if ann.exists():
            try:
                data = json.loads(ann.read_text(encoding="utf-8"))
                ok(f"raw annotations loaded: images={len(data.get('images', []))}, annotations={len(data.get('annotations', []))}")
            except Exception as e:
                fail(f"raw annotations parse error: {e}")
                errors += 1
        else:
            warn("raw annotations not found yet")

    if args.check_scripts:
        py_files = [
            ML_ROOT / "scripts" / "download_public_datasets.py",
            ML_ROOT / "scripts" / "merge_public_datasets.py",
            ML_ROOT / "scripts" / "prepare_dataset.py",
            ML_ROOT / "scripts" / "augment_occlusions.py",
            ML_ROOT / "scripts" / "train.py",
            ML_ROOT / "scripts" / "evaluate.py",
            ML_ROOT / "scripts" / "export_model.py",
            ML_ROOT / "scripts" / "bootstrap_all.py",
            ML_ROOT / "inference" / "pose_infer.py",
        ]
        ok_flag, out = run_cmd([sys.executable, "-m", "py_compile", *[str(p) for p in py_files]])
        if ok_flag:
            ok("Python syntax check passed")
        else:
            fail(f"Python syntax check failed: {out}")
            errors += 1

    if args.check_server:
        ok_flag, out = run_cmd(["node", "--check", str(ROOT / "server" / "index.js")])
        if ok_flag:
            ok("Node syntax check passed")
        else:
            fail(f"Node syntax check failed: {out}")
            errors += 1

    print("\n=== AUTO-CHECK RESULT ===")
    if errors == 0:
        ok("All selected checks passed")
        sys.exit(0)
    fail(f"Checks failed: {errors}")
    sys.exit(1)


if __name__ == "__main__":
    main()
