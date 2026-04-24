import argparse
import ssl
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


COCO_FILES = [
    ("annotations_trainval2017.zip", "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"),
    ("annotations_trainval2017.zip", "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"),
    ("train2017.zip", "https://images.cocodataset.org/zips/train2017.zip"),
    ("train2017.zip", "http://images.cocodataset.org/zips/train2017.zip"),
    ("val2017.zip", "https://images.cocodataset.org/zips/val2017.zip"),
    ("val2017.zip", "http://images.cocodataset.org/zips/val2017.zip"),
]

# Lightweight public dataset for quick bootstrap.
LITE_FILES = [
    ("coco8-pose.zip", "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-pose.zip"),
]

# Community mirror URL can change; script will continue if this source fails.
CROWDPOSE_FILES = [
    ("crowdpose.zip", "https://github.com/Jeff-sjtu/CrowdPose/releases/download/v0.1/crowdpose.zip"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Auto-download public pose datasets.")
    p.add_argument("--out-dir", type=str, default="ml-pipeline/datasets/public")
    p.add_argument("--skip-coco", action="store_true")
    p.add_argument("--try-crowdpose", action="store_true")
    p.add_argument("--lite", action="store_true", help="Download small dataset instead of full COCO")
    return p.parse_args()


def download_file(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url}")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=180) as resp, dst.open("wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except URLError as e:
        # Fallback for environments with broken TLS inspection/cert chain.
        if "CERTIFICATE_VERIFY_FAILED" not in str(e):
            raise
        print("TLS verify failed, retrying with unverified SSL context...")
        insecure_ctx = ssl._create_unverified_context()
        with urlopen(req, timeout=180, context=insecure_ctx) as resp, dst.open("wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)


def is_valid_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    if not zipfile.is_zipfile(path):
        return False
    return True


def prepare_valid_archive(urls, dst: Path) -> Path:
    if is_valid_zip(dst):
        return dst

    alt = dst.with_name(dst.stem + ".retry" + dst.suffix)

    if dst.exists():
        try:
            dst.unlink()
        except PermissionError:
            pass

    target = dst if not dst.exists() else alt
    if target.exists():
        try:
            target.unlink()
        except PermissionError:
            target = alt.with_name(alt.stem + ".2" + alt.suffix)

    last_error = None
    for url in urls:
        try:
            download_file(url, target)
            if is_valid_zip(target):
                return target
        except Exception as e:
            last_error = e
    if last_error:
        raise RuntimeError(f"All download mirrors failed for {dst.name}: {last_error}")
    if not is_valid_zip(target):
        raise RuntimeError(f"Downloaded file is not a valid zip: {target}")
    return target


def extract_archive(archive_path: Path, out_dir: Path):
    print(f"Extracting: {archive_path.name}")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(out_dir)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    archives_dir = out_dir / "_archives"
    archives_dir.mkdir(parents=True, exist_ok=True)

    if args.lite:
        lite_dir = out_dir / "lite"
        lite_dir.mkdir(parents=True, exist_ok=True)
        grouped = {}
        for fname, url in LITE_FILES:
            grouped.setdefault(fname, []).append(url)
        for fname, urls in grouped.items():
            dst = archives_dir / fname
            archive = prepare_valid_archive(urls, dst)
            extract_archive(archive, lite_dir)
        print("Lite dataset downloaded.")
        print(f"Public datasets root: {out_dir.resolve()}")
        return

    if not args.skip_coco:
        coco_dir = out_dir / "coco2017"
        coco_dir.mkdir(parents=True, exist_ok=True)
        grouped = {}
        for fname, url in COCO_FILES:
            grouped.setdefault(fname, []).append(url)
        for fname, urls in grouped.items():
            dst = archives_dir / fname
            archive = prepare_valid_archive(urls, dst)
            extract_archive(archive, coco_dir)

    if args.try_crowdpose:
        crowd_dir = out_dir / "crowdpose"
        crowd_dir.mkdir(parents=True, exist_ok=True)
        grouped = {}
        for fname, url in CROWDPOSE_FILES:
            grouped.setdefault(fname, []).append(url)
        for fname, urls in grouped.items():
            dst = archives_dir / fname
            try:
                archive = prepare_valid_archive(urls, dst)
                extract_archive(archive, crowd_dir)
            except Exception as e:
                print(f"CrowdPose download skipped: {e}")

    print("Done.")
    print(f"Public datasets root: {out_dir.resolve()}")
    print("Next: run merge_public_datasets.py")


if __name__ == "__main__":
    main()
