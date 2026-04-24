import argparse
import contextlib
import io
import json
from pathlib import Path

from ultralytics import YOLO


KEYPOINT_NAMES = [
    # COCO-17 keypoints order used by Ultralytics pose models (e.g. yolo11n-pose)
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def parse_args():
    p = argparse.ArgumentParser(description="Run pose inference for one image.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    # Suppress third-party logs to keep stdout JSON-only for API parser.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = YOLO(args.model, task="pose")
        results = model.predict(
            source=args.image,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
            save=False,
        )

    payload = {"detections": []}
    if not results:
        print(json.dumps(payload, ensure_ascii=False))
        return

    r = results[0]
    boxes = r.boxes
    kpts = r.keypoints
    if boxes is None or kpts is None:
        print(json.dumps(payload, ensure_ascii=False))
        return

    for i in range(len(boxes)):
        box_xyxy = boxes.xyxy[i].tolist()
        score = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
        kp_xy = kpts.xy[i].tolist()
        kp_conf = kpts.conf[i].tolist() if kpts.conf is not None else [1.0] * len(kp_xy)

        keypoints = []
        for idx, (xy, c) in enumerate(zip(kp_xy, kp_conf)):
            name = KEYPOINT_NAMES[idx] if idx < len(KEYPOINT_NAMES) else f"kp_{idx}"
            keypoints.append(
                {
                    "name": name,
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                    "score": float(c),
                }
            )

        payload["detections"].append(
            {
                "score": score,
                "bbox_xyxy": box_xyxy,
                "keypoints": keypoints,
            }
        )

    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
