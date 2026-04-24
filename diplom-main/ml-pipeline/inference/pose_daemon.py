import json
import sys
import traceback
import contextlib
import io
from ultralytics import YOLO


def main():
    """
    Line-based daemon:
    - Reads JSON lines from stdin: {"model": "...", "image": "...", "imgsz": 640, "conf": 0.15, "device":"cpu"}
    - Writes one JSON line to stdout: {"detections":[...]}
    """
    model = None
    model_path = None
    task = "pose"

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            # Always respond with valid JSON to keep server stable.
            sys.stdout.write(json.dumps({"detections": []}) + "\n")
            sys.stdout.flush()
            continue

        mp = req.get("model")
        image = req.get("image")
        imgsz = int(req.get("imgsz") or 640)
        conf = float(req.get("conf") or 0.15)
        device = str(req.get("device") or "cpu")

        try:
            # Keep stdout clean (protocol), but allow stderr for debugging.
            with contextlib.redirect_stdout(io.StringIO()):
                if model is None or mp != model_path:
                    model = YOLO(mp, task=task)
                    model_path = mp

                results = model.predict(
                    source=image,
                    imgsz=imgsz,
                    conf=conf,
                    device=device,
                    verbose=False,
                    save=False,
                )

            payload = {"detections": []}
            if results:
                r = results[0]
                boxes = r.boxes
                kpts = r.keypoints
                if boxes is not None and kpts is not None:
                    # COCO-17
                    names = [
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
                    for i in range(len(boxes)):
                        score = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
                        kp_xy = kpts.xy[i].tolist()
                        kp_conf = (
                            kpts.conf[i].tolist()
                            if kpts.conf is not None
                            else [1.0] * len(kp_xy)
                        )
                        keypoints = []
                        for idx, (xy, c) in enumerate(zip(kp_xy, kp_conf)):
                            name = names[idx] if idx < len(names) else f"kp_{idx}"
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
                                "bbox_xyxy": boxes.xyxy[i].tolist(),
                                "keypoints": keypoints,
                            }
                        )

            sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        except Exception:
            sys.stderr.write(traceback.format_exc() + "\n")
            sys.stderr.flush()
            sys.stdout.write(json.dumps({"detections": []}) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()

