"""
========================================================
  YOLOv8 / YOLOv11 Object Detection Pipeline
  Author : Senior ML Engineer – Computer Vision
  Libs   : ultralytics, opencv-python, tqdm
========================================================

pip install requirements:
    pip install ultralytics opencv-python tqdm

Usage:
    python yolo_detection.py --input input_video.mp4 \
                             --model yolov8n.pt \
                             --output output_video.mp4 \
                             --conf 0.25
"""

import argparse
import json
import time
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO


# ──────────────────────────────────────────────
# 1.  Argument Parsing
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the detection pipeline."""
    parser = argparse.ArgumentParser(
        description="YOLO Object Detection – video → annotated video + JSON"
    )
    parser.add_argument(
        "--input",  default="input_video.mp4",
        help="Path to the input video file (default: input_video.mp4)"
    )
    parser.add_argument(
        "--model", default="yolov8n.pt",
        help="YOLO model weights (e.g. yolov8n.pt, yolov8s.pt, yolo11n.pt)"
    )
    parser.add_argument(
        "--output", default="output_video.mp4",
        help="Path for the annotated output video (default: output_video.mp4)"
    )
    parser.add_argument(
        "--json_out", default="detection_results.json",
        help="Path for the metadata JSON file (default: detection_results.json)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Minimum confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--device", default="",
        help="Inference device: '' = auto, 'cpu', '0' = GPU 0 (default: auto)"
    )
    return parser.parse_args()


# ──────────────────────────────────────────────
# 2.  Video I/O Helpers
# ──────────────────────────────────────────────
def open_video(path: str) -> tuple[cv2.VideoCapture, dict]:
    """
    Open a video file and return the capture object plus its properties.

    Returns
    -------
    cap   : cv2.VideoCapture
    props : dict with keys fps, width, height, total_frames
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    props = {
        "fps":          cap.get(cv2.CAP_PROP_FPS),
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    return cap, props


def create_writer(output_path: str, props: dict) -> cv2.VideoWriter:
    """
    Create a VideoWriter that matches the source video resolution and FPS.
    Uses the MP4V codec (works on all platforms; swap for 'avc1' if needed).
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        props["fps"],
        (props["width"], props["height"]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create VideoWriter at: {output_path}")
    return writer


# ──────────────────────────────────────────────
# 3.  Drawing Utilities
# ──────────────────────────────────────────────
# Colour palette – one colour per class index (BGR)
_PALETTE = [
    (56,  56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72),   (23,  204,  146), (134, 219, 61),
    (52, 147, 26),  (187, 212, 0),   (168, 153, 44),  (255, 194, 0),
    (255, 111, 31), (0,  69, 255),   (188,  0, 255),  (255, 180, 0),
]

def _color_for(class_id: int) -> tuple[int, int, int]:
    """Deterministic colour from class index."""
    return _PALETTE[class_id % len(_PALETTE)]


def draw_detections(frame: "cv2.Mat", detections: list[dict]) -> "cv2.Mat":
    """
    Overlay bounding boxes and labels on *frame* in-place and return it.

    Each detection dict must contain:
        class_id, class_name, confidence, bbox [x1,y1,x2,y2]
    """
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        color            = _color_for(det["class_id"])
        label            = f"{det['class_name']} {det['confidence']:.2f}"

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

        # Label background
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        top_left  = (x1, max(y1 - th - baseline - 4, 0))
        bot_right = (x1 + tw + 2, max(y1, th + baseline + 4))
        cv2.rectangle(frame, top_left, bot_right, color, cv2.FILLED)

        # Label text
        cv2.putText(
            frame, label,
            (x1 + 2, max(y1 - baseline - 2, th + 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), thickness=1,
            lineType=cv2.LINE_AA,
        )

    return frame


# ──────────────────────────────────────────────
# 4.  Metadata Extraction
# ──────────────────────────────────────────────
def extract_detections(result, names: dict, conf_thresh: float) -> list[dict]:
    """
    Convert a single YOLO result object into a list of detection dicts.

    Parameters
    ----------
    result      : ultralytics Results object for one frame
    names       : {class_id: class_name} mapping from the model
    conf_thresh : minimum confidence to keep

    Returns
    -------
    list of dicts, each with keys:
        class_id, class_name, confidence, bbox [x1,y1,x2,y2]
    """
    detections = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < conf_thresh:
            continue

        cls_id   = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class_id":   cls_id,
            "class_name": names.get(cls_id, str(cls_id)),
            "confidence": round(conf, 4),
            "bbox":       [round(x1, 1), round(y1, 1),
                           round(x2, 1), round(y2, 1)],
        })

    return detections


# ──────────────────────────────────────────────
# 5.  Main Pipeline
# ──────────────────────────────────────────────
def run_detection(args: argparse.Namespace) -> None:
    """
    Full detection pipeline:
      1. Load model
      2. Open video
      3. Process frame-by-frame
      4. Write annotated video
      5. Save metadata JSON
    """
    # ── Load model ──────────────────────────────
    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)
    model.to(args.device if args.device else "cpu")   # explicit device
    names: dict = model.names                         # {0: 'person', ...}

    # ── Open source video ───────────────────────
    print(f"[INFO] Opening video: {args.input}")
    cap, props = open_video(args.input)
    fps = props["fps"]

    print(
        f"[INFO] Video: {props['total_frames']} frames | "
        f"{props['width']}×{props['height']} | {fps:.2f} FPS"
    )

    # ── Create output video writer ───────────────
    writer = create_writer(args.output, props)
    print(f"[INFO] Output video → {args.output}")

    # ── Metadata container ───────────────────────
    all_frames_metadata: list[dict] = []
    pipeline_start = time.perf_counter()

    # ── Frame loop ───────────────────────────────
    pbar = tqdm(total=props["total_frames"], unit="frame", desc="Detecting")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break                          # end of video or read error

        # Timestamp from actual FPS (seconds, 3 d.p.)
        timestamp_sec = round(frame_idx / fps, 3)

        # ── Run YOLO inference ──────────────────
        # verbose=False  ⟶ suppress per-frame console spam
        results = model(frame, conf=args.conf, verbose=False)

        # results is a list (one element per image in the batch)
        result = results[0]

        # ── Extract structured detections ───────
        detections = extract_detections(result, names, conf_thresh=args.conf)

        # ── Build frame metadata ─────────────────
        frame_meta = {
            "frame_index": frame_idx,
            "timestamp_sec": timestamp_sec,
            "num_detections": len(detections),
            "detections": detections,
        }
        all_frames_metadata.append(frame_meta)

        # ── Annotate frame ───────────────────────
        annotated = draw_detections(frame.copy(), detections)

        # Optional: stamp frame index + timestamp on the frame
        cv2.putText(
            annotated,
            f"Frame {frame_idx:05d} | {timestamp_sec:.3f}s",
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (200, 200, 200), 1, cv2.LINE_AA,
        )

        writer.write(annotated)

        frame_idx += 1
        pbar.update(1)

    pbar.close()

    # ── Release resources ────────────────────────
    cap.release()
    writer.release()

    elapsed = time.perf_counter() - pipeline_start
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0
    print(
        f"\n[INFO] Processed {frame_idx} frames in {elapsed:.1f}s "
        f"(avg {avg_fps:.1f} FPS)"
    )

    # ── Save metadata JSON ───────────────────────
    output_json = {
        "source_video":   str(Path(args.input).resolve()),
        "model":          args.model,
        "conf_threshold": args.conf,
        "video_fps":      fps,
        "total_frames":   frame_idx,
        "class_names":    names,
        "frames":         all_frames_metadata,
    }

    json_path = Path(args.json_out)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(output_json, fh, indent=2)

    print(f"[INFO] Metadata saved → {json_path}")
    print("[DONE] Pipeline complete.")


# ──────────────────────────────────────────────
# 6.  Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    run_detection(args)
