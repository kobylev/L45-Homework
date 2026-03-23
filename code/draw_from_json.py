"""
================================================================
  draw_from_json.py
  Reads detection_results.json + original video,
  redraws bounding boxes frame-by-frame, and saves the
  annotated video into the  Assets/  directory.
================================================================

Usage:
    python draw_from_json.py
    python draw_from_json.py --video input_video.mp4 --json detection_results.json
    python draw_from_json.py --video footage.mp4 --json results.json --assets_dir MyAssets
"""

import argparse
import json
import os
from pathlib import Path

import cv2
from tqdm import tqdm


# ──────────────────────────────────────────────
# 1.  Argument Parsing
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw YOLO detections from JSON onto the source video."
    )
    parser.add_argument(
        "--video", default="input_video.mp4",
        help="Path to the original source video (default: input_video.mp4)"
    )
    parser.add_argument(
        "--json", default="detection_results.json",
        help="Path to detection_results.json produced by yolo_detection.py"
    )
    parser.add_argument(
        "--assets_dir", default="Assets",
        help="Output directory for the annotated video (default: Assets/)"
    )
    parser.add_argument(
        "--box_thickness", type=int, default=2,
        help="Bounding box border thickness in pixels (default: 2)"
    )
    parser.add_argument(
        "--font_scale", type=float, default=0.55,
        help="Label font scale (default: 0.55)"
    )
    return parser.parse_args()


# ──────────────────────────────────────────────
# 2.  Colour Palette (mirrors yolo_detection.py)
# ──────────────────────────────────────────────
_PALETTE = [
    ( 56,  56, 255), (151, 157, 255), ( 31, 112, 255), ( 29, 178, 255),
    ( 49, 210, 207), ( 10, 249,  72), ( 23, 204, 146), (134, 219,  61),
    ( 52, 147,  26), (187, 212,   0), (168, 153,  44), (255, 194,   0),
    (255, 111,  31), (  0,  69, 255), (188,   0, 255), (255, 180,   0),
]

def _color_for(class_id: int) -> tuple[int, int, int]:
    """Deterministic per-class colour (same mapping as yolo_detection.py)."""
    return _PALETTE[class_id % len(_PALETTE)]


# ──────────────────────────────────────────────
# 3.  Drawing
# ──────────────────────────────────────────────
def draw_boxes(frame, detections: list[dict],
               thickness: int, font_scale: float) -> None:
    """
    Draw bounding boxes and labels onto *frame* in-place.

    Parameters
    ----------
    frame       : BGR numpy array from cv2.VideoCapture.read()
    detections  : list of detection dicts from the JSON
    thickness   : border line thickness (px)
    font_scale  : OpenCV font scale for labels
    """
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        cls_id = det["class_id"]
        color  = _color_for(cls_id)
        label  = f"{det['class_name']}  {det['confidence']:.2f}"

        # ── Bounding rectangle ─────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=thickness)

        # ── Label background ───────────────────────────────
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        bg_top_left  = (x1, max(y1 - th - baseline - 4, 0))
        bg_bot_right = (x1 + tw + 4, max(y1, th + baseline + 4))
        cv2.rectangle(frame, bg_top_left, bg_bot_right, color, cv2.FILLED)

        # ── Label text ─────────────────────────────────────
        cv2.putText(
            frame, label,
            (x1 + 2, max(y1 - baseline - 2, th + 2)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255), thickness=1, lineType=cv2.LINE_AA,
        )


# ──────────────────────────────────────────────
# 4.  Assets Directory Setup
# ──────────────────────────────────────────────
def ensure_assets_dir(assets_dir: str) -> Path:
    """
    Create the Assets directory if it does not already exist.
    Returns the resolved Path object.
    """
    path = Path(assets_dir)
    path.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Assets directory ready → {path.resolve()}")
    return path


# ──────────────────────────────────────────────
# 5.  Output Path Builder
# ──────────────────────────────────────────────
def build_output_path(assets_dir: Path, source_video: str) -> Path:
    """
    Derive the output filename from the source video name.
    Example: input_video.mp4  →  Assets/input_video_annotated.mp4
    """
    stem   = Path(source_video).stem          # e.g. "input_video"
    output = assets_dir / f"{stem}_annotated.mp4"
    return output


# ──────────────────────────────────────────────
# 6.  Main Pipeline
# ──────────────────────────────────────────────
def run(args: argparse.Namespace) -> None:

    # ── Load JSON ───────────────────────────────
    print(f"[INFO] Loading JSON → {args.json}")
    with open(args.json, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Build a fast frame_index → detections lookup
    frame_lookup: dict[int, list[dict]] = {
        f["frame_index"]: f["detections"]
        for f in data["frames"]
    }
    total_json_frames = len(data["frames"])
    print(f"[INFO] JSON contains {total_json_frames} annotated frames.")

    # ── Open source video ────────────────────────
    print(f"[INFO] Opening video → {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or data.get("video_fps", 25.0)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video: {total_frames} frames | {width}×{height} | {fps:.2f} FPS")

    # ── Set up Assets directory & output path ────
    assets_path = ensure_assets_dir(args.assets_dir)
    output_path = build_output_path(assets_path, args.video)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create VideoWriter at: {output_path}")

    print(f"[INFO] Output video  → {output_path.resolve()}")

    # ── Frame loop ───────────────────────────────
    frame_idx = 0
    pbar = tqdm(total=total_frames, unit="frame", desc="Redrawing")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Pull detections for this frame (empty list if frame had none)
        detections = frame_lookup.get(frame_idx, [])

        # Draw directly onto the frame (no copy needed — raw frame discarded after write)
        draw_boxes(frame, detections,
                   thickness=args.box_thickness,
                   font_scale=args.font_scale)

        # Stamp frame index + timestamp
        timestamp_sec = frame_idx / fps
        cv2.putText(
            frame,
            f"Frame {frame_idx:05d} | {timestamp_sec:.3f}s",
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (200, 200, 200), 1, cv2.LINE_AA,
        )

        writer.write(frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()

    print(f"\n[INFO] Processed {frame_idx} frames.")
    print(f"[DONE] Annotated video saved → {output_path.resolve()}")


# ──────────────────────────────────────────────
# 7.  Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    run(args)