# code/metadata_handler.py
import json
from config import Config

class MetadataHandler:
    """
    Handles the accumulation and export of detection metadata.
    """
    def __init__(self):
        self.results_metadata = []

    def add_frame_data(self, frame_index: int, fps: float, results):
        """
        Extracts and stores metadata from a YOLO results object for a single frame.
        """
        timestamp = frame_index / fps
        frame_detections = []

        # Extract boxes and classes from YOLO Results object
        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = results.names[cls_id]
            coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            conf = float(box.conf[0])

            frame_detections.append({
                "class_name": class_name,
                "bbox": [round(c, 2) for c in coords],
                "confidence": round(conf, 4)
            })

        self.results_metadata.append({
            "frame_index": frame_index,
            "timestamp_sec": round(timestamp, 4),
            "detections": frame_detections
        })

    def export(self, filename: str = Config.RESULTS_JSON):
        """
        Saves the accumulated metadata to a JSON file.
        """
        with open(filename, 'w') as f:
            json.dump(self.results_metadata, f, indent=4)
        print(f"Metadata results saved to: {filename}")
