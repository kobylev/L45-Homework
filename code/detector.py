# code/detector.py
from ultralytics import YOLO
from config import Config

class Detector:
    def __init__(self, model_path: str = Config.MODEL_NAME):
        """
        Initializes the YOLO model from the provided path.
        """
        self.model = YOLO(model_path)
        self.device = Config.DEVICE
        print(f"Model loaded on: {self.device}")

    def detect_frame(self, frame):
        """
        Runs inference on a single frame.
        """
        results = self.model.predict(
            source=frame, 
            conf=Config.CONFIDENCE_THRESHOLD, 
            device=self.device, 
            verbose=False
        )
        return results[0]  # Return results for the single image
