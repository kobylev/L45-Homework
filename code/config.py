# code/config.py
import torch

class Config:
    # Model configuration
    MODEL_NAME = "yolov8n.pt"  # Pre-trained model
    CONFIDENCE_THRESHOLD = 0.5
    
    # Input/Output configuration
    INPUT_VIDEO = "input_video.mp4"
    OUTPUT_VIDEO = "output_video_detected.mp4"
    RESULTS_JSON = "../metadata/detection_results.json"
    
    # Processing parameters
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_VISUALS = True
