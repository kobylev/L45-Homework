# code/processor.py
import cv2
from config import Config
from detector import Detector
from metadata_handler import MetadataHandler
from video_writer import VideoWriterHandler

class VideoProcessor:
    """
    Coordinates the video detection pipeline, processing frames sequentially.
    """
    def __init__(self, input_path: str = Config.INPUT_VIDEO, output_path: str = Config.OUTPUT_VIDEO):
        self.input_path = input_path
        self.output_path = output_path
        self.detector = Detector()
        self.meta_handler = MetadataHandler()
        self.writer = None

    def _setup_resources(self, cap):
        """
        Initializes the video writer and extracts video properties.
        """
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.writer = VideoWriterHandler(self.output_path, fps, width, height)
        return fps

    def process_video(self):
        """
        Loops through each frame, runs detection, and delegates storage/output.
        """
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.input_path}")
            return

        fps = self._setup_resources(cap)
        frame_index = 0
        print(f"Starting processing: {self.input_path} at {fps:.2f} FPS")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Inference
                results = self.detector.detect_frame(frame)
                
                # Metadata extraction and storage
                self.meta_handler.add_frame_data(frame_index, fps, results)

                # Visualization and writing (if enabled)
                if Config.SAVE_VISUALS:
                    annotated_frame = results.plot()
                    self.writer.write_frame(annotated_frame)
                
                frame_index += 1
                if frame_index % 30 == 0:
                    print(f"Processed frame {frame_index} ({(frame_index/fps):.2f}s)")

        finally:
            cap.release()
            if self.writer:
                self.writer.release()

    def export_metadata(self, filename: str = Config.RESULTS_JSON):
        """
        Delegates the metadata export task to the metadata handler.
        """
        self.meta_handler.export(filename)
