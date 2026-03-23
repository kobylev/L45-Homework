# code/video_writer.py
import cv2
from config import Config

class VideoWriterHandler:
    """
    Manages the initialization and frame-by-frame writing for the output video.
    """
    def __init__(self, output_path: str, fps: float, width: int, height: int):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.writer = self._init_writer()

    def _init_writer(self):
        """
        Initializes the OpenCV VideoWriter with the specified parameters.
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(
            self.output_path, 
            fourcc, 
            self.fps, 
            (self.width, self.height)
        )

    def write_frame(self, frame):
        """
        Writes a single annotated frame to the output video.
        """
        if self.writer is not None:
            self.writer.write(frame)

    def release(self):
        """
        Releases the VideoWriter resource.
        """
        if self.writer is not None:
            self.writer.release()
            print(f"Visual results saved to: {self.output_path}")
