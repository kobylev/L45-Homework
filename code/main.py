# code/main.py
from processor import VideoProcessor
from config import Config

def main():
    """
    Main orchestration function for YOLO video detection.
    """
    print("Initializing Video Detection Pipeline...")
    
    # Create the processor instance
    processor = VideoProcessor(
        input_path=Config.INPUT_VIDEO,
        output_path=Config.OUTPUT_VIDEO
    )
    
    # 1. Process the video and save visuals (Output 1)
    processor.process_video()
    
    # 2. Export metadata (Output 2)
    processor.export_metadata(filename=Config.RESULTS_JSON)
    
    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()
