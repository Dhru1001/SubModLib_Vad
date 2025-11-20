
# ============================================================================
# frame_extraction.py
# ============================================================================
import cv2
import os
from config import Config

class FrameExtractor:
    """Extract frames from video files."""
    
    def __init__(self, frame_interval=1):
        self.frame_interval = frame_interval
    
    def extract_frames(self, video_path, output_folder):
        """
        Extract frames from a video.
        
        Args:
            video_path: path to video file
            output_folder: folder to save extracted frames
            
        Returns:
            frame_count: number of frames extracted
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARN] Cannot open video: {video_path}")
            return 0
        
        os.makedirs(output_folder, exist_ok=True)
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % self.frame_interval == 0:
                frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        print(f"[INFO] Extracted {saved_count} frames from {os.path.basename(video_path)}")
        return saved_count
    
    def extract_all_videos(self, videos_dir, output_base_dir):
        """Extract frames from all videos in a directory."""
        os.makedirs(output_base_dir, exist_ok=True)
        videos = [f for f in os.listdir(videos_dir) 
                 if f.lower().endswith(Config.VIDEO_EXTENSIONS)]
        
        if not videos:
            print(f"[ERROR] No video files found in: {videos_dir}")
            return False
        
        for video_file in videos:
            video_path = os.path.join(videos_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            output_folder = os.path.join(output_base_dir, video_name)
            self.extract_frames(video_path, output_folder)
        
        print("[DONE] All frames extracted successfully.")
        return True
