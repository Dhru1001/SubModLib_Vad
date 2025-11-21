# ============================================================================
# merge_video.py
# ============================================================================
import os
import numpy as np
from PIL import Image
import cv2
from config import Config
from embeddings import EmbeddingsExtractor
from functions import (
    FacilityLocationSelector,
    GraphCutSelector,
    DisparityMinSelector,
    DisparitySumSelector
)

class VideoMerger:
    """Merge selected frames into a video using frame selection algorithms."""
    
    def __init__(self, num_selected=64, fps=30, overwrite=True):
        self.num_selected = num_selected
        self.fps = fps
        self.overwrite = overwrite
        self.embeddings_extractor = EmbeddingsExtractor()
    
    def _write_video(self, out_path, frame_images, selected_indices):
        """Write selected frames to video file."""
        if not selected_indices:
            print("[WARN] No frames to write.")
            return False
        
        first_frame = frame_images[selected_indices[0]]
        width, height = first_frame.size
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, self.fps, (width, height))
        
        if not writer.isOpened():
            print(f"[ERROR] Failed to open video writer: {out_path}")
            return False
        
        for idx in selected_indices:
            pil_img = frame_images[idx]
            
            if pil_img.size != (width, height):
                pil_img = pil_img.resize((width, height), Image.BILINEAR)
            
            frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            if frame_bgr.dtype != np.uint8:
                frame_bgr = (255 * frame_bgr).astype(np.uint8)
            
            writer.write(frame_bgr)
        
        writer.release()
        return True
    
    def merge_with_facility_location(self, frames_dir, output_dir=None):
        """Merge frames using Facility Location selection with delta expansion."""
        if output_dir is None:
            output_dir = Config.FL_OUTPUT_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        for subfolder in sorted(os.listdir(frames_dir)):
            subfolder_path = os.path.join(frames_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            
            out_video_path = os.path.join(output_dir, f"{subfolder}_selected.mp4")
            
            if os.path.exists(out_video_path) and not self.overwrite:
                print(f"[SKIP] Output exists: {out_video_path}")
                continue
            
            # Load frames
            frame_files = sorted([
                os.path.join(subfolder_path, f)
                for f in os.listdir(subfolder_path)
                if f.lower().endswith(Config.IMAGE_EXTENSIONS)
            ])
            
            if not frame_files:
                print(f"[WARN] No frames in {subfolder_path}")
                continue
            
            frame_images = [Image.open(f).convert("RGB") for f in frame_files]
            
            # Load embeddings (should already exist)
            print(f"[INFO] Processing {subfolder}...")
            feature_matrix = self.embeddings_extractor.load_embeddings(subfolder)
            
            if feature_matrix is None:
                print(f"[WARN] Embeddings not found for {subfolder}. Computing now...")
                feature_matrix = self.embeddings_extractor.extract_and_store(
                    frame_images, subfolder
                )
            
            # Select frames with delta
            selector = FacilityLocationSelector(budget=self.num_selected, delta=Config.FL_DELTA)
            selected_indices = selector.select(feature_matrix)
            
            # Write video
            if self._write_video(out_video_path, frame_images, selected_indices):
                print(f"[SUCCESS] Saved: {out_video_path}")
    
    def merge_with_graph_cut(self, frames_dir, output_dir=None, lambda_values=None):
        """Merge frames using Graph Cut selection with multiple lambda values."""
        if output_dir is None:
            output_dir = Config.GC_OUTPUT_DIR
        
        if lambda_values is None:
            lambda_values = Config.LAMBDA_VALUES
        
        os.makedirs(output_dir, exist_ok=True)
        
        for subfolder in sorted(os.listdir(frames_dir)):
            subfolder_path = os.path.join(frames_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            
            # Load frames
            frame_files = sorted([
                os.path.join(subfolder_path, f)
                for f in os.listdir(subfolder_path)
                if f.lower().endswith(Config.IMAGE_EXTENSIONS)
            ])
            
            if not frame_files:
                print(f"[WARN] No frames in {subfolder_path}")
                continue
            
            frame_images = [Image.open(f).convert("RGB") for f in frame_files]
            
            # Load embeddings (should already exist)
            print(f"[INFO] Processing {subfolder}...")
            feature_matrix = self.embeddings_extractor.load_embeddings(subfolder)
            
            if feature_matrix is None:
                print(f"[WARN] Embeddings not found for {subfolder}. Computing now...")
                feature_matrix = self.embeddings_extractor.extract_and_store(
                    frame_images, subfolder
                )
            
            # Try each lambda value
            for lambda_val in lambda_values:
                lambda_str = f"{lambda_val}"
                lambda_folder = os.path.join(output_dir, f"lambda_{lambda_str}")
                os.makedirs(lambda_folder, exist_ok=True)
                
                out_video_path = os.path.join(
                    lambda_folder,
                    f"{subfolder}_lambda_{lambda_str}.mp4"
                )
                
                if os.path.exists(out_video_path) and not self.overwrite:
                    print(f"[SKIP] λ={lambda_str}: {out_video_path}")
                    continue
                
                # Select with GraphCut
                selector = GraphCutSelector(budget=self.num_selected)
                selected_indices = selector.select(feature_matrix, lambda_val)
                
                # Write video
                if self._write_video(out_video_path, frame_images, selected_indices):
                    print(f"[SUCCESS] λ={lambda_str}: {out_video_path}")
    
    def merge_with_disparity_min(self, frames_dir, output_dir=None):
        """Merge frames using Disparity Min selection with delta expansion."""
        if output_dir is None:
            output_dir = Config.DM_OUTPUT_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        for subfolder in sorted(os.listdir(frames_dir)):
            subfolder_path = os.path.join(frames_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            
            out_video_path = os.path.join(output_dir, f"{subfolder}_selected.mp4")
            
            if os.path.exists(out_video_path) and not self.overwrite:
                print(f"[SKIP] Output exists: {out_video_path}")
                continue
            
            # Load frames
            frame_files = sorted([
                os.path.join(subfolder_path, f)
                for f in os.listdir(subfolder_path)
                if f.lower().endswith(Config.IMAGE_EXTENSIONS)
            ])
            
            if not frame_files:
                print(f"[WARN] No frames in {subfolder_path}")
                continue
            
            frame_images = [Image.open(f).convert("RGB") for f in frame_files]
            
            # Load embeddings (should already exist)
            print(f"[INFO] Processing {subfolder}...")
            feature_matrix = self.embeddings_extractor.load_embeddings(subfolder)
            
            if feature_matrix is None:
                print(f"[WARN] Embeddings not found for {subfolder}. Computing now...")
                feature_matrix = self.embeddings_extractor.extract_and_store(
                    frame_images, subfolder
                )
            
            # Select frames with delta
            selector = DisparityMinSelector(budget=self.num_selected, delta=Config.FL_DELTA)
            selected_indices = selector.select(feature_matrix)
            
            # Write video
            if self._write_video(out_video_path, frame_images, selected_indices):
                print(f"[SUCCESS] Saved: {out_video_path}")
    
    def merge_with_disparity_sum(self, frames_dir, output_dir=None):
        """Merge frames using Disparity Sum selection with delta expansion."""
        if output_dir is None:
            output_dir = Config.DS_OUTPUT_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        for subfolder in sorted(os.listdir(frames_dir)):
            subfolder_path = os.path.join(frames_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            
            out_video_path = os.path.join(output_dir, f"{subfolder}_selected.mp4")
            
            if os.path.exists(out_video_path) and not self.overwrite:
                print(f"[SKIP] Output exists: {out_video_path}")
                continue
            
            # Load frames
            frame_files = sorted([
                os.path.join(subfolder_path, f)
                for f in os.listdir(subfolder_path)
                if f.lower().endswith(Config.IMAGE_EXTENSIONS)
            ])
            
            if not frame_files:
                print(f"[WARN] No frames in {subfolder_path}")
                continue
            
            frame_images = [Image.open(f).convert("RGB") for f in frame_files]
            
            # Load embeddings (should already exist)
            print(f"[INFO] Processing {subfolder}...")
            feature_matrix = self.embeddings_extractor.load_embeddings(subfolder)
            
            if feature_matrix is None:
                print(f"[WARN] Embeddings not found for {subfolder}. Computing now...")
                feature_matrix = self.embeddings_extractor.extract_and_store(
                    frame_images, subfolder
                )
            
            # Select frames with delta
            selector = DisparitySumSelector(budget=self.num_selected, delta=Config.FL_DELTA)
            selected_indices = selector.select(feature_matrix)
            
            # Write video
            if self._write_video(out_video_path, frame_images, selected_indices):
                print(f"[SUCCESS] Saved: {out_video_path}")
