
# ============================================================================
# merge_video.py
# ============================================================================
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import clip
from config import Config
from functions import FacilityLocationSelector, GraphCutSelector

class VideoMerger:
    """Merge selected frames into a video using frame selection algorithms."""
    
    def __init__(self, num_selected=64, fps=30, overwrite=True):
        self.num_selected = num_selected
        self.fps = fps
        self.overwrite = overwrite
        
        # Setup device
        self.device = Config.DEVICE if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device}")
        
        # Load CLIP model
        self._load_clip_model()
        
        # Preprocessing
        self.preprocess_fn = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
    
    def _load_clip_model(self):
        """Load CLIP model."""
        model, _ = clip.load(Config.CLIP_MODEL, device=self.device, jit=False)
        self.image_encoder = model.visual
        self.image_encoder.eval()
    
    def _extract_features(self, frame_images):
        """Extract CLIP features for all frames."""
        feature_matrix = []
        
        for image in frame_images:
            img_tensor = self.preprocess_fn(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if self.device == "cuda":
                    features = self.image_encoder(img_tensor.half()).cpu().numpy().squeeze()
                else:
                    features = self.image_encoder(img_tensor).cpu().numpy().squeeze()
            
            feature_matrix.append(features)
        
        return np.array(feature_matrix)
    
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
    
    def merge_with_facility_location(self, frames_dir, output_dir):
        """Merge frames using Facility Location selection."""
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
            
            # Extract features and select
            print(f"[INFO] Processing {subfolder}...")
            feature_matrix = self._extract_features(frame_images)
            
            selector = FacilityLocationSelector(budget=self.num_selected)
            selected_indices = selector.select(feature_matrix)
            
            # Write video
            if self._write_video(out_video_path, frame_images, selected_indices):
                print(f"[SUCCESS] Saved: {out_video_path}")
    
    def merge_with_graph_cut(self, frames_dir, output_dir, lambda_values=None):
        """Merge frames using Graph Cut selection with multiple lambda values."""
        if lambda_values is None:
            lambda_values = [-0.5, -1.0]
        
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
            
            # Extract features once
            print(f"[INFO] Processing {subfolder}...")
            feature_matrix = self._extract_features(frame_images)
            
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
