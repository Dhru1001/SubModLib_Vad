# ============================================================================
# config.py - Centralized configuration
# ============================================================================
import os

class Config:
    # Frame Extraction
    VIDEOS_DIR = "/backup/palak/VAD/VERA/UCF_Dummy_Dataset"
    FRAMES_OUTPUT_DIR = "/backup/palak/SubModLib/UCF_Dummy_Dataset_Frames"
    FRAME_INTERVAL = 1
    
    # Frame Selection & Video Merging
    NUM_SELECTED = 64
    OUTPUT_FPS = 30
    OVERWRITE = True
    
    # Device
    DEVICE = "cuda"  # will be set to cpu if cuda not available
    
    # Video extensions
    VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
    
    # CLIP Model
    CLIP_MODEL = "ViT-B/16"
    LAMBDA_VALUES = [-15, 15]