# ============================================================================
# config.py - Centralized configuration
# ============================================================================
import os

class Config:
    # Frame Extraction
    VIDEOS_DIR = "/backup/palak/VAD/VERA/UCF_Dummy_Dataset"
    FRAMES_OUTPUT_DIR = "/backup/palak/SubModLib/data/UCF_Dummy_Dataset_Frames"
    FRAME_INTERVAL = 1
    
    # Frame Selection & Video Merging
    NUM_SELECTED = 64
    OUTPUT_FPS = 30
    OVERWRITE = True
    
    # Output Folders for Selection Algorithms
    FL_OUTPUT_DIR = "/backup/palak/SubModLib/data/Dummmy_Data_Facility_location"
    GC_OUTPUT_DIR = "/backup/palak/SubModLib/data/Dummmy_Data_Graph_Cut"
    
    # Graph Cut Lambda Values
    LAMBDA_VALUES = [-0.5, -1.0]
    
    # Embeddings Configuration
    EMBEDDINGS_DB_DIR = "/backup/palak/SubModLib/data/Dummmy_Data_Embeddings_DB"
    
    # Device
    DEVICE = "cuda"  # will be set to cpu if cuda not available
    
    # Video extensions
    VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
    
    # CLIP Model
    CLIP_MODEL = "ViT-B/16"