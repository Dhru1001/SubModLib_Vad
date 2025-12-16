# ============================================================================
# config.py - Centralized configuration
# ============================================================================
import os

class Config:
    # Frame Extraction
    VIDEOS_DIR = "/backup/palak/VAD/VERA/UCF_Dummy_Dataset"
    # VIDEOS_DIR = "/backup/palak/SubModLib/adc"
    FRAMES_OUTPUT_DIR = "/backup/palak/SubModLib/data/UCF_Dummy_Dataset_Frames"
    FRAME_INTERVAL = 1
    
    # Frame Selection & Video Merging
    # --> consider Delta For FL in case of NUM_SELECTED 
    NUM_SELECTED = 64
    OUTPUT_FPS = 30
    OVERWRITE = True

    # Delta Expansion Configuration
    # Set to True to enable delta expansion for FL, DM, DS
    # Set to False to disable delta expansion (use selected frames as-is)
    USE_DELTA = False
    
    # Output Folders for Selection Algorithms
    FL_OUTPUT_DIR = "/backup/palak/SubModLib/data/Dummmy_Data_Facility_location_16_frames"
    GC_OUTPUT_DIR = "/backup/palak/SubModLib/data/Dummmy_Data_Graph_Cut"
    DM_OUTPUT_DIR = "/backup/palak/SubModLib/data/Dummmy_Data_DisparityMin"
    DS_OUTPUT_DIR = "/backup/palak/SubModLib/data/Dummmy_Data_DisparitySum"
    FLCG_OUTPUT_DIR = "/backup/palak/SubModLib/data/Dummmy_Data_FLConditionalGain"
    
    # Facility Location Delta (neighbor offsets to include)
    # Example:
    # If delta = [-2, 0, +2] and selected_indices = [102]
    # Then: 102 + (-2) = 100
    #       102 + 0 = 102
    #       102 + (+2) = 104
    # Result: [100, 101, 102, 103, 104] (sorted and unique)
    FL_DELTA = [0]

    '''
    You can modify it to whatever you want:
    - `FL_DELTA = [-1, +1]` → Only neighbors, skip center
    - `FL_DELTA = [-2, -1, 0, +1, +2]` → Wider neighborhood
    - `FL_DELTA = [0]` → Just original frames (no expansion)

    ## 2. **Facility Location - Delta Expansion**
    The `_apply_delta()` method:
    - Takes FL selected frames `[3, 6, 7, 9]`
    - Applies delta: `[-1, 0, +1]`
    - Generates: `{2,3,4,5,6,7,6,7,8,8,9,10}`
    - **Removes duplicates** using set
    - **Sorts** the result
    - Returns: `[2,3,4,5,6,7,8,9,10]`

    **Example Output:**
    [INFO] FL selected frames: [3, 6, 7, 9]
    [INFO] After delta [-1, 0, +1] expansion: [2, 3, 4, 5, 6, 7, 8, 9, 10]
    '''
    # Facility Location Conditional Gain Private Set
    # Path to folder containing private frames (already selected frames)
    # Empty string means no private data (standard FL behavior)
    # Example: "/path/to/private/frames" - folder with selected key frames
    FLCG_PRIVATE_FRAMES_DIR = "/backup/palak/SubModLib/data/private_set_normal"  # Empty by default

    # Feature Extraction Method
    # Options: "clip" or "numpy"
    # "clip" - Uses OpenAI CLIP embeddings (deep learning features)
    # "numpy" - Uses raw pixel values as features (simple, fast)
    FEATURE_TYPE = "clip"  # Default: CLIP embeddings


    # Graph Cut Lambda Values
    LAMBDA_VALUES = [-25, -30, -50]
    
    # Embeddings Configuration
    EMBEDDINGS_DB_DIR = "/backup/palak/SubModLib/data/Dummy_Data_clip_Embeddings_DB"
    
    # Device
    DEVICE = "cuda"  # will be set to cpu if cuda not available
    
    # Video extensions
    VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
    
    # CLIP Model
    CLIP_MODEL = "ViT-B/16"