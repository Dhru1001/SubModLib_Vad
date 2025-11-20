# Video Frame Extraction & Selection Pipeline

A comprehensive Python pipeline for extracting frames from videos, computing CLIP embeddings, and selecting representative frames using submodular optimization techniques (Facility Location & Graph Cut). Includes intelligent embeddings storage and delta-based frame expansion.

## Features

✅ **Multi-stage Pipeline**
- Frame extraction from videos
- CLIP embedding computation & storage
- Intelligent frame selection using submodular functions
- Video merging from selected frames

✅ **Advanced Frame Selection**
- **Facility Location** with configurable delta expansion
- **Graph Cut** with multiple lambda values
- Memoization support for submodlib functions

✅ **Embeddings Management**
- Vector database storage (`.npz` format)
- Pre-computed embeddings reuse
- Metadata tracking with JSON

✅ **Flexible Execution**
- Run full pipeline or individual steps
- Use pre-computed embeddings without recomputation
- Configurable parameters for all stages

## Project Structure

```
video-processing-repo/
├── main.py                    # Main orchestration script
├── config.py                  # Centralized configuration
├── frame_extraction.py        # Frame extraction module
├── merge_video.py             # Video merging & selection
├── embeddings.py              # CLIP embeddings extraction & management
├── functions/
│   ├── __init__.py
│   ├── facility_location.py  # Facility Location selector with delta
│   └── graph_cut.py          # Graph Cut selector
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone and Setup

```bash
git clone <repository-url>
cd video-processing-repo
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### requirements.txt

```
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
Pillow>=8.0.0
numpy>=1.19.0
clip @ git+https://github.com/openai/CLIP.git
submodlib>=1.1.1
```

## Configuration

Edit `config.py` to customize all parameters:

```python
class Config:
    # Input/Output Paths
    VIDEOS_DIR = "/path/to/videos"
    FRAMES_OUTPUT_DIR = "/path/to/frames"
    EMBEDDINGS_DB_DIR = "/path/to/embeddings"
    
    # Facility Location Output
    FL_OUTPUT_DIR = "/path/to/fl_output"
    
    # Graph Cut Output
    GC_OUTPUT_DIR = "/path/to/gc_output"
    
    # Processing Parameters
    NUM_SELECTED = 64              # Initial frames to select
    OUTPUT_FPS = 30                # Output video frame rate
    FRAME_INTERVAL = 1             # Extract every nth frame
    OVERWRITE = True               # Overwrite existing outputs
    
    # Facility Location Delta (frame expansion)
    # If delta=[-2, 0, +2] and FL selects frame 102,
    # it expands to [100, 101, 102, 103, 104]
    FL_DELTA = [-1, 0, +1]
    
    # Graph Cut Lambda Values (diversity parameters)
    LAMBDA_VALUES = [-0.5, -1.0]
    
    # Model Configuration
    CLIP_MODEL = "ViT-B/16"
    DEVICE = "cuda"  # Auto-switches to CPU if CUDA unavailable
```

### Key Configuration Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `NUM_SELECTED` | Frames selected by algorithm (before delta expansion) | `64` |
| `FL_DELTA` | Range expansion around selected frames | `[-2, 0, +2]` |
| `LAMBDA_VALUES` | Graph Cut diversity parameters | `[-0.5, -1.0]` |
| `OUTPUT_FPS` | Output video frame rate | `30` |
| `CLIP_MODEL` | CLIP model variant | `"ViT-B/16"` |

## Usage

### Full Pipeline (First Run)

Run all steps: extract frames → compute embeddings → select frames with FL & GC

```bash
python main.py
```

### Step-by-Step Execution

**Step 1: Extract Frames**
```bash
python main.py --step 1
```
Extracts all frames from videos in `VIDEOS_DIR`

**Step 2: Facility Location Selection**
```bash
python main.py --step 2
```
Uses pre-computed embeddings (or computes if missing)
Applies delta expansion to selected frames

**Step 3: Graph Cut Selection**
```bash
python main.py --step 3
```
Uses pre-computed embeddings
Creates videos for each lambda value

### Subsequent Runs (With Pre-computed Embeddings)

After embeddings are stored, re-running is instant:

```bash
# This will use cached embeddings
python main.py --step 2

# Or just run a specific algorithm
python main.py --step 3
```

### View Embeddings Database

```bash
python main.py --show-embeddings
```

Output:
```
======================================================================
EMBEDDINGS DATABASE STATISTICS
======================================================================

Video: video1
  Frames: 450
  Embedding Dimension: 512

Video: video2
  Frames: 380
  Embedding Dimension: 512

Total Videos: 2
======================================================================
```

## How Delta Expansion Works

### Example with FL_DELTA = [-2, 0, +2]

**Step 1: Facility Location selects frames**
```
Selected: [3, 6, 7, 9]
```

**Step 2: Delta expands each frame**
```
3 + [-2, 0, +2] → [1, 2, 3, 4, 5]
6 + [-2, 0, +2] → [4, 5, 6, 7, 8]
7 + [-2, 0, +2] → [5, 6, 7, 8, 9]
9 + [-2, 0, +2] → [7, 8, 9, 10, 11]
```

**Step 3: Deduplicate and sort**
```
Result: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
Total frames: 11
```

### Controlling Final Frame Count

To get exactly ~80 frames:

```python
# With delta = [-2, 0, +2], each frame expands ~5x
# So reduce initial selection:
NUM_SELECTED = 16  # ~16 × 5 = ~80 frames
```

## API Reference

### EmbeddingsExtractor

```python
from embeddings import EmbeddingsExtractor

extractor = EmbeddingsExtractor()

# Extract and store embeddings
embeddings = extractor.extract_and_store(frame_images, video_name)

# Load pre-computed embeddings
embeddings = extractor.load_embeddings(video_name)

# View stored embeddings
extractor.print_database_stats()

# List all stored videos
videos = extractor.list_stored_embeddings()
```

### FrameExtractor

```python
from frame_extraction import FrameExtractor

extractor = FrameExtractor(frame_interval=1)

# Extract from single video
extractor.extract_frames(video_path, output_folder)

# Extract from all videos in directory
extractor.extract_all_videos(videos_dir, output_dir)
```

### FacilityLocationSelector

```python
from functions import FacilityLocationSelector

selector = FacilityLocationSelector(budget=64, delta=[-1, 0, +1])
selected_indices = selector.select(feature_matrix)

# Clear submodlib memoization
selector.clear_memoization()
```

### GraphCutSelector

```python
from functions import GraphCutSelector

selector = GraphCutSelector(budget=64)
selected_indices = selector.select(feature_matrix, lambda_val=-0.5)

# Clear submodlib memoization
selector.clear_memoization()
```

### VideoMerger

```python
from merge_video import VideoMerger

merger = VideoMerger(num_selected=64, fps=30, overwrite=True)

# Merge with Facility Location
merger.merge_with_facility_location(frames_dir, output_dir)

# Merge with Graph Cut
merger.merge_with_graph_cut(frames_dir, output_dir, lambda_values=[-0.5, -1.0])
```

## Output Structure

### Embeddings Database
```
Embeddings_DB/
├── video1_embeddings.npz       # Shape: (num_frames, 512)
├── video1_metadata.json        # Metadata with frame count
├── video2_embeddings.npz
├── video2_metadata.json
...
```

### Facility Location Output
```
Selected_frames64(usingCLIP)/
├── video1_selected.mp4
├── video2_selected.mp4
...
```

### Graph Cut Output
```
Selected_frames64(GraphCut_Negative_Values)/
├── lambda_-0.5/
│   ├── video1_lambda_-0.5.mp4
│   ├── video2_lambda_-0.5.mp4
│   ...
├── lambda_-1.0/
│   ├── video1_lambda_-1.0.mp4
│   ├── video2_lambda_-1.0.mp4
│   ...
```

## Performance Tips

1. **GPU Acceleration**
   - Ensure CUDA is properly installed
   - Set `DEVICE = "cuda"` in config
   - Automatic CPU fallback if GPU unavailable

2. **Memory Optimization**
   - Reduce `NUM_SELECTED` for large videos
   - Process videos in batches if needed
   - Embeddings stored as compressed `.npz` files

3. **Reuse Embeddings**
   - First run computes embeddings once
   - Subsequent runs load from database
   - Clear database if you need to recompute

4. **Frame Extraction**
   - Increase `FRAME_INTERVAL` to extract fewer frames
   - Reduces computation for high FPS videos

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution:** Set `DEVICE = "cpu"` or reduce `NUM_SELECTED`

### Issue: Embeddings Not Found
**Solution:** Ensure embeddings are computed first with `python main.py`

### Issue: No Frames Extracted
**Solution:** Check video format is in `VIDEO_EXTENSIONS` and path is correct

### Issue: Graph Cut Selection Fails
**Solution:** Verify `LAMBDA_VALUES` are appropriate (typically negative for diversity)

### Issue: Output Video Won't Play
**Solution:** Ensure all selected frame indices are within valid range; check frame count

## Algorithm Details

### Facility Location
- **Purpose:** Select diverse, representative frames
- **Method:** Submodular optimization using FacilityLocationFunction
- **Delta:** Adds neighboring frames around selected ones for temporal continuity
- **Use Case:** Key frame extraction, video summarization

### Graph Cut
- **Purpose:** Trade-off between diversity and representativeness
- **Method:** Submodular optimization using GraphCutFunction
- **Lambda:** Controls diversity trade-off (negative values encourage diversity)
- **Use Case:** Flexible frame selection with parameter tuning

## Advanced Usage

### Custom Delta Values

```python
# In config.py
FL_DELTA = [-3, -1, 0, +1, +3]  # Skip adjacent frames
```

Output: Sparser frame selection with larger gaps

### Multiple Lambda Experiments

```python
# In config.py
LAMBDA_VALUES = [-0.1, -0.5, -1.0, -2.0]  # More lambda values
```

Creates separate videos for each lambda parameter

### Different Models

```python
# In config.py
CLIP_MODEL = "ViT-L/14"  # Larger model, slower but better features
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework |
| `torchvision` | Vision utilities |
| `clip` | OpenAI's CLIP model |
| `opencv-python` | Video processing |
| `Pillow` | Image handling |
| `numpy` | Numerical computing |
| `submodlib` | Submodular optimization |

## System Requirements

- **Python:** 3.8+
- **RAM:** 8GB+ (16GB+ recommended)
- **VRAM:** 4GB+ (for GPU acceleration, optional)
- **Disk:** Sufficient space for frames and videos
- **Internet:** For initial CLIP model download

## License

This project uses open-source libraries. Refer to individual licenses.

## Citation

If you use this pipeline in research, please cite:

- CLIP: https://github.com/openai/CLIP
- SubmodLib: https://github.com/decile-team/submodlib

## Support & Issues

For bugs or feature requests, please refer to the repository issues page.

## Changelog

### v1.0.0
- Initial release
- Frame extraction, CLIP embeddings, Facility Location, Graph Cut
- Delta expansion for temporal continuity
- Embeddings database management

---

**Last Updated:** November 2024  
**Status:** Production Ready