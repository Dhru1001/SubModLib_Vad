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
    
    # Selection Algorithm Output Directories
    FL_OUTPUT_DIR = "/path/to/fl_output"
    GC_OUTPUT_DIR = "/path/to/gc_output"
    DM_OUTPUT_DIR = "/path/to/dm_output"
    DS_OUTPUT_DIR = "/path/to/ds_output"
    FLCG_OUTPUT_DIR = "/path/to/flcg_output"
    
    # Processing Parameters
    NUM_SELECTED = 64              # Initial frames to select
    OUTPUT_FPS = 30                # Output video frame rate
    FRAME_INTERVAL = 1             # Extract every nth frame
    OVERWRITE = True               # Overwrite existing outputs
    
    # Delta Expansion Configuration
    # Set to True to enable delta expansion for FL, DM, DS
    # Set to False to disable delta expansion (use selected frames as-is)
    USE_DELTA = True
    
    # Facility Location Delta (frame expansion range)
    # Only applied if USE_DELTA = True
    # If delta=[-2, 0, +2] and FL selects frame 102,
    # it expands to [100, 101, 102, 103, 104]
    FL_DELTA = [-1, 0, +1]
    
    # Graph Cut Lambda Values (diversity parameters)
    LAMBDA_VALUES = [-0.5, -1.0]
    
    # Facility Location Conditional Gain Private Set
    # Indices of frames already selected (conditioning set)
    # Empty list = standard FL behavior
    # Example: [0, 10, 20] = condition on frames 0, 10, 20
    FLCG_PRIVATE_SET = []
    
    # Model Configuration
    CLIP_MODEL = "ViT-B/16"
    DEVICE = "cuda"  # Auto-switches to CPU if CUDA unavailable
```

### Key Configuration Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `NUM_SELECTED` | int | Frames selected by algorithm (before delta expansion) | `64` |
| `USE_DELTA` | bool | Enable/disable delta expansion for FL, DM, DS | `True` |
| `FL_DELTA` | list | Range expansion around selected frames | `[-2, 0, +2]` |
| `LAMBDA_VALUES` | list | Graph Cut diversity parameters | `[-0.5, -1.0]` |
| `OUTPUT_FPS` | int | Output video frame rate | `30` |
| `CLIP_MODEL` | str | CLIP model variant | `"ViT-B/16"` |
| `FL_OUTPUT_DIR` | str | Facility Location output directory | `/path/to/fl_output` |
| `GC_OUTPUT_DIR` | str | Graph Cut output directory | `/path/to/gc_output` |
| `DM_OUTPUT_DIR` | str | Disparity Min output directory | `/path/to/dm_output` |
| `DS_OUTPUT_DIR` | str | Disparity Sum output directory | `/path/to/ds_output` |
| `FLCG_OUTPUT_DIR` | str | FL Conditional Gain output directory | `/path/to/flcg_output` |
| `FLCG_PRIVATE_SET` | list | Private set for FLCG (already selected frames) | `[0, 10, 20]` |

## Usage

### Full Pipeline (First Run)

Run all steps: extract frames → compute embeddings → select frames with all algorithms

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
Output: `FL_OUTPUT_DIR`

**Step 3: Graph Cut Selection**
```bash
python main.py --step 3
```
Uses pre-computed embeddings
Creates videos for each lambda value
Output: `GC_OUTPUT_DIR/lambda_*/`

**Step 4: Disparity Min Selection**
```bash
python main.py --step 4
```
Uses pre-computed embeddings
Maximum diversity selection with delta expansion
Output: `DM_OUTPUT_DIR`

**Step 5: Disparity Sum Selection**
```bash
python main.py --step 5
```
Uses pre-computed embeddings
Balanced diversity selection with delta expansion
Output: `DS_OUTPUT_DIR`

**Step 6: Facility Location Conditional Gain Selection**
```bash
python main.py --step 6
```
Uses pre-computed embeddings
Incremental selection based on private set with delta expansion
Output: `FLCG_OUTPUT_DIR`

### Subsequent Runs (With Pre-computed Embeddings)

After embeddings are stored, re-running is instant:

```bash
# This will use cached embeddings
python main.py --step 2

# Or just run a specific algorithm
python main.py --step 3
python main.py --step 4
python main.py --step 5

# Run all selection algorithms (skips extraction)
python main.py --step 2,3,4,5
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

## Selection Algorithms

The pipeline supports **6 different submodular optimization algorithms** for frame selection:

### 1. Facility Location (FL)
- **Purpose:** Select diverse, representative frames
- **Method:** Maximizes coverage of feature space
- **Delta:** Optional - adds neighboring frames around selected ones for temporal continuity
- **Best For:** Key frame extraction, video summarization
- **Output:** `FL_OUTPUT_DIR`
- **Parameters:** `NUM_SELECTED`, `FL_DELTA`, `USE_DELTA`

### 2. Graph Cut (GC)
- **Purpose:** Trade-off between diversity and representativeness
- **Method:** Submodular optimization with lambda parameter
- **Lambda:** Controls diversity trade-off (negative values encourage diversity)
- **Delta:** Not used (no delta expansion)
- **Best For:** Flexible frame selection with parameter tuning
- **Output:** `GC_OUTPUT_DIR/lambda_*/`
- **Parameters:** `NUM_SELECTED`, `LAMBDA_VALUES`

### 3. Disparity Min (DM)
- **Purpose:** Maximize frame diversity by minimizing maximum similarity
- **Method:** Minimizes max similarity between selected and unselected frames
- **Delta:** Optional - adds neighboring frames around selected ones for temporal continuity
- **Best For:** Outlier detection, maximum diversity selection
- **Output:** `DM_OUTPUT_DIR`
- **Parameters:** `NUM_SELECTED`, `FL_DELTA`, `USE_DELTA`
- **Concept:** Ensures selected frames are maximally dissimilar to unselected frames

### 4. Disparity Sum (DS)
- **Purpose:** Balanced diversity by minimizing total similarity sum
- **Method:** Minimizes sum of similarities between selected and unselected frames
- **Delta:** Optional - adds neighboring frames around selected ones for temporal continuity
- **Best For:** Balanced diversity, representative sampling
- **Output:** `DS_OUTPUT_DIR`
- **Parameters:** `NUM_SELECTED`, `FL_DELTA`, `USE_DELTA`
- **Concept:** Holistically minimizes total affinity to unselected frames

### 5. Facility Location Conditional Gain (FLCG)
- **Purpose:** Select frames that complement an existing set (incremental selection)
- **Method:** Maximizes coverage gain given a private set of already selected frames
- **Delta:** Optional - adds neighboring frames around selected ones for temporal continuity
- **Best For:** Incremental selection, query-focused summarization, conditional diversity
- **Output:** `FLCG_OUTPUT_DIR`
- **Parameters:** `NUM_SELECTED`, `FL_DELTA`, `USE_DELTA`, `FLCG_PRIVATE_SET`
- **Concept:** Finds frames that provide maximum additional coverage beyond the private set
- **Private Set:** Indices of frames already selected (e.g., `[0, 10, 20]`). Empty list means standard FL behavior

### 6. Custom Combinations
You can chain multiple algorithms or use them independently for comparison.

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

### DisparityMinSelector

```python
from functions import DisparityMinSelector

selector = DisparityMinSelector(budget=64, delta=[-1, 0, +1])
selected_indices = selector.select(feature_matrix)

# Clear submodlib memoization
selector.clear_memoization()
```

### DisparitySumSelector

```python
from functions import DisparitySumSelector

selector = DisparitySumSelector(budget=64, delta=[-1, 0, +1])
selected_indices = selector.select(feature_matrix)

# Clear submodlib memoization
selector.clear_memoization()
```

### FacilityLocationConditionalGainSelector

```python
from functions import FacilityLocationConditionalGainSelector

# Without private set (behaves like standard FL)
selector = FacilityLocationConditionalGainSelector(budget=64, delta=[-1, 0, +1])
selected_indices = selector.select(feature_matrix)

# With private set (incremental selection)
private_set = [0, 10, 20, 30]  # Already selected frames
selector = FacilityLocationConditionalGainSelector(budget=64, delta=[-1, 0, +1])
selected_indices = selector.select(feature_matrix, private_set=private_set)

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

# Merge with Disparity Min
merger.merge_with_disparity_min(frames_dir, output_dir)

# Merge with Disparity Sum
merger.merge_with_disparity_sum(frames_dir, output_dir)

# Merge with FL Conditional Gain
merger.merge_with_fl_conditional_gain(frames_dir, output_dir, private_set=[0, 10, 20])
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

### Disparity Min Output
```
Selected_frames64(DisparityMin)/
├── video1_selected.mp4
├── video2_selected.mp4
...
```

### Disparity Sum Output
```
Selected_frames64(DisparitySum)/
├── video1_selected.mp4
├── video2_selected.mp4
...
```

### FL Conditional Gain Output
```
Selected_frames64(FLConditionalGain)/
├── video1_selected.mp4
├── video2_selected.mp4
...
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

### Compare All Algorithms

```bash
# Extract frames once
python main.py --step 1

# Run all selection algorithms
python main.py --step 2  # Facility Location
python main.py --step 3  # Graph Cut
python main.py --step 4  # Disparity Min
python main.py --step 5  # Disparity Sum

# Compare outputs in different directories
```

### Algorithm Comparison Matrix

| Algorithm | Optimization Target | Best For | Output Control |
|-----------|-------------------|----------|-----------------|
| Facility Location | Coverage maximization | Key frames, summarization | Budget + Delta |
| Graph Cut | Diversity trade-off | Flexible selection | Budget + Lambda |
| Disparity Min | Max diversity (minimize max sim) | Outlier detection | Budget + Delta |
| Disparity Sum | Balanced diversity (minimize sum) | Representative sampling | Budget + Delta |

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

### v1.2.0
- Added Facility Location Conditional Gain algorithm
- Support for private set (conditioning frames)
- Incremental selection capability
- 6 total selection algorithms now available

### v1.1.0
- Added DisparityMin selection algorithm
- Added DisparitySum selection algorithm
- Support for 5 different selection algorithms
- Algorithm comparison matrix
- Updated documentation with new algorithms

### v1.0.0
- Initial release
- Frame extraction, CLIP embeddings, Facility Location, Graph Cut
- Delta expansion for temporal continuity
- Embeddings database management

---

**Last Updated:** November 2024  
**Status:** Production Ready