# ============================================================================
# main.py
# ============================================================================
import torch
import argparse
from config import Config
from frame_extraction import FrameExtractor
from merge_video import VideoMerger

def main(args):
    """Main orchestration script."""
    
    print("=" * 70)
    print("VIDEO FRAME EXTRACTION & SELECTION PIPELINE")
    print("=" * 70)
    
    if args.step in ["1", "all"]:
        print("\n[STEP 1] Extracting frames from videos...")
        print("-" * 70)
        
        extractor = FrameExtractor(frame_interval=Config.FRAME_INTERVAL)
        extractor.extract_all_videos(Config.VIDEOS_DIR, Config.FRAMES_OUTPUT_DIR)
    
    if args.step in ["2", "all"]:
        print("\n[STEP 2] Merging frames using Facility Location...")
        print("-" * 70)
        
        merger_fl = VideoMerger(
            num_selected=Config.NUM_SELECTED,
            fps=Config.OUTPUT_FPS,
            overwrite=Config.OVERWRITE
        )
        output_fl = Config.FRAMES_OUTPUT_DIR.replace("Frames", "Selected_frames64(usingCLIP)")
        merger_fl.merge_with_facility_location(Config.FRAMES_OUTPUT_DIR, output_fl)
    
    if args.step in ["3", "all"]:
        print("\n[STEP 3] Merging frames using Graph Cut...")
        print("-" * 70)
        
        merger_gc = VideoMerger(
            num_selected=Config.NUM_SELECTED,
            fps=Config.OUTPUT_FPS,
            overwrite=Config.OVERWRITE
        )
        output_gc = Config.FRAMES_OUTPUT_DIR.replace("Frames", "Selected_frames64(GraphCut_Negative_Values)")
        merger_gc.merge_with_graph_cut(
            Config.FRAMES_OUTPUT_DIR,
            output_gc,
            lambda_values=Config.LAMBDA_VALUES
        )
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video frame extraction and selection pipeline with memoization"
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["1", "2", "3", "all"],
        help="Which pipeline step to run (1=extract, 2=FL, 3=GC, all=all steps)"
    )
    
    args = parser.parse_args()
    main(args)