# ============================================================================
# main.py
# ============================================================================
import torch
import argparse
from config import Config
from frame_extraction import FrameExtractor
from merge_video import VideoMerger
from embeddings import EmbeddingsExtractor

def main(args):
    """Main orchestration script."""
    
    print("=" * 70)
    print("VIDEO FRAME EXTRACTION & SELECTION PIPELINE")
    print("=" * 70)
    
    # Initialize embeddings extractor
    embeddings_extractor = EmbeddingsExtractor()
    
    # Step 1: Extract frames from videos
    if args.step in ["1", "all"]:
        print("\n[STEP 1] Extracting frames from videos...")
        print("-" * 70)
        
        extractor = FrameExtractor(frame_interval=Config.FRAME_INTERVAL)
        extractor.extract_all_videos(Config.VIDEOS_DIR, Config.FRAMES_OUTPUT_DIR)
    
    # Step 2: Merge with Facility Location
    if args.step in ["2", "all"]:
        print("\n[STEP 2] Merging frames using Facility Location...")
        print("-" * 70)
        
        merger_fl = VideoMerger(
            num_selected=Config.NUM_SELECTED,
            fps=Config.OUTPUT_FPS,
            overwrite=Config.OVERWRITE
        )
        merger_fl.merge_with_facility_location(Config.FRAMES_OUTPUT_DIR)
    
    # Step 3: Merge with Graph Cut
    if args.step in ["3", "all"]:
        print("\n[STEP 3] Merging frames using Graph Cut...")
        print("-" * 70)
        
        merger_gc = VideoMerger(
            num_selected=Config.NUM_SELECTED,
            fps=Config.OUTPUT_FPS,
            overwrite=Config.OVERWRITE
        )
        merger_gc.merge_with_graph_cut(Config.FRAMES_OUTPUT_DIR)
    
    # Step 4: Merge with Disparity Min
    if args.step in ["4", "all"]:
        print("\n[STEP 4] Merging frames using Disparity Min...")
        print("-" * 70)
        
        merger_dm = VideoMerger(
            num_selected=Config.NUM_SELECTED,
            fps=Config.OUTPUT_FPS,
            overwrite=Config.OVERWRITE
        )
        merger_dm.merge_with_disparity_min(Config.FRAMES_OUTPUT_DIR)
    
    # Step 5: Merge with Disparity Sum
    if args.step in ["5", "all"]:
        print("\n[STEP 5] Merging frames using Disparity Sum...")
        print("-" * 70)
        
        merger_ds = VideoMerger(
            num_selected=Config.NUM_SELECTED,
            fps=Config.OUTPUT_FPS,
            overwrite=Config.OVERWRITE
        )
        merger_ds.merge_with_disparity_sum(Config.FRAMES_OUTPUT_DIR)
    
    # Step 6: Merge with FL Conditional Gain
    if args.step in ["6", "all"]:
        print("\n[STEP 6] Merging frames using FL Conditional Gain...")
        print("-" * 70)
        
        merger_flcg = VideoMerger(
            num_selected=Config.NUM_SELECTED,
            fps=Config.OUTPUT_FPS,
            overwrite=Config.OVERWRITE
        )
        merger_flcg.merge_with_fl_conditional_gain(Config.FRAMES_OUTPUT_DIR)
    
    # Show embeddings database stats
    if args.show_embeddings:
        embeddings_extractor.print_database_stats()
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video frame extraction and selection pipeline with embeddings storage"
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["1", "2", "3", "4", "5", "6", "all"],
        help="Which pipeline step to run (1=extract, 2=FL, 3=GC, 4=DM, 5=DS, 6=FLCG, all=all steps)"
    )
    parser.add_argument(
        "--show-embeddings",
        action="store_true",
        help="Display embeddings database statistics"
    )
    
    args = parser.parse_args()
    main(args)