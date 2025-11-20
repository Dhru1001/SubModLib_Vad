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
        choices=["1", "2", "3", "all"],
        help="Which pipeline step to run (1=extract, 2=FL, 3=GC, all=all steps)"
    )
    parser.add_argument(
        "--show-embeddings",
        action="store_true",
        help="Display embeddings database statistics"
    )
    
    args = parser.parse_args()
    main(args)