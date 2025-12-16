# ============================================================================
# embeddings.py - Separate embeddings extraction
# ============================================================================
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import json
import clip
from config import Config

class EmbeddingsExtractor:
    """Extract and manage features (CLIP embeddings or numpy features) with vector database storage."""
    
    def __init__(self, feature_type=None):
        self.feature_type = feature_type if feature_type is not None else Config.FEATURE_TYPE
        
        # Setup device
        self.device = Config.DEVICE if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device}")
        print(f"[INFO] Feature type: {self.feature_type}")
        
        if self.feature_type == "clip":
            # Load CLIP model
            self._load_clip_model()
            
            # Preprocessing for CLIP
            self.preprocess_fn = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
        elif self.feature_type == "numpy":
            print("[INFO] Using raw numpy features (no CLIP model needed)")
            # Simple resize for numpy features
            self.target_size = (224, 224)
        else:
            raise ValueError(f"Invalid feature_type: {self.feature_type}. Must be 'clip' or 'numpy'")
        
        # Create embeddings database directory
        os.makedirs(Config.EMBEDDINGS_DB_DIR, exist_ok=True)
    
    def _load_clip_model(self):
        """Load CLIP model."""
        model, _ = clip.load(Config.CLIP_MODEL, device=self.device, jit=False)
        self.image_encoder = model.visual
        self.image_encoder.eval()
    
    def _get_embedding_file_path(self, video_name):
        """Get the path for storing embeddings of a video."""
        return os.path.join(Config.EMBEDDINGS_DB_DIR, f"{video_name}_{self.feature_type}_embeddings.npz")
    
    def _get_metadata_file_path(self, video_name):
        """Get the path for storing metadata of a video."""
        return os.path.join(Config.EMBEDDINGS_DB_DIR, f"{video_name}_{self.feature_type}_metadata.json")
    
    def extract_and_store(self, frame_images, video_name, force_recompute=False):
        """
        Extract embeddings from frames and store in vector database.
        
        Args:
            frame_images: list of PIL Image objects
            video_name: name of the video/subfolder
            force_recompute: if True, recompute even if exists
            
        Returns:
            feature_matrix: np.ndarray of shape (n_frames, feature_dim)
        """
        embedding_path = self._get_embedding_file_path(video_name)
        metadata_path = self._get_metadata_file_path(video_name)
        
        # Check if embeddings already exist
        if os.path.exists(embedding_path) and not force_recompute:
            print(f"[INFO] Loading {self.feature_type} embeddings from database: {video_name}")
            data = np.load(embedding_path)
            feature_matrix = data['embeddings']
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"[INFO] Loaded {len(feature_matrix)} embeddings for {video_name}")
            return feature_matrix
        
        # Extract embeddings
        print(f"[INFO] Computing {self.feature_type} embeddings for {video_name}...")
        feature_matrix = self._extract_features(frame_images)
        
        # Store embeddings in database
        self._store_embeddings(embedding_path, feature_matrix, video_name, len(frame_images))
        
        return feature_matrix
    
    def _extract_features(self, frame_images):
        """Extract features for all frames based on feature type."""
        if self.feature_type == "clip":
            return self._extract_clip_features(frame_images)
        elif self.feature_type == "numpy":
            return self._extract_numpy_features(frame_images)
    
    def _extract_clip_features(self, frame_images):
        """Extract CLIP features for all frames."""
        feature_matrix = []
        
        for idx, image in enumerate(frame_images):
            img_tensor = self.preprocess_fn(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if self.device == "cuda":
                    features = self.image_encoder(img_tensor.half()).cpu().numpy().squeeze()
                else:
                    features = self.image_encoder(img_tensor).cpu().numpy().squeeze()
            
            feature_matrix.append(features)
            
            if (idx + 1) % 10 == 0:
                print(f"  [{idx + 1}/{len(frame_images)}] frames processed")
        
        return np.array(feature_matrix)
    
    def _extract_numpy_features(self, frame_images):
        """Extract numpy (raw pixel) features for all frames."""
        feature_matrix = []
        
        for idx, image in enumerate(frame_images):
            # Resize image
            img_resized = image.resize(self.target_size, Image.BILINEAR)
            
            # Convert to numpy array and flatten
            img_array = np.array(img_resized).astype(np.float32) / 255.0  # Normalize to [0, 1]
            img_flat = img_array.flatten()  # Shape: (224*224*3,) = (150528,)
            
            feature_matrix.append(img_flat)
            
            if (idx + 1) % 10 == 0:
                print(f"  [{idx + 1}/{len(frame_images)}] frames processed")
        
        return np.array(feature_matrix)
    
    def _store_embeddings(self, embedding_path, feature_matrix, video_name, frame_count):
        """Store embeddings in vector database as npz file."""
        np.savez_compressed(
            embedding_path,
            embeddings=feature_matrix,
            dtype=str(feature_matrix.dtype),
            shape=feature_matrix.shape
        )
        
        # Store metadata
        metadata = {
            "video_name": video_name,
            "frame_count": frame_count,
            "embedding_dim": feature_matrix.shape[1],
            "feature_type": self.feature_type,
            "model": Config.CLIP_MODEL if self.feature_type == "clip" else "numpy",
            "total_frames": len(feature_matrix)
        }
        
        metadata_path = self._get_metadata_file_path(video_name)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[SUCCESS] Stored {self.feature_type} embeddings for {video_name}: {embedding_path}")
    
    def load_embeddings(self, video_name):
        """Load pre-computed embeddings from database."""
        embedding_path = self._get_embedding_file_path(video_name)
        
        if not os.path.exists(embedding_path):
            print(f"[ERROR] {self.feature_type} embeddings not found for {video_name}")
            return None
        
        data = np.load(embedding_path)
        return data['embeddings']
    
    def load_embeddings_from_folder(self, folder_path):
        """
        Load embeddings for all frames in a folder.
        Used for FLCG private data.
        
        Args:
            folder_path: path to folder containing frames
            
        Returns:
            feature_matrix: np.ndarray of features for all frames in folder
        """
        if not os.path.exists(folder_path):
            print(f"[ERROR] Folder not found: {folder_path}")
            return None
        
        # Load all frames from folder
        frame_files = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(Config.IMAGE_EXTENSIONS)
        ])
        
        if not frame_files:
            print(f"[WARN] No frames found in {folder_path}")
            return None
        
        frame_images = [Image.open(f).convert("RGB") for f in frame_files]
        
        # Extract features
        print(f"[INFO] Extracting {self.feature_type} features from {len(frame_images)} frames in {folder_path}")
        feature_matrix = self._extract_features(frame_images)
        
        return feature_matrix
    
    def list_stored_embeddings(self):
        """List all stored embeddings in the database."""
        embeddings_list = []
        
        for file in os.listdir(Config.EMBEDDINGS_DB_DIR):
            if file.endswith('_metadata.json'):
                metadata_path = os.path.join(Config.EMBEDDINGS_DB_DIR, file)
                
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                embeddings_list.append({
                    "video_name": metadata.get("video_name"),
                    "frame_count": metadata.get("frame_count"),
                    "embedding_dim": metadata.get("embedding_dim"),
                    "feature_type": metadata.get("feature_type", "unknown")
                })
        
        return embeddings_list
    
    def print_database_stats(self):
        """Print statistics about stored embeddings."""
        embeddings_list = self.list_stored_embeddings()
        
        if not embeddings_list:
            print("[INFO] No embeddings in database.")
            return
        
        print("\n" + "=" * 70)
        print("EMBEDDINGS DATABASE STATISTICS")
        print("=" * 70)
        
        for item in embeddings_list:
            print(f"\nVideo: {item['video_name']}")
            print(f"  Frames: {item['frame_count']}")
            print(f"  Embedding Dimension: {item['embedding_dim']}")
            print(f"  Feature Type: {item['feature_type']}")
        
        total_videos = len(embeddings_list)
        print(f"\nTotal Videos: {total_videos}")
        print("=" * 70 + "\n")