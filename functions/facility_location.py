# ============================================================================
# functions/facility_location.py
# ============================================================================
from submodlib.functions.facilityLocation import FacilityLocationFunction
from config import Config

class FacilityLocationSelector:
    """Frame selection using Facility Location submodular function with memoization."""
    
    def __init__(self, budget=64, metric="cosine", delta=None):
        self.budget = budget
        self.metric = metric
        self.delta = delta if delta is not None else Config.FL_DELTA
        self.fb_function = None  # Store function instance for memoization control
    
    def _apply_delta(self, selected_indices, total_frames):
        """
        Apply delta offsets to selected indices and remove duplicates.
        
        Delta works by expanding each selected frame with all offsets.
        
        Example:
            If delta = [-2, 0, +2] and selected_indices = [102]
            Then: 102 + (-2) = 100
                  102 + 0 = 102
                  102 + (+2) = 104
            Result: [100, 101, 102, 103, 104] (sorted and unique)
        
        Args:
            selected_indices: sorted list of selected frame indices
            total_frames: total number of frames available
            
        Returns:
            expanded_indices: sorted list of unique frame indices with delta applied
        """
        expanded_indices = set()
        
        # Get min and max delta for range expansion
        min_delta = min(self.delta)
        max_delta = max(self.delta)
        
        for idx in selected_indices:
            # Create range from (idx + min_delta) to (idx + max_delta)
            start = idx + min_delta
            end = idx + max_delta
            
            # Add all frames in the range
            for frame_idx in range(start, end + 1):
                # Ensure index is within bounds
                if 0 <= frame_idx < total_frames:
                    expanded_indices.add(frame_idx)
        
        # Sort and return as list
        return sorted(list(expanded_indices))
    
    def select(self, feature_matrix):
        """
        Select frames using Facility Location with delta expansion.
        
        Args:
            feature_matrix: np.ndarray of shape (n_frames, feature_dim)
            
        Returns:
            selected_indices: sorted list of selected frame indices with delta applied
        """
        try:
            self.fb_function = FacilityLocationFunction(
                n=len(feature_matrix),
                mode="dense",
                data=feature_matrix,
                metric=self.metric
            )
            selected = self.fb_function.maximize(budget=self.budget, optimizer="NaiveGreedy")
            selected_indices = sorted([t[0] for t in selected])
            
            print(f"[INFO] FL selected frames: {selected_indices}")
            
            # Apply delta expansion
            if self.delta and len(self.delta) > 0:
                expanded_indices = self._apply_delta(selected_indices, len(feature_matrix))
                print(f"[INFO] After delta {self.delta} expansion: {expanded_indices}")
                print(f"[INFO] Total frames after delta: {len(expanded_indices)}")
                return expanded_indices
            
            return selected_indices
        
        except Exception as e:
            print(f"[ERROR] FacilityLocation selection failed: {e}")
            return self._fallback_selection(len(feature_matrix))
    
    def _fallback_selection(self, total_frames):
        """Evenly spaced fallback selection."""
        selected_indices = sorted(list({
            int(round(i * (total_frames - 1) / max(1, self.budget - 1)))
            for i in range(self.budget)
        }))
        return [i for i in selected_indices if 0 <= i < total_frames]
    
    def clear_memoization(self):
        """Clear the computed memoized statistics from submodlib."""
        if self.fb_function is not None:
            try:
                self.fb_function.clearMemoization()
                print("[INFO] FacilityLocation memoization cleared.")
            except Exception as e:
                print(f"[WARN] Failed to clear memoization: {e}")
        else:
            print("[WARN] No FacilityLocation function to clear.")