# ============================================================================
# functions/facility_location_conditional_gain.py
# ============================================================================
from submodlib.functions.facilityLocationConditionalGain import FacilityLocationConditionalGainFunction
from config import Config

class FacilityLocationConditionalGainSelector:
    """Frame selection using Facility Location Conditional Gain submodular function.
    
    Facility Location Conditional Gain selects frames that provide maximum coverage
    gain given a set of already selected frames (private set). This is useful for
    incremental selection where you want to find frames that complement an existing
    selection.
    
    Use Case: Incremental frame selection, query-focused summarization, 
              conditional diversity
    """
    
    def __init__(self, budget=64, metric="cosine", delta=None, use_delta=None):
        self.budget = budget
        self.metric = metric
        self.use_delta = use_delta if use_delta is not None else Config.USE_DELTA
        self.delta = delta if delta is not None else Config.FL_DELTA
        self.flcg_function = None  # Store function instance for memoization control
    
    def _apply_delta(self, selected_indices, total_frames):
        """
        Apply delta offsets to selected indices and remove duplicates.
        
        Delta works by expanding each selected frame with a range.
        
        Example:
            If delta = [-2, 0, +2] and selected_indices = [102]
            Then: Range from 102-2 to 102+2 = [100, 101, 102, 103, 104]
        
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
    
    def select(self, feature_matrix, private_set=None):
        """
        Select frames using Facility Location Conditional Gain with optional delta expansion.
        
        Args:
            feature_matrix: np.ndarray of shape (n_frames, feature_dim)
            private_set: list of indices already selected (optional)
                        If None, an empty set is used
            
        Returns:
            selected_indices: sorted list of selected frame indices
        """
        # Initialize private set
        if private_set is None:
            private_set = []
        
        try:
            self.flcg_function = FacilityLocationConditionalGainFunction(
                n=len(feature_matrix),
                mode="dense",
                data=feature_matrix,
                metric=self.metric,
                privacyHardness=0  # No privacy constraint
            )
            
            # Set the private set (conditioning set)
            if private_set:
                print(f"[INFO] FLCG using private set: {private_set}")
                # Note: submodlib expects private set to be set via the function
                # The maximize function will consider this private set
            
            selected = self.flcg_function.maximize(
                budget=self.budget, 
                optimizer="NaiveGreedy",
                privacySet=private_set  # Provide private set here
            )
            selected_indices = sorted([t[0] for t in selected])
            
            print(f"[INFO] FLCG selected frames: {selected_indices}")
            
            # Apply delta expansion only if enabled
            if self.use_delta and self.delta and len(self.delta) > 0:
                expanded_indices = self._apply_delta(selected_indices, len(feature_matrix))
                print(f"[INFO] Delta enabled: {self.delta}")
                print(f"[INFO] After delta expansion: {expanded_indices}")
                print(f"[INFO] Total frames after delta: {len(expanded_indices)}")
                return expanded_indices
            else:
                print(f"[INFO] Delta disabled - returning selected frames as-is")
                return selected_indices
        
        except Exception as e:
            print(f"[ERROR] FacilityLocationConditionalGain selection failed: {e}")
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
        if self.flcg_function is not None:
            try:
                self.flcg_function.clearMemoization()
                print("[INFO] FacilityLocationConditionalGain memoization cleared.")
            except Exception as e:
                print(f"[WARN] Failed to clear memoization: {e}")
        else:
            print("[WARN] No FacilityLocationConditionalGain function to clear.")