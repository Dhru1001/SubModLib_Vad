
# ============================================================================
# functions/facility_location.py
# ============================================================================
import numpy as np
from submodlib.functions.facilityLocation import FacilityLocationFunction

class FacilityLocationSelector:
    """Frame selection using Facility Location submodular function with memoization."""
    
    def __init__(self, budget=64, metric="cosine"):
        self.budget = budget
        self.metric = metric
        self.fb_function = None  # Store function instance for memoization control
    
    def select(self, feature_matrix):
        """
        Select frames using Facility Location.
        
        Args:
            feature_matrix: np.ndarray of shape (n_frames, feature_dim)
            
        Returns:
            selected_indices: sorted list of selected frame indices
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