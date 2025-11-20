
# ============================================================================
# functions/graph_cut.py
# ============================================================================
import numpy as np
from submodlib.functions.graphCut import GraphCutFunction

class GraphCutSelector:
    """Frame selection using Graph Cut submodular function with memoization."""
    
    def __init__(self, budget=64, metric="cosine"):
        self.budget = budget
        self.metric = metric
        self.gc_function = None  # Store function instance for memoization control
    
    def select(self, feature_matrix, lambda_val=-0.5):
        """
        Select frames using Graph Cut.
        
        Args:
            feature_matrix: np.ndarray of shape (n_frames, feature_dim)
            lambda_val: trade-off parameter
            
        Returns:
            selected_indices: sorted list of selected frame indices
        """
        try:
            self.gc_function = GraphCutFunction(
                n=len(feature_matrix),
                mode="dense",
                lambdaVal=lambda_val,
                data=feature_matrix,
                metric=self.metric
            )
            selected = self.gc_function.maximize(self.budget, optimizer="NaiveGreedy")
            selected_indices = sorted([t[0] for t in selected])
            return selected_indices
        
        except Exception as e:
            print(f"[ERROR] GraphCut selection failed: {e}")
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
        if self.gc_function is not None:
            try:
                self.gc_function.clearMemoization()
                print("[INFO] GraphCut memoization cleared.")
            except Exception as e:
                print(f"[WARN] Failed to clear memoization: {e}")
        else:
            print("[WARN] No GraphCut function to clear.")
