import numpy as np
import pandas as pd

class HybridFireDetector:
    def __init__(self, isolation_forest_model=None, temp_threshold=55.0, co_threshold=80.0, feature_names=None):
        """
        Hybrid Fire Detector combining an Isolation Forest model and rule-based thresholds.
        
        Parameters:
        -----------
        isolation_forest_model : sklearn.ensemble.IsolationForest (optional)
            The trained Isolation Forest model. If None, it works purely as a rule-based fallback.
        temp_threshold : float
            Temperature threshold for rule-based fire detection (default: 55.0).
        co_threshold : float
            Carbon Monoxide (CO) concentration threshold (default: 80.0).
        feature_names : list of str
            Feature names in order. Defaults to ['온도', '습도', '일산화농도'].
        """
        self.model = isolation_forest_model
        self.temp_threshold = temp_threshold
        self.co_threshold = co_threshold
        self.feature_names = feature_names or ['온도', '습도', '일산화농도']

    def predict(self, X):
        """
        Predict whether samples are normal (1) or anomalies/fire (-1).
        Supports both pandas DataFrames and numpy arrays.
        """
        preds = None
        
        # Check type of X to safely apply threshold rules
        if isinstance(X, pd.DataFrame):
            # Try to map columns by name, fallback to positions if names aren't present
            temp_col = '온도' if '온도' in X.columns else (self.feature_names[0] if self.feature_names[0] in X.columns else X.columns[0])
            co_col = '일산화농도' if '일산화농도' in X.columns else (self.feature_names[2] if self.feature_names[2] in X.columns else X.columns[-1])
            
            temp_values = X[temp_col]
            co_values = X[co_col]
            num_samples = len(X)
        else:
            # For NumPy array or lists, map by index
            X_arr = np.asarray(X)
            
            # Map indices based on feature names
            try:
                temp_idx = self.feature_names.index('온도')
            except ValueError:
                temp_idx = 0
                
            try:
                co_idx = self.feature_names.index('일산화농도')
            except ValueError:
                co_idx = min(2, X_arr.shape[1] - 1)
                
            temp_values = X_arr[:, temp_idx]
            co_values = X_arr[:, co_idx]
            num_samples = len(X_arr)

        # Run Isolation Forest prediction if model is loaded
        if self.model is not None:
            try:
                preds = self.model.predict(X)
            except Exception as e:
                print(f"[WARNING] Isolation Forest prediction failed, using only rule-based: {e}")
                preds = np.ones(num_samples)
        else:
            # If no model, default to all normal (1), and let rules override
            preds = np.ones(num_samples)
            
        # Apply rule-based overrides (fire if temp > threshold OR co > threshold)
        rule_based_fire = (temp_values > self.temp_threshold) | (co_values > self.co_threshold)
        
        # Convert to boolean numpy array for indexing
        rule_based_fire = np.asarray(rule_based_fire, dtype=bool)
        
        # Force rule-based detections to be anomalies (-1)
        preds[rule_based_fire] = -1
        return preds
