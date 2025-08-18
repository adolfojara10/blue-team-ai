from pathlib import Path
import joblib 
import xgboost as xgb
import cupy as cp

class EnsembleModel:
    """Individual model wrapper for the ensemble"""

    def __init__(self, config, logger):
        self.config = config
        self.classifier_lightgbm = None
        self.classifier_xgboost = None
        self.logger = logger
        self._load_classifiers()
    
    def _load_classifiers(self):
        """Load the trained models"""
        try:
            # Load LightGBM
            model_path_lgbm = Path(self.config.model_lightgbm_file)
            if not model_path_lgbm.exists():
                raise FileNotFoundError(f"Model file not found: {self.config.model_lightgbm_file}")
            self.classifier_lightgbm = joblib.load(model_path_lgbm)
            self.logger.info(f"✅ Loaded LightGBM classifier for {self.config.name}")
        except Exception as e:
            self.logger.error(f"❌ Failed to load LightGBM classifier for {self.config.name}: {e}")
            raise

        try:
            # Load XGBoost
            model_path_xgb = Path(self.config.model_xgboost_file)
            if not model_path_xgb.exists():
                raise FileNotFoundError(f"Model file not found: {self.config.model_xgboost_file}")
            self.classifier_xgboost = joblib.load(model_path_xgb)
            self.logger.info(f"✅ Loaded XGBoost classifier for {self.config.name}")
        except Exception as e:
            self.logger.error(f"❌ Failed to load XGBoost classifier for {self.config.name}: {e}")
            raise

    def predict_proba_lightgbm(self, embedding):
        """Get prediction probabilities from the LightGBM classifier"""
        if self.classifier_lightgbm is None:
            raise ValueError(f"LightGBM classifier not loaded for {self.config.name}")
        
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        raw_probs = self.classifier_lightgbm.predict_proba(embedding)[0]
        weighted_probs = self.config.weight * raw_probs
        return raw_probs, weighted_probs

    def predict_proba_xgboost(self, embedding):
        """Get prediction probabilities from the XGBoost classifier"""
        if self.classifier_xgboost is None:
            raise ValueError(f"XGBoost classifier not loaded for {self.config.name}")
        
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # --- NEW: Move embedding to GPU if GPU is available ---
        try:
            
            # Convert numpy array to cupy array (moves data to GPU)
            embedding_gpu = cp.asarray(embedding)
            # Pass the GPU-resident array to predict_proba
            raw_probs_gpu = self.classifier_xgboost.predict_proba(embedding_gpu)
            # Convert probabilities back to numpy if needed for further CPU processing
            raw_probs = raw_probs_gpu.get()[0] # .get() moves data from GPU to CPU
        

            weighted_probs = self.config.weight * raw_probs
            return raw_probs, weighted_probs
        
        except Exception as e:
            self.logger.error(f"❌ Error during XGBoost GPU prediction, proceding in CPU")

            raw_probs = self.classifier_xgboost.predict_proba(embedding)[0]
            weighted_probs = self.config.weight * raw_probs
            return raw_probs, weighted_probs
    
    def _predict_with_classifier(self, clf, embedding):
        """Helper to make prediction with a given classifier"""
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        raw_probs = clf.predict_proba(embedding)[0]
        weighted_probs = self.config.weight * raw_probs
        return raw_probs, weighted_probs




# 
# 
# from pathlib import Path
# import joblib 

# class EnsembleModel:
#     """Individual model wrapper for the ensemble"""
    
#     def __init__(self, config, logger):
#         self.config = config
#         self.classifier_lightgbm = None
#         self.classifier_xgboost = None
#         self.logger = logger
#         self._load_classifier()
        
    
#     def _load_classifier(self):
#         """Load the trained model"""
#         try:
#             model_path = Path(self.config.model_lightgbm_file)
#             if not model_path.exists():
#                 raise FileNotFoundError(f"Model file not found: {self.config.model_lightgbm_file}")
            
#             self.classifier_lightgbm = joblib.load(self.config.model_lightgbm_file)
#             self.logger.info(f"✅ Loaded classifier_lightgbm for {self.config.name}")
#         except Exception as e:
#             self.logger.error(f"❌ Failed to load classifier_lightgbm for {self.config.name}: {e}")
#             raise
    
#     def predict_proba(self, embedding):
#         """Get prediction probabilities from the classifier"""
#         if self.classifier_lightgbm is None:
#             raise ValueError(f"classifier_lightgbm not loaded for {self.config.name}")
        
#         # Ensure 2D shape
#         if embedding.ndim == 1:
#             embedding = embedding.reshape(1, -1)
        
#         # Get raw probabilities
#         raw_probs = self.classifier_lightgbm.predict_proba(embedding)[0]
        
#         # Apply weight
#         weighted_probs = self.config.weight * raw_probs
        
#         return raw_probs, weighted_probs