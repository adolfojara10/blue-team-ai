# File: /final-filter/final-filter/src/ensemble.py
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any, Optional
from model_wrapper import EnsembleModel
from model_config import ModelConfig 
from embedding_client import OllamaEmbeddingClient
from resource_monitor import ResourceMonitor, ResourceStats
from pathlib import Path
import time
from datetime import datetime
import sys
import os

# Adjust path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class ParallelEnsembleClassifier:
    """Main ensemble classifier with parallel processing capabilities and resource monitoring"""
    
    def __init__(self, logger, llama3, model_configs: List[ModelConfig], 
                 max_workers: int = 4, csv_logger=None, enable_resource_monitoring: bool = True):
        self.model_configs = model_configs
        self.models = []
        self.max_workers = max_workers
        self.llama3 = llama3  # Llama3Classifier instance
        self.embedding_client = OllamaEmbeddingClient(logger=logger)
        self.logger = logger
        self.csv_logger = csv_logger
        self.enable_resource_monitoring = enable_resource_monitoring
        
        # Initialize resource monitor
        if enable_resource_monitoring:
            self.resource_monitor = ResourceMonitor(
                sample_interval=0.05,  # 50ms sampling
                logger=logger
            )
        else:
            self.resource_monitor = None
        
        # Load all models
        self._load_models()
    
    def _load_models(self):
        """Load all models in parallel"""
        self.logger.info("🔄 Loading ensemble models...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_config = {
                executor.submit(self._load_single_model, config): config 
                for config in self.model_configs
            }
            
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    model = future.result()
                    self.models.append(model)
                    self.logger.info(f"✅ Model {config.name} loaded successfully")
                except Exception as e:
                    import traceback
                    tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                    self.logger.error(f"❌ Failed to load model {config.name}: {e}\nTraceback:\n{tb_str}")
    
    def _load_single_model(self, config: ModelConfig) -> EnsembleModel:
        """Load a single model (used by parallel loader)"""
        return EnsembleModel(config, logger=self.logger)
    
    def _get_embedding_single(self, prompt: str, model_name: str) -> Tuple[str, np.ndarray]:
        """Get embedding for a single model (synchronous)"""
        try:
            embedding = self.embedding_client.get_embedding(model_name, prompt)
            return model_name, embedding
        except Exception as e:
            self.logger.error(f"Failed to get embedding from {model_name}: {e}")
            return model_name, None
    
    async def _get_embedding_parallel(self, prompt: str, model_name: str) -> Tuple[str, np.ndarray]:
        """Get embedding for a single model (async wrapper)"""
        try:
            embedding = await self.embedding_client.get_embedding_async(model_name, prompt)
            return model_name, embedding
        except Exception as e:
            self.logger.error(f"Failed to get embedding from {model_name}: {e}")
            return model_name, None
    
    async def _get_all_embeddings_async(self, prompt: str) -> Dict[str, np.ndarray]:
        """Get embeddings from all models in parallel (async version)"""
        tasks = [
            self._get_embedding_parallel(prompt, model.config.embedding_model)
            for model in self.models
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        embeddings = {}
        
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Exception in embedding task: {result}")
                continue
            
            model_name, embedding = result
            if embedding is not None:
                embeddings[model_name] = embedding
        
        return embeddings
    
    def _get_all_embeddings_sync(self, prompt: str) -> Dict[str, np.ndarray]:
        """Get embeddings from all models in parallel (synchronous version)"""
        embeddings = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_model = {
                executor.submit(self._get_embedding_single, prompt, model.config.embedding_model): model.config.embedding_model
                for model in self.models
            }
            
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result_model_name, embedding = future.result()
                    if embedding is not None:
                        embeddings[result_model_name] = embedding
                        self.logger.debug(f"✅ Got embedding from {result_model_name}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to get embedding from {model_name}: {e}")
        
        return embeddings
    
    def _predict_single_model(self, model: EnsembleModel, embedding: np.ndarray, classifier_name: str) -> Tuple[str, Dict[str, Any]]:
        """Make prediction with a single classifier of a model"""
        try:
            if 'lightgbm' in classifier_name:
                raw_probs, weighted_probs = model.predict_proba_lightgbm(embedding)
            elif 'xgboost' in classifier_name:
                raw_probs, weighted_probs = model.predict_proba_xgboost(embedding)

            result = {
                'model_name': model.config.name,
                'classifier': classifier_name,
                'raw_probabilities': raw_probs.tolist(),
                'weighted_probabilities': weighted_probs.tolist(),
                'weight': model.config.weight,
                'embedding_model': model.config.embedding_model,
                'success': True
            }

            self.logger.info(f"✅ {model.config.name} ({classifier_name}) - Raw: {raw_probs}, Weighted: {weighted_probs}")
            return model.config.name, result

        except Exception as e:
            self.logger.error(f"❌ Prediction failed for {model.config.name} ({classifier_name}): {e}")
            result = {
                'model_name': model.config.name,
                'classifier': classifier_name,
                'error': str(e),
                'success': False
            }
        return model.config.name, result

    def _make_parallel_predictions(self, embeddings: Dict[str, np.ndarray]) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """Make predictions with all models (and their classifiers) in parallel"""
        valid_predictions = []
        all_model_results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_model_info = {}
            
            # Submit both classifiers from each model
            for model in self.models:
                if model.config.embedding_model in embeddings:
                    embedding = embeddings[model.config.embedding_model]

                    # LightGBM
                    if model.classifier_lightgbm is not None:
                        future = executor.submit(self._predict_single_model, model, embedding, 'lightgbm')
                        future_to_model_info[future] = (model, 'lightgbm')
                    
                    # XGBoost
                    if model.classifier_xgboost is not None:
                        future = executor.submit(self._predict_single_model, model, embedding, 'xgboost')
                        future_to_model_info[future] = (model, 'xgboost')

            # Collect results
            for future in as_completed(future_to_model_info):
                model, classifier_name = future_to_model_info[future]
                try:
                    model_name, result = future.result()
                    all_model_results.append(result)

                    if result['success']:
                        valid_predictions.append(np.array(result['weighted_probabilities']))
                        self.logger.debug(f"✅ Prediction completed for {model_name} ({classifier_name})")
                except Exception as e:
                    self.logger.error(f"❌ Prediction failed for {model.config.name} ({classifier_name}): {e}")
                    all_model_results.append({
                        'model_name': model.config.name,
                        'classifier': classifier_name,
                        'error': str(e),
                        'success': False
                    })

        return valid_predictions, all_model_results

    def _log_detailed_results(self, prompt: str, final_prediction: int, final_probabilities: np.ndarray, 
                             model_results: List[Dict[str, Any]], metadata: Dict[str, Any],
                             resource_stats: Optional[Dict[str, Any]] = None):
        """Log detailed results to both console and file"""
        
        # Log summary to console and file
        self.logger.info("=" * 80)
        self.logger.info("📋 PREDICTION SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"🎯 Prompt: {prompt}")
        label_map = {0: "Safe", 1: "Unsafe"}
        if metadata.get('llama3_prediction'):
            final_prediction_display = metadata['llama3_prediction']
            self.logger.info(f"🧠 Final Prediction (Llama Decision): {final_prediction_display}\n")
            self.logger.info(f"     Ensemble Information:")
        else:
            self.logger.info(f"🧠 Final Prediction: {label_map.get(final_prediction, 'Unknown')}")
        self.logger.info(f"🗳️ Vote counts: Safe = {metadata['vote_counts'][0]}, Unsafe = {metadata['vote_counts'][1]}")
        self.logger.info(f"📊 Combined Probabilities: {final_probabilities}")
        self.logger.info(f"📈 Confidence: {metadata['confidence']:.4f}")
        self.logger.info(f"📉 Entropy: {metadata['entropy']:.4f}")
        self.logger.info(f"⏱️  Total Time: {metadata['total_time']:.3f}s")
        self.logger.info(f"🔧 Models Used: {metadata['num_models_used']}/{metadata['total_models']}")
        
        # Log resource stats if available
        if resource_stats:
            self.logger.info("\n📊 RESOURCE UTILIZATION:")
            self.logger.info(f"   CPU Avg: {resource_stats.get('cpu_percent_avg', 0):.1f}% | Max: {resource_stats.get('cpu_percent_max', 0):.1f}%")
            self.logger.info(f"   Memory Avg: {resource_stats.get('memory_percent_avg', 0):.1f}% | Max: {resource_stats.get('memory_percent_max', 0):.1f}%")
            if resource_stats.get('gpu_available'):
                self.logger.info(f"   GPU Avg: {resource_stats.get('gpu_percent_avg', 0):.1f}% | Max: {resource_stats.get('gpu_percent_max', 0):.1f}%")
                self.logger.info(f"   GPU Memory Avg: {resource_stats.get('gpu_memory_percent_avg', 0):.1f}% | Max: {resource_stats.get('gpu_memory_percent_max', 0):.1f}%")
                if resource_stats.get('gpu_temperature_max', 0) > 0:
                    self.logger.info(f"   GPU Temp Avg: {resource_stats.get('gpu_temperature_avg', 0):.1f}°C | Max: {resource_stats.get('gpu_temperature_max', 0):.1f}°C")
                if resource_stats.get('gpu_power_watts_max', 0) > 0:
                    self.logger.info(f"   GPU Power Avg: {resource_stats.get('gpu_power_watts_avg', 0):.1f}W | Max: {resource_stats.get('gpu_power_watts_max', 0):.1f}W")
        
        self.logger.info("\n📋 INDIVIDUAL MODEL RESULTS:")
        for model_result in model_results:
            if model_result['success']:
                self.logger.info(f"  🔹 {model_result['model_name']} - {model_result['classifier']}:")
                weighted_probs = np.array(model_result['weighted_probabilities'])
                prediction_idx = np.argmax(weighted_probs)
                prediction_label = label_map.get(prediction_idx, "Unknown")

                self.logger.info(f"     Prediction: {prediction_label}")
                self.logger.info(f"     Raw Probabilities: {model_result['raw_probabilities']}")
                self.logger.info(f"     Weighted Probabilities: {model_result['weighted_probabilities']}")
                self.logger.info(f"     Weight: {model_result['weight']}")
                self.logger.info(f"     Embedding Model: {model_result['embedding_model']}")
                self.logger.info(f"     Boosting Model: {model_result['classifier']}")
            else:
                self.logger.info(f"  ❌ {model_result['model_name']}: FAILED - {model_result.get('error', 'Unknown error')}")

    def predict_sync(self, prompt: str) -> Tuple[int, np.ndarray, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Main prediction method with confidence-aware voting (synchronous version)
        
        Returns:
            - final_prediction: Predicted class index
            - final_probabilities: Combined probability distribution
            - metadata: Additional information about the prediction
            - model_results: Individual model results for CSV logging
            - resource_stats: Resource utilization statistics
        """
        # Start resource monitoring
        resource_stats = {}
        if self.resource_monitor:
            self.resource_monitor.start()
        
        start_time = time.time()
        
        self.logger.info(f"🚀 Starting prediction for prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
        try:
            # Get embeddings in parallel
            self.logger.info("🔄 Getting embeddings from all models...")
            embeddings = self._get_all_embeddings_sync(prompt)
            
            if not embeddings:
                raise ValueError("Failed to get embeddings from any model")
            
            embedding_time = time.time() - start_time
            self.logger.info(f"✅ Got embeddings from {len(embeddings)} models in {embedding_time:.3f}s")
            
            # Make predictions in parallel
            self.logger.info("🔄 Making predictions with all models...")
            prediction_start = time.time()
            valid_predictions, model_results = self._make_parallel_predictions(embeddings)
            
            if not valid_predictions:
                raise ValueError("Failed to get predictions from any model")
            
            prediction_time = time.time() - prediction_start
            
            # Confidence-aware voting: sum weighted probabilities
            final_probabilities = np.sum(valid_predictions, axis=0)
            final_prediction = np.argmax(final_probabilities)
            
            # Count how many times each class got the highest score
            individual_votes = [np.argmax(p) for p in valid_predictions]
            vote_counts = np.bincount(individual_votes, minlength=2)

            # CHECK IF LLAMA SHOULD BE CALLED
            llama3_prediction = None
            llama3_time = 0.0

            if abs(vote_counts[0] - vote_counts[1]) <= 2:
                self.logger.info("*" * 80)
                self.logger.info("*" * 80)
                self.logger.warning("⚠️ Warning: Close decision. Calling Llama3Classifier for final decision.")
                llama3_start = time.time()
                llama3_prediction = self.llama3.classify_prompt(prompt)
                llama3_time = time.time() - llama3_start
                self.logger.info("=" * 80)

            # Calculate confidence metrics
            prob_sum = np.sum(final_probabilities)
            if prob_sum > 0:
                normalized_probs = final_probabilities / prob_sum
                confidence = np.max(normalized_probs)
                entropy = -np.sum(normalized_probs * np.log(normalized_probs + 1e-10))
            else:
                confidence = 0.0
                entropy = 0.0
            
            total_time = time.time() - start_time
            
            metadata = {
                "num_models_used": len(valid_predictions),
                "total_models": len(self.models) * 2,
                "confidence": confidence,
                "vote_counts": vote_counts.tolist(),
                "entropy": entropy,
                "embedding_time": embedding_time,
                "prediction_time": prediction_time,
                "total_time": total_time,
                "models_with_embeddings": list(embeddings.keys()),
                "llama3_prediction": llama3_prediction,
                "llama3_time": llama3_time
            }
            
        finally:
            # Stop resource monitoring and get stats
            if self.resource_monitor:
                stats = self.resource_monitor.stop()
                resource_stats = stats.to_dict()
        
        # Log detailed results
        self._log_detailed_results(prompt, final_prediction, final_probabilities, 
                                   model_results, metadata, resource_stats)
        
        return final_prediction, final_probabilities, metadata, model_results, resource_stats

    async def predict(self, prompt: str) -> Tuple[int, np.ndarray, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Main prediction method with confidence-aware voting (async version)
        """
        # Start resource monitoring
        resource_stats = {}
        if self.resource_monitor:
            self.resource_monitor.start()
        
        start_time = time.time()
        
        self.logger.info(f"🚀 Starting async prediction for prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
        try:
            # Get embeddings in parallel
            self.logger.info("🔄 Getting embeddings from all models...")
            embeddings = await self._get_all_embeddings_async(prompt)
            
            if not embeddings:
                raise ValueError("Failed to get embeddings from any model")
            
            embedding_time = time.time() - start_time
            self.logger.info(f"✅ Got embeddings from {len(embeddings)} models in {embedding_time:.3f}s")
            
            # Make predictions in parallel
            self.logger.info("🔄 Making predictions with all models...")
            prediction_start = time.time()
            valid_predictions, model_results = self._make_parallel_predictions(embeddings)
            
            if not valid_predictions:
                raise ValueError("Failed to get predictions from any model")
            
            prediction_time = time.time() - prediction_start
            
            # Confidence-aware voting: sum weighted probabilities
            final_probabilities = np.sum(valid_predictions, axis=0)
            final_prediction = np.argmax(final_probabilities)

            # Count how many times each class got the highest score
            individual_votes = [np.argmax(p) for p in valid_predictions]
            vote_counts = np.bincount(individual_votes, minlength=2)

            # CHECK IF LLAMA SHOULD BE CALLED
            llama3_prediction = None
            llama3_time = 0.0

            if abs(vote_counts[0] - vote_counts[1]) <= 2:
                self.logger.info("*" * 80)
                self.logger.warning("⚠️ Warning: Close decision. Calling Llama3Classifier for final decision.")
                llama3_start = time.time()
                llama3_prediction = self.llama3.classify_prompt(prompt)
                llama3_time = time.time() - llama3_start
                self.logger.info(" Llama decision (Final Decision): " + str(llama3_prediction))
                self.logger.info("=" * 80)

            # Calculate confidence metrics
            prob_sum = np.sum(final_probabilities)
            if prob_sum > 0:
                normalized_probs = final_probabilities / prob_sum
                confidence = np.max(normalized_probs)
                entropy = -np.sum(normalized_probs * np.log(normalized_probs + 1e-10))
            else:
                confidence = 0.0
                entropy = 0.0
            
            total_time = time.time() - start_time
            
            metadata = {
                "num_models_used": len(valid_predictions),
                "total_models": len(self.models) * 2,
                "confidence": confidence,
                "vote_counts": vote_counts.tolist(),
                "entropy": entropy,
                "embedding_time": embedding_time,
                "prediction_time": prediction_time,
                "total_time": total_time,
                "models_with_embeddings": list(embeddings.keys()),
                "llama3_prediction": llama3_prediction,
                "llama3_time": llama3_time
            }
            
        finally:
            # Stop resource monitoring and get stats
            if self.resource_monitor:
                stats = self.resource_monitor.stop()
                resource_stats = stats.to_dict()
        
        # Log detailed results
        self._log_detailed_results(prompt, final_prediction, final_probabilities, 
                                   model_results, metadata, resource_stats)
        
        return final_prediction, final_probabilities, metadata, model_results, resource_stats
    
    def predict_batch(self, prompts: List[str]) -> List[Tuple[int, np.ndarray, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]]:
        """
        Predict multiple prompts (synchronous wrapper for batch processing)
        """
        async def _batch_predict():
            tasks = [self.predict(prompt) for prompt in prompts]
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        # Run async batch prediction
        results = asyncio.run(_batch_predict())
        
        # Handle exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Batch prediction failed for prompt {i}: {result}")
                valid_results.append((None, None, {"error": str(result)}, [], {}))
            else:
                valid_results.append(result)
        
        return valid_results