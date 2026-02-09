# File: /final-filter/final-filter/src/csv_logger.py
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import threading


class CSVLogger:
    """Thread-safe CSV logger for prediction results with resource monitoring"""
    
    def __init__(self, csv_path: str = None, logger=None):
        self.logger = logger
        self._lock = threading.Lock()
        
        if csv_path is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            csv_path = f"/home/adolfo/Desktop/cybermacs/cybermacs/thesis/code/final-filter/final-filter/logs/predictions_{timestamp}.csv"
        
        self.csv_path = csv_path
        self._ensure_csv_exists()
    
    def _get_headers(self):
        """Define CSV headers including resource monitoring, ground truth, and response"""
        return [
            # Basic info
            "timestamp",
            "prompt",
            "prompt_length",
            
            # Ground Truth Label (from input CSV)
            "ground_truth_label",
            
            # Individual model results
            "minilm_lightgbm_prob_safe",
            "minilm_lightgbm_prob_unsafe",
            "minilm_lightgbm_prediction",
            "minilm_xgboost_prob_safe",
            "minilm_xgboost_prob_unsafe",
            "minilm_xgboost_prediction",
            
            "snowflake_lightgbm_prob_safe",
            "snowflake_lightgbm_prob_unsafe",
            "snowflake_lightgbm_prediction",
            "snowflake_xgboost_prob_safe",
            "snowflake_xgboost_prob_unsafe",
            "snowflake_xgboost_prediction",
            
            "granite_lightgbm_prob_safe",
            "granite_lightgbm_prob_unsafe",
            "granite_lightgbm_prediction",
            "granite_xgboost_prob_safe",
            "granite_xgboost_prob_unsafe",
            "granite_xgboost_prediction",
            
            # Voting results
            "vote_safe",
            "vote_unsafe",
            "total_models_voted",
            
            # Combined probabilities
            "combined_prob_safe",
            "combined_prob_unsafe",
            
            # Final ensemble prediction
            "ensemble_prediction",
            "ensemble_prediction_label",
            "confidence",
            "entropy",
            
            # Timing
            "embedding_time_sec",
            "prediction_time_sec",
            "classifiers_total_time_sec",
            
            # LLM Guard
            "needed_llmguard",
            "llmguard_input_result",
            "llmguard_input_category",
            "llmguard_output_result",
            "llmguard_output_category",
            "llmguard_time_sec",
            
            # Final results
            "final_prediction",
            "final_prediction_label",
            "total_time_sec",
            
            # LLM Response
            "response",
            "response_length",
            "response_blocked",
            
            # CPU metrics
            "cpu_percent_avg",
            "cpu_percent_max",
            "cpu_percent_min",
            "memory_percent_avg",
            "memory_percent_max",
            "memory_used_mb_avg",
            "memory_used_mb_max",
            
            # GPU metrics
            "gpu_percent_avg",
            "gpu_percent_max",
            "gpu_percent_min",
            "gpu_memory_percent_avg",
            "gpu_memory_percent_max",
            "gpu_memory_used_mb_avg",
            "gpu_memory_used_mb_max",
            "gpu_temperature_avg",
            "gpu_temperature_max",
            "gpu_power_watts_avg",
            "gpu_power_watts_max",
            
            # Resource monitoring metadata
            "num_resource_samples",
            "monitoring_duration_sec",
            "gpu_available",
            "gpu_name",
            "cpu_count",
            
            # Additional metadata
            "models_with_embeddings",
            "num_models_used",
            "total_models_available"
        ]
    
    def _ensure_csv_exists(self):
        """Create CSV file with headers if it doesn't exist"""
        csv_path = Path(self.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not csv_path.exists():
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self._get_headers())
            if self.logger:
                self.logger.info(f"📄 Created new CSV log file: {self.csv_path}")
    
    def _extract_model_results(self, model_results: list) -> Dict[str, Any]:
        """Extract individual model results into a flat dictionary"""
        extracted = {}
        
        # Initialize all model fields with None
        model_names = ["minilm", "snowflake", "granite"]
        classifiers = ["lightgbm", "xgboost"]
        
        for model in model_names:
            for clf in classifiers:
                extracted[f"{model}_{clf}_prob_safe"] = None
                extracted[f"{model}_{clf}_prob_unsafe"] = None
                extracted[f"{model}_{clf}_prediction"] = None
        
        # Fill in actual results
        for result in model_results:
            if not result.get('success', False):
                continue
            
            model_name = result.get('model_name', '').lower()
            classifier = result.get('classifier', '').lower()
            
            # Map model names to our standard names
            if 'minilm' in model_name or 'mini' in model_name:
                model_key = 'minilm'
            elif 'snowflake' in model_name:
                model_key = 'snowflake'
            elif 'granite' in model_name:
                model_key = 'granite'
            else:
                continue
            
            # Get probabilities
            raw_probs = result.get('raw_probabilities', [None, None])
            if raw_probs and len(raw_probs) >= 2:
                extracted[f"{model_key}_{classifier}_prob_safe"] = raw_probs[0]
                extracted[f"{model_key}_{classifier}_prob_unsafe"] = raw_probs[1]
                # Prediction is the argmax
                extracted[f"{model_key}_{classifier}_prediction"] = 0 if raw_probs[0] > raw_probs[1] else 1
        
        return extracted
    
    def _parse_llmguard_result(self, llmguard_result: Optional[str]) -> tuple:
        """Parse LLM Guard result string into (result, category)"""
        if llmguard_result is None:
            return None, None
        
        result_str = str(llmguard_result)
        
        # Check if safe or unsafe
        if "safe" in result_str.lower():
            if "unsafe" in result_str.lower():
                result = "unsafe"
            else:
                result = "safe"
        else:
            result = "unknown"
        
        # Extract category if present
        category = None
        if "Reason Code:" in result_str:
            try:
                parts = result_str.split("Reason Code:")
                if len(parts) > 1:
                    category = parts[1].split(",")[0].strip()
            except:
                pass
        
        return result, category
    
    def log_prediction(self, 
                       prompt: str,
                       final_prediction: int,
                       final_probabilities: list,
                       metadata: Dict[str, Any],
                       model_results: list,
                       llmguard_input_result: Optional[str] = None,
                       llmguard_output_result: Optional[str] = None,
                       llmguard_time: float = 0.0,
                       llm_response: Optional[str] = None,
                       resource_stats: Optional[Dict[str, Any]] = None,
                       ground_truth_label: Optional[int] = None,
                       response_blocked: bool = False):
        """Log a prediction result to CSV including resource metrics, ground truth label, and response"""
        
        label_map = {0: "Safe", 1: "Unsafe"}
        
        # Extract model results
        model_data = self._extract_model_results(model_results)
        
        # Parse LLM Guard results
        input_result, input_category = self._parse_llmguard_result(llmguard_input_result)
        output_result, output_category = self._parse_llmguard_result(llmguard_output_result)
        
        # Determine if LLM Guard was needed
        needed_llmguard = 1 if metadata.get('llama3_prediction') is not None else 0
        
        # Determine final prediction
        if needed_llmguard and llmguard_input_result:
            if "safe" in str(llmguard_input_result).lower() and "unsafe" not in str(llmguard_input_result).lower():
                final_pred = 0
            else:
                final_pred = 1
        else:
            final_pred = final_prediction
        
        # Default resource stats if not provided
        if resource_stats is None:
            resource_stats = {}
        
        # Process response for CSV storage
        response_text = None
        response_len = 0
        if llm_response:
            # Clean response for CSV (remove newlines, limit length)
            response_text = llm_response.replace('\n', ' ').replace('\r', ' ')[:1000]
            response_len = len(llm_response)
        
        # Build row data
        row = {
            # Basic info
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt.replace('\n', ' ').replace('\r', ' ')[:500],
            "prompt_length": len(prompt),
            
            # Ground truth label
            "ground_truth_label": ground_truth_label,
            
            # Individual model results
            **model_data,
            
            # Voting results
            "vote_safe": metadata.get('vote_counts', [0, 0])[0],
            "vote_unsafe": metadata.get('vote_counts', [0, 0])[1],
            "total_models_voted": metadata.get('num_models_used', 0),
            
            # Combined probabilities
            "combined_prob_safe": final_probabilities[0] if len(final_probabilities) > 0 else None,
            "combined_prob_unsafe": final_probabilities[1] if len(final_probabilities) > 1 else None,
            
            # Ensemble prediction
            "ensemble_prediction": final_prediction,
            "ensemble_prediction_label": label_map.get(final_prediction, "Unknown"),
            "confidence": metadata.get('confidence', 0),
            "entropy": metadata.get('entropy', 0),
            
            # Timing
            "embedding_time_sec": round(metadata.get('embedding_time', 0), 4),
            "prediction_time_sec": round(metadata.get('prediction_time', 0), 4),
            "classifiers_total_time_sec": round(
                metadata.get('embedding_time', 0) + metadata.get('prediction_time', 0), 4
            ),
            
            # LLM Guard
            "needed_llmguard": needed_llmguard,
            "llmguard_input_result": input_result,
            "llmguard_input_category": input_category,
            "llmguard_output_result": output_result,
            "llmguard_output_category": output_category,
            "llmguard_time_sec": round(llmguard_time, 4),
            
            # Final results
            "final_prediction": final_pred,
            "final_prediction_label": label_map.get(final_pred, "Unknown"),
            "total_time_sec": round(metadata.get('total_time', 0) + llmguard_time, 4),
            
            # LLM Response
            "response": response_text,
            "response_length": response_len,
            "response_blocked": 1 if response_blocked else 0,
            
            # CPU metrics
            "cpu_percent_avg": resource_stats.get('cpu_percent_avg', 0),
            "cpu_percent_max": resource_stats.get('cpu_percent_max', 0),
            "cpu_percent_min": resource_stats.get('cpu_percent_min', 0),
            "memory_percent_avg": resource_stats.get('memory_percent_avg', 0),
            "memory_percent_max": resource_stats.get('memory_percent_max', 0),
            "memory_used_mb_avg": resource_stats.get('memory_used_mb_avg', 0),
            "memory_used_mb_max": resource_stats.get('memory_used_mb_max', 0),
            
            # GPU metrics
            "gpu_percent_avg": resource_stats.get('gpu_percent_avg', 0),
            "gpu_percent_max": resource_stats.get('gpu_percent_max', 0),
            "gpu_percent_min": resource_stats.get('gpu_percent_min', 0),
            "gpu_memory_percent_avg": resource_stats.get('gpu_memory_percent_avg', 0),
            "gpu_memory_percent_max": resource_stats.get('gpu_memory_percent_max', 0),
            "gpu_memory_used_mb_avg": resource_stats.get('gpu_memory_used_mb_avg', 0),
            "gpu_memory_used_mb_max": resource_stats.get('gpu_memory_used_mb_max', 0),
            "gpu_temperature_avg": resource_stats.get('gpu_temperature_avg', 0),
            "gpu_temperature_max": resource_stats.get('gpu_temperature_max', 0),
            "gpu_power_watts_avg": resource_stats.get('gpu_power_watts_avg', 0),
            "gpu_power_watts_max": resource_stats.get('gpu_power_watts_max', 0),
            
            # Resource monitoring metadata
            "num_resource_samples": resource_stats.get('num_resource_samples', 0),
            "monitoring_duration_sec": resource_stats.get('monitoring_duration_sec', 0),
            "gpu_available": resource_stats.get('gpu_available', False),
            "gpu_name": resource_stats.get('gpu_name', ''),
            "cpu_count": resource_stats.get('cpu_count', 0),
            
            # Additional metadata
            "models_with_embeddings": ",".join(metadata.get('models_with_embeddings', [])),
            "num_models_used": metadata.get('num_models_used', 0),
            "total_models_available": metadata.get('total_models', 0)
        }
        
        # Write to CSV (thread-safe)
        with self._lock:
            try:
                with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self._get_headers())
                    writer.writerow(row)
                
                if self.logger:
                    self.logger.info(f"📝 Logged prediction to CSV: {self.csv_path}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"❌ Failed to write to CSV: {e}")
    
    def get_csv_path(self) -> str:
        """Return the current CSV file path"""
        return self.csv_path