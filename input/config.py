# File: /final-filter/final-filter/src/config.py
from model_config import ModelConfig
MODEL_CONFIGS = [
    ModelConfig(
        name="MiniLM",
        embedding_model="all-minilm:33m",
        model_lightgbm_file="your_path/lightgbm_model_allminilm_without_pca.pkl",
        model_xgboost_file="your_path/xgboost_model_allminilm_without_pca.pkl",
        weight=0.79
    ),
    ModelConfig(
        name="Snowflake",
        embedding_model="snowflake-arctic-embed:33m",
        model_lightgbm_file="your_path/lightgbm_model_snowflake_without_pca.pkl",
        model_xgboost_file="your_path/xgboost_model_snowflake_without_pca.pkl",
        weight=0.79
    ),
    ModelConfig(
        name="Granite",
        embedding_model="granite-embedding:30m",
        model_lightgbm_file="your_path/lightgbm_model_granite_without_pca.pkl",
        model_xgboost_file="your_path/xgboost_model_granite_without_pca.pkl",
        weight=0.79
    )

]

LOG_FILENAME_TEMPLATE = "your_path/logs/ensemble_classifier_{timestamp}.log"
JSON_LOG_FILENAME_TEMPLATE = "your_path/logs/detailed_predictions_{date}.jsonl"
