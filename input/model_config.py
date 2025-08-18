from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    embedding_model: str
    model_lightgbm_file: str
    model_xgboost_file: str
    weight: float