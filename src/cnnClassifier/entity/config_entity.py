from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    auth_method: str = None
    api_key: str = None
    username: str = None
    password: str = None
    oauth_token: str = None
    
@dataclass(frozen=True) 
class PrepareBaseModelConfig:
    root_dir : Path
    base_model_path : Path
    updated_base_model_path : Path
    params_image_size : list 
    params_learning_rate : float 
    params_include_top : bool
    params_weights : str
    params_classes : int
    
    
@dataclass(frozen=True)
class PrepareCallbackConfig:
    root_dir : Path
    tensorboard_root_log_dir : Path
    checkpoint_model_filepath : Path
    
@dataclass(frozen=True)
class TrainingConfig:
    root_dir : Path
    trained_model_path : Path
    updated_base_model_path : Path
    training_data : Path
    score_path : Path
    params_epochs: int
    params_batch_size : int
    params_is_augmentation: bool
    params_image_size: list

@dataclass(frozen=True)
class EvaluationConfig:
    path_to_model : Path
    training_data : Path
    score_path : Path
    params_image_size : list
    params_batch_size :int
    
@dataclass(frozen=True)
class InferenceConfig:
    path_to_model : Path
    testing_data : Path
    params_image_size : list
    params_batch_size :int
    