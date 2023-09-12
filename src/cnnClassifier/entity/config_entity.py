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
    tensorboard_root_log_dir = Path
    checkpoint_model_filepath : Path