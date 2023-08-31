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