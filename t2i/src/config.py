import os
import yaml
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent


def load_config(config_path: str) -> Dict[str, Any]:
    full_path = PROJECT_ROOT / config_path
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)


def get_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def ensure_dir(path: str) -> Path:
    full_path = get_path(path)
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path


MODELS_CONFIG = load_config('config/models.yaml')
INFERENCE_CONFIG = load_config('config/inference.yaml')

