import yaml
from pathlib import Path

def load_yaml(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class Config:
    def __init__(self, yaml_path=None):
        # Default path relative to project root
        if yaml_path is None:
            yaml_path = Path(__file__).parent.parent.parent / "configs" / "absa.yaml"
        
        cfg = load_yaml(yaml_path)

        # Model config
        self.MODEL_NAME = cfg["model"]["name"]
        self.MAX_LEN = cfg["model"]["max_len"]

        # Training config
        self.TRAIN_BATCH_SIZE = cfg["training"]["batch_size_train"]
        self.VALID_BATCH_SIZE = cfg["training"]["batch_size_valid"]
        self.EPOCHS = cfg["training"]["epochs"]
        self.LR = cfg["training"]["learning_rate"]

        # Aspects
        self.ASPECTS = cfg["aspects"]

        # Paths
        self.TRAIN_PATH = cfg["paths"]["train"]
        self.VALID_PATH = cfg["paths"]["valid"]
        self.OUTPUT_DIR = cfg["paths"]["output_dir"]

_config_instance = None

def get_config(yaml_path=None):
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(yaml_path)
    return _config_instance