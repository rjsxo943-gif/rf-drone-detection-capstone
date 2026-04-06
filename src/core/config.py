from pathlib import Path
import yaml


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def load_all_configs(config_dir: str | Path = "configs") -> dict:
    config_dir = Path(config_dir)
    return {
        "receiver": load_yaml(config_dir / "receiver.yaml"),
        "detect": load_yaml(config_dir / "detect.yaml"),
        "paths": load_yaml(config_dir / "paths.yaml"),
    }
