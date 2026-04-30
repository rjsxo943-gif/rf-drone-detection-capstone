from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


CONFIG_FILES = {
    "receiver": "receiver.yaml",
    "detect": "detect.yaml",
    "ml": "ml.yaml",
    "aoa": "aoa.yaml",
    "paths": "paths.yaml",
    "ui": "ui.yaml",
    "scan": "scan.yaml",
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    YAML 파일 하나를 읽어서 dict로 반환한다.
    파일이 비어 있으면 빈 dict를 반환한다.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data if data is not None else {}


def load_all_configs(config_dir: str | Path = "configs") -> dict[str, dict[str, Any]]:
    """
    configs/ 폴더 안의 주요 설정 파일들을 모두 읽는다.

    반환 구조:
    {
        "receiver": {...},
        "detect": {...},
        "ml": {...},
        "aoa": {...},
        "paths": {...},
        "ui": {...},
        "scan": {...},
    }
    """
    config_dir = Path(config_dir)

    configs: dict[str, dict[str, Any]] = {}

    for name, filename in CONFIG_FILES.items():
        configs[name] = load_yaml(config_dir / filename)

    validate_block_size_consistency(configs)

    return configs


def validate_block_size_consistency(configs: dict[str, dict[str, Any]]) -> None:
    """
    receiver / detect / ml / aoa 설정의 block_size가 서로 다르면 오류를 발생시킨다.

    현재 프로젝트 기준:
    - block_size = 16384
    - 주요 설정 파일에서 같은 값을 사용하는 것이 안전하다.
    """
    block_sizes: dict[str, int] = {}

    for config_name in ["receiver", "detect", "ml", "aoa"]:
        cfg = configs.get(config_name, {})

        if "block_size" in cfg:
            block_sizes[config_name] = int(cfg["block_size"])

    if not block_sizes:
        return

    unique_values = set(block_sizes.values())

    if len(unique_values) > 1:
        raise ValueError(
            "block_size mismatch across config files: "
            f"{block_sizes}. All block_size values should be the same."
        )


def get_block_size(configs: dict[str, dict[str, Any]]) -> int:
    """
    전체 프로젝트에서 사용할 block_size를 반환한다.

    우선순위:
    1. ml.yaml의 block_size
    2. receiver.yaml의 block_size
    3. 기본값 16384
    """
    if "block_size" in configs.get("ml", {}):
        return int(configs["ml"]["block_size"])

    if "block_size" in configs.get("receiver", {}):
        return int(configs["receiver"]["block_size"])

    return 16384