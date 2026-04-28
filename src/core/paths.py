from __future__ import annotations

from pathlib import Path
from typing import Any


def get_project_root() -> Path:
    """
    프로젝트 루트 경로를 반환한다.

    기준:
    - 이 파일 위치: src/core/paths.py
    - parents[2] = 프로젝트 루트
    """
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, root: str | Path | None = None) -> Path:
    """
    문자열 경로를 Path 객체로 변환한다.

    상대경로이면 프로젝트 루트 기준으로 변환한다.
    절대경로이면 그대로 반환한다.
    """
    path = Path(path)

    if path.is_absolute():
        return path

    if root is None:
        root = get_project_root()

    return Path(root) / path


def flatten_paths(paths_cfg: dict[str, Any]) -> dict[str, Path]:
    """
    paths.yaml의 중첩 구조를 평평한 dict로 변환한다.

    예:
    paths_cfg["outputs"]["latest_run"]
    → result["outputs.latest_run"]
    """
    result: dict[str, Path] = {}

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, sub_value in value.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                _walk(new_prefix, sub_value)
        else:
            result[prefix] = resolve_path(value)

    _walk("", paths_cfg)

    return result


def ensure_dir(path: str | Path) -> Path:
    """
    폴더가 없으면 생성하고 Path를 반환한다.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_project_dirs(paths_cfg: dict[str, Any]) -> dict[str, Path]:
    """
    paths.yaml에 등록된 주요 폴더들을 생성한다.

    반환:
    - key: "outputs.latest_run" 같은 이름
    - value: 실제 Path 객체
    """
    paths = flatten_paths(paths_cfg)

    for path in paths.values():
        ensure_dir(path)

    return paths