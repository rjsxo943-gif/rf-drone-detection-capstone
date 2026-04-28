from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import get_block_size, load_all_configs
from .paths import (
    ensure_dir,
    ensure_project_dirs,
    flatten_paths,
    get_project_root,
    resolve_path,
)
from .utils import format_block_filename


@dataclass
class PipelineContext:
    """
    파이프라인 실행에 필요한 공통 정보를 묶어두는 객체.

    포함 내용:
    - 프로젝트 루트 경로
    - configs/*.yaml 전체 설정
    - paths.yaml에서 읽은 주요 경로
    - 공통 block_size
    """

    project_root: Path
    config_dir: Path
    configs: dict[str, dict[str, Any]]
    paths: dict[str, Path]
    block_size: int

    @property
    def receiver_cfg(self) -> dict[str, Any]:
        return self.configs.get("receiver", {})

    @property
    def detect_cfg(self) -> dict[str, Any]:
        return self.configs.get("detect", {})

    @property
    def ml_cfg(self) -> dict[str, Any]:
        return self.configs.get("ml", {})

    @property
    def aoa_cfg(self) -> dict[str, Any]:
        return self.configs.get("aoa", {})

    @property
    def paths_cfg(self) -> dict[str, Any]:
        return self.configs.get("paths", {})

    @property
    def ui_cfg(self) -> dict[str, Any]:
        return self.configs.get("ui", {})

    def get_path(self, key: str) -> Path:
        """
        paths.yaml에 등록된 경로를 가져온다.

        예:
        ctx.get_path("outputs.latest_run")
        ctx.get_path("outputs.stage1")
        ctx.get_path("data.raw_iq_pluto")
        """
        if key not in self.paths:
            raise KeyError(
                f"Path key not found: {key}. "
                f"Available keys: {sorted(self.paths.keys())}"
            )

        return self.paths[key]

    def ensure_path(self, key: str) -> Path:
        """
        paths.yaml에 등록된 경로를 가져오고, 폴더가 없으면 생성한다.
        """
        path = self.get_path(key)
        ensure_dir(path)
        return path

    def latest_run_dir(self) -> Path:
        """
        최신 실행 결과 저장 폴더를 반환한다.
        """
        return self.ensure_path("outputs.latest_run")

    def stage1_dir(self) -> Path:
        """
        Stage 1 산출물 저장 폴더를 반환한다.
        """
        return self.ensure_path("outputs.stage1")

    def stage1_artifact_path(self, block_index: int) -> Path:
        """
        block_index에 해당하는 Stage 1 artifact 저장 경로를 만든다.

        예:
        outputs/runs/latest/stage1/block_000000.npz
        """
        return self.stage1_dir() / format_block_filename(block_index)

    def raw_iq_pluto_dir(self) -> Path:
        """
        Pluto+ raw IQ 저장 루트 폴더를 반환한다.
        """
        return self.ensure_path("data.raw_iq_pluto")


def setup_pipeline(
    config_dir: str | Path = "configs",
    create_dirs: bool = True,
) -> PipelineContext:
    """
    파이프라인 실행에 필요한 공통 초기화를 수행한다.

    하는 일:
    1. 프로젝트 루트 찾기
    2. configs/*.yaml 로드
    3. block_size 일관성 검사
    4. paths.yaml 경로 변환
    5. 필요한 폴더 생성

    반환:
    - PipelineContext
    """

    project_root = get_project_root()
    config_dir_path = resolve_path(config_dir, root=project_root)

    configs = load_all_configs(config_dir_path)
    block_size = get_block_size(configs)

    paths_cfg = configs.get("paths", {})

    if create_dirs:
        paths = ensure_project_dirs(paths_cfg)
    else:
        paths = flatten_paths(paths_cfg)

    return PipelineContext(
        project_root=project_root,
        config_dir=config_dir_path,
        configs=configs,
        paths=paths,
        block_size=block_size,
    )