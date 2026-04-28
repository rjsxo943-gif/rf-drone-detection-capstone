from __future__ import annotations

# ============================================================
# Config
# ============================================================

from .config import (
    get_block_size,
    load_all_configs,
    load_yaml,
    validate_block_size_consistency,
)

# ============================================================
# Paths
# ============================================================

from .paths import (
    ensure_dir,
    ensure_project_dirs,
    flatten_paths,
    get_project_root,
    resolve_path,
)

# ============================================================
# Pipeline
# ============================================================

from .pipeline import (
    PipelineContext,
    setup_pipeline,
)

# ============================================================
# Raw IQ store
# ============================================================

from .raw_iq_store import (
    create_raw_iq_session,
    load_raw_iq_block,
    save_raw_iq_block,
)

# ============================================================
# Stage 1 artifact store
# ============================================================

from .stage1_artifact_store import (
    save_stage1_artifacts,
)

# ============================================================
# Types
# ============================================================

from .types import (
    AOAResult,
    BlockPipelineResult,
    ClassificationResult,
    RawIQBlock,
    STFTParams,
    Stage1Artifacts,
)

# ============================================================
# Utils
# ============================================================

from .utils import (
    check_non_empty_array,
    check_same_shape,
    dumps_json,
    ensure_parent_dir,
    ensure_suffix,
    format_block_filename,
    get_sample_range,
    load_json,
    loads_json,
    now_iso,
    now_string,
    save_json,
    to_complex64_1d,
    to_float32_array,
)


__all__ = [
    # config
    "load_yaml",
    "load_all_configs",
    "validate_block_size_consistency",
    "get_block_size",

    # paths
    "get_project_root",
    "resolve_path",
    "flatten_paths",
    "ensure_dir",
    "ensure_project_dirs",

    # pipeline
    "PipelineContext",
    "setup_pipeline",

    # raw IQ store
    "create_raw_iq_session",
    "save_raw_iq_block",
    "load_raw_iq_block",

    # stage1 artifact store
    "save_stage1_artifacts",

    # types
    "RawIQBlock",
    "STFTParams",
    "Stage1Artifacts",
    "ClassificationResult",
    "AOAResult",
    "BlockPipelineResult",

    # utils
    "now_string",
    "now_iso",
    "format_block_filename",
    "ensure_parent_dir",
    "ensure_suffix",
    "to_complex64_1d",
    "to_float32_array",
    "check_same_shape",
    "check_non_empty_array",
    "get_sample_range",
    "dumps_json",
    "loads_json",
    "save_json",
    "load_json",
]