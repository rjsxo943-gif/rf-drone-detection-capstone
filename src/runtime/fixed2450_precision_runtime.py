"""Fixed 2.450GHz precision runtime.

이 모드는 SCAN 정책을 사용하지 않는다.
2.450GHz에 고정하여 AoA / sector / distance dashboard를 실행한다.

역할:
- 발표/검증 기준 모드
- scripts/experimental/live_aoa_sector_dashboard.py 정책을 그대로 사용
- scan/candidate/pass/ratio 정책과 분리
"""

from __future__ import annotations

import runpy
import sys
from typing import Sequence


def run_fixed2450_precision_runtime(
    *,
    config_dir: str = "configs",
    center_freq_hz: float = 2.450e9,
    extra_args: Sequence[str] | None = None,
) -> None:
    """Run the fixed 2.450GHz AoA/sector/distance dashboard."""

    script_path = "scripts/experimental/live_aoa_sector_dashboard.py"

    argv = [
        script_path,
        "--config-dir",
        str(config_dir),
        "--center-freq",
        str(int(center_freq_hz)),
    ]

    if extra_args:
        argv.extend(str(arg) for arg in extra_args)

    old_argv = sys.argv[:]

    try:
        sys.argv = argv
        runpy.run_path(script_path, run_name="__main__")

    finally:
        sys.argv = old_argv


def main() -> None:
    run_fixed2450_precision_runtime()


if __name__ == "__main__":
    main()
