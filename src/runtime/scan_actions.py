from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from src.calibration import load_calibration_params
from src.runtime.calibration_actions import (
    DEFAULT_NOISE_OUTPUT,
    DEFAULT_PHASE_GAIN_OUTPUT,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCAN_SCRIPT = PROJECT_ROOT / "scripts" / "run_scan.py"


def _make_pythonpath_env() -> dict[str, str]:
    """
    subprocess로 scripts/run_scan.py를 실행할 때도
    PYTHONPATH=. 과 같은 효과가 나도록 환경변수를 구성한다.
    """
    env = os.environ.copy()

    current = env.get("PYTHONPATH", "")
    paths = [str(PROJECT_ROOT)]

    if current:
        paths.append(current)

    env["PYTHONPATH"] = os.pathsep.join(paths)

    return env


def _print_loaded_calibration_status(
    *,
    require_noise: bool,
    require_phase_gain: bool,
) -> None:
    """
    scan 시작 전에 calibration 파일 상태를 확인한다.

    현재 정책:
    - noise calibration은 scan 시작에 필요하다고 본다.
    - phase/gain calibration은 없으면 AoA 신뢰도가 낮아질 수 있으므로 경고만 한다.
    """
    calibration = load_calibration_params(
        noise_path=DEFAULT_NOISE_OUTPUT,
        phase_gain_path=DEFAULT_PHASE_GAIN_OUTPUT,
        require_noise=require_noise,
        require_phase_gain=require_phase_gain,
    )

    print()
    print("=== Scan Calibration Check ===")

    if calibration.noise is not None:
        print("[Noise] loaded")
        print(f"threshold   : {calibration.noise.threshold:.10g}")
        print(f"noise_floor : {calibration.noise.noise_floor:.10g}")
        print(f"source      : {calibration.noise.source_path}")
    else:
        print("[Noise] not loaded")

    if calibration.phase_gain is not None:
        print()
        print("[Phase/Gain] loaded")
        print(f"gain_correction : {calibration.phase_gain.gain_correction:.10g}")
        print(f"phase_offset    : {calibration.phase_gain.phase_offset_rad:.10g} rad")
        print(f"phase_offset    : {calibration.phase_gain.phase_offset_deg:.6f} deg")
        print(f"coherence_like  : {calibration.phase_gain.coherence_like:.10g}")
        print(f"source          : {calibration.phase_gain.source_path}")
    else:
        print()
        print("[Phase/Gain] not loaded")
        print("[WARN] phase/gain calibration이 없으면 AoA 보정은 아직 신뢰하기 어렵다.")

    print()


def run_scan_action(
    *,
    scan_script: str | Path = DEFAULT_SCAN_SCRIPT,
    extra_args: Sequence[str] | None = None,
    require_noise: bool = True,
    require_phase_gain: bool = False,
    verbose: bool = True,
) -> int:
    """
    runtime CLI에서 scan을 실행하는 action.

    현재는 기존 scripts/run_scan.py를 subprocess로 실행한다.

    주의:
    - 이 함수는 scan 실행 연결용 브리지다.
    - calibration 값을 run_scan.py 내부에 직접 적용하는 작업은 다음 단계에서 한다.
    - 정식 구조에서는 state_machine.py의 SCAN/BAND_HOLD 상태로 대체될 수 있다.
    """
    scan_script = Path(scan_script)

    if not scan_script.exists():
        raise FileNotFoundError(f"Scan script not found: {scan_script}")

    _print_loaded_calibration_status(
        require_noise=require_noise,
        require_phase_gain=require_phase_gain,
    )

    cmd = [sys.executable, str(scan_script)]

    if extra_args:
        cmd.extend(list(extra_args))

    if verbose:
        print("=== Run Scan Action ===")
        print(f"project_root : {PROJECT_ROOT}")
        print(f"scan_script  : {scan_script}")
        print(f"command      : {' '.join(cmd)}")
        print()

    env = _make_pythonpath_env()

    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )

    if verbose:
        print()
        print("=== Scan Finished ===")
        print(f"return_code : {completed.returncode}")

    return int(completed.returncode)