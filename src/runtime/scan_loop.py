from __future__ import annotations

import json
import select
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.core.config import load_all_configs
from src.calibration import load_calibration_params
from src.runtime.calibration_actions import DEFAULT_NOISE_OUTPUT, DEFAULT_PHASE_GAIN_OUTPUT
from src.ml.runtime_classifier_factory import build_runtime_cnn_classifier
from src.ml.runtime_decision import RuntimeDecisionConfig, load_runtime_decision_config
from src.receiver.factory import build_receiver
from src.scan import FrequencyScanner
from src.scan.precision_analyzer import PrecisionAnalyzer


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ScanRuntime:
    cfg: dict[str, Any]
    receiver: Any
    scanner: FrequencyScanner
    analyzer: PrecisionAnalyzer
    decision_cfg: RuntimeDecisionConfig | None

    run_dir: Path
    precision_dir: Path

    start_freq: float
    stop_freq: float
    step_freq: float
    num_samples: int

    scan_blocks: int
    min_pass_blocks: int
    threshold: float

    cnn_enabled: bool
    save_spectrogram: bool
    save_stft: bool

    def close(self) -> None:
        close_fn = getattr(self.receiver, "close", None)
        if callable(close_fn):
            close_fn()


def _unwrap_scan_cfg(scan_cfg_raw: dict[str, Any]) -> dict[str, Any]:
    if "scan" in scan_cfg_raw and isinstance(scan_cfg_raw["scan"], dict):
        return scan_cfg_raw["scan"]
    return scan_cfg_raw


def _get_receiver_gain(receiver: Any, receiver_cfg: dict[str, Any]) -> float | None:
    get_gain = getattr(receiver, "get_gain", None)
    if callable(get_gain):
        try:
            return float(get_gain())
        except Exception:
            pass

    if hasattr(receiver, "gain"):
        try:
            return float(receiver.gain)
        except Exception:
            pass

    if isinstance(receiver_cfg.get("sdr"), dict) and "gain" in receiver_cfg["sdr"]:
        return float(receiver_cfg["sdr"]["gain"])

    if "gain" in receiver_cfg:
        return float(receiver_cfg["gain"])

    return None


def _receiver_sample_rate(receiver_cfg: dict[str, Any]) -> float:
    if isinstance(receiver_cfg.get("sdr"), dict) and "sample_rate" in receiver_cfg["sdr"]:
        return float(receiver_cfg["sdr"]["sample_rate"])
    return float(receiver_cfg["sample_rate"])


def setup_scan_runtime(
    *,
    config_dir: str | Path = PROJECT_ROOT / "configs",
) -> ScanRuntime:
    """
    scan에 필요한 객체들을 한 번만 생성한다.

    원칙:
    - CNN model path / class_names / threshold / temporal voting은 configs/ml.yaml에서만 읽는다.
    - AoA geometry / coherence / smoothing / sector는 configs/aoa.yaml에서만 읽는다.
    - runtime code 내부 hardcoded model path / AoA threshold 사용을 금지한다.
    """
    cfg = load_all_configs(config_dir)

    receiver_cfg = cfg["receiver"]
    paths_cfg = cfg["paths"]
    ml_cfg = cfg["ml"]
    aoa_cfg = cfg.get("aoa", {}) or {}
    stft_cfg = ml_cfg.get("stft", {})
    coherence_cfg = aoa_cfg.get("coherence", {}) or {}
    smoothing_cfg = aoa_cfg.get("smoothing", {}) or {}

    scan_cfg = _unwrap_scan_cfg(cfg["scan"])

    print()
    print("=== SCAN CONFIG DEBUG ===")
    print(f"scan_loop file : {__file__}")
    print(f"PROJECT_ROOT   : {PROJECT_ROOT}")
    print(f"config_dir     : {Path(config_dir).resolve()}")
    print(f"scan_cfg       : {scan_cfg}")
    print("=========================")
    print()

    start_freq = float(scan_cfg["start_freq"])
    stop_freq = float(scan_cfg["stop_freq"])
    step_freq = float(scan_cfg["step_freq"])

    num_samples = int(scan_cfg["num_samples"])

    threshold = float(scan_cfg["threshold"])
    scan_blocks = int(scan_cfg["scan_blocks"])
    min_pass_blocks = int(scan_cfg["min_pass_blocks"])

    save_spectrogram = bool(scan_cfg.get("save_spectrogram", False))
    save_stft = bool(scan_cfg.get("save_stft", False))

    cnn_enabled = bool(scan_cfg.get("cnn_enabled", False))

    run_dir = PROJECT_ROOT / paths_cfg["outputs"]["runs"] / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)

    precision_dir = run_dir / "scan_precision"
    precision_dir.mkdir(parents=True, exist_ok=True)

    receiver = build_receiver(receiver_cfg)
    current_gain = _get_receiver_gain(receiver, receiver_cfg)

    scanner = FrequencyScanner(
        receiver=receiver,
        start_freq=start_freq,
        stop_freq=stop_freq,
        step_freq=step_freq,
        num_samples=num_samples,
        threshold=threshold,
        scan_blocks=scan_blocks,
        min_pass_blocks=min_pass_blocks,
    )

    cnn_classifier = None
    decision_cfg = None

    if cnn_enabled:
        decision_cfg = load_runtime_decision_config(ml_cfg)
        cnn_classifier = build_runtime_cnn_classifier(ml_cfg)
        print("=== CNN RUNTIME CONFIG ===")
        print(f"backend        : {decision_cfg.backend}")
        print(f"model_path     : {decision_cfg.model_path}")
        print(f"class_names    : {ml_cfg.get('class_names')}")
        print(f"positive_class : {decision_cfg.positive_class}")
        print(f"gain           : {current_gain}")
        print(f"default_thr    : {decision_cfg.default_drone_threshold}")
        print(f"temporal       : window={decision_cfg.temporal_voting.window_size}, "
              f"candidate={decision_cfg.temporal_voting.candidate_vote_k}, "
              f"confirmed={decision_cfg.temporal_voting.confirmed_vote_k}")
        print("==========================")
        print()

    phase_offset_rad = float(aoa_cfg.get("phase_offset_rad", 0.0))

    try:
        calib = load_calibration_params(
            noise_path=PROJECT_ROOT / DEFAULT_NOISE_OUTPUT,
            phase_gain_path=PROJECT_ROOT / DEFAULT_PHASE_GAIN_OUTPUT,
            require_noise=False,
            require_phase_gain=False,
        )

        if calib.phase_gain is not None:
            phase_offset_rad = float(calib.phase_gain.phase_offset_rad)
            print(f"[AoA CAL] phase_offset_rad loaded: {phase_offset_rad:.10f} rad")

    except Exception as e:
        print(f"[AoA CAL WARN] failed to load phase calibration: {e}")
        print(f"[AoA CAL WARN] use aoa.yaml phase_offset_rad={phase_offset_rad:.10f}")

    print("=== AOA RUNTIME CONFIG ===")
    print(f"ref/target     : {aoa_cfg.get('ref_channel', 0)} -> {aoa_cfg.get('target_channel', 1)}")
    print(f"spacing_m      : {aoa_cfg.get('antenna_spacing_m', 0.06)}")
    print(f"phase_offset   : {phase_offset_rad:.10f} rad")
    print(f"coherence_thr  : {coherence_cfg.get('threshold', 0.6)}")
    print(f"energy_pct     : {coherence_cfg.get('energy_percentile', 75.0)}")
    print(f"smoothing      : enabled={smoothing_cfg.get('enabled', False)}, "
          f"method={smoothing_cfg.get('method', 'median')}, "
          f"window={smoothing_cfg.get('window_size', 5)}, "
          f"min={smoothing_cfg.get('min_valid_samples', 1)}")
    print("==========================")
    print()

    analyzer = PrecisionAnalyzer(
        receiver=receiver,
        num_samples=num_samples,
        sample_rate=_receiver_sample_rate(receiver_cfg),
        antenna_spacing_m=float(aoa_cfg.get("antenna_spacing_m", 0.06)),
        nperseg=int(stft_cfg.get("nperseg", 128)),
        noverlap=int(stft_cfg.get("noverlap", 96)),
        nfft=int(stft_cfg.get("nfft", 128)),
        window=str(stft_cfg.get("window", "hann")),
        coherence_threshold=float(coherence_cfg.get("threshold", 0.6)),
        phase_offset_rad=phase_offset_rad,
        settle_sec=float(scan_cfg.get("settle_sec", 0.0)),
        precision_blocks=int(scan_cfg.get("precision_blocks_per_candidate", 1)),
        save_dir=str(precision_dir),
        save_spectrogram=save_spectrogram,
        save_stft=save_stft,
        cnn_classifier=cnn_classifier,
        decision_cfg=decision_cfg,
        current_gain=current_gain,
        aoa_cfg=aoa_cfg,
    )

    return ScanRuntime(
        cfg=cfg,
        receiver=receiver,
        scanner=scanner,
        analyzer=analyzer,
        decision_cfg=decision_cfg,
        run_dir=run_dir,
        precision_dir=precision_dir,
        start_freq=start_freq,
        stop_freq=stop_freq,
        step_freq=step_freq,
        num_samples=num_samples,
        scan_blocks=scan_blocks,
        min_pass_blocks=min_pass_blocks,
        threshold=threshold,
        cnn_enabled=cnn_enabled,
        save_spectrogram=save_spectrogram,
        save_stft=save_stft,
    )


def run_one_scan_cycle(
    runtime: ScanRuntime,
    *,
    cycle_index: int = 1,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    events = runtime.scanner.scan_once()
    event_dicts: list[dict[str, Any]] = []

    if verbose:
        print()
        print(f"=== Scan Cycle {cycle_index} Start ===")
        print(f"cnn_enabled: {runtime.cnn_enabled}")

    for event in events:
        event_dict = asdict(event)

        if not event.triggered:
            event_dict["analysis"] = None
            event_dicts.append(event_dict)
            continue

        if verbose:
            print(
                f"\n[TRIGGER] {event.center_freq / 1e9:.3f} GHz | "
                f"max_fft_power={event.max_fft_power:.3e} | "
                f"pass_count={event.pass_count}/{runtime.scan_blocks}"
            )

        result = runtime.analyzer.analyze(event.center_freq)
        analysis_dict = asdict(result)

        event_dict["analysis"] = analysis_dict
        event_dicts.append(event_dict)

        if verbose:
            print(
                f"  stft_done={result.stft_done} | "
                f"cnn={result.cnn_label}({result.cnn_score}) | "
                f"drone_prob={result.drone_probability} | "
                f"thr={result.drone_threshold} | "
                f"votes={result.drone_vote_count}/{result.temporal_window} | "
                f"candidate={result.candidate_status} | "
                f"confirmed={result.confirmed_status} | "
                f"final={result.final_decision} | "
                f"coherence={result.coherence} | "
                f"angle={result.angle_deg} deg | "
                f"smooth={result.aoa_smoothed_angle_deg} deg "
                f"smooth_valid={result.aoa_smoothing_valid} | "
                f"sector={result.sector_index}({result.sector_label}) | "
                f"sel_block={result.selected_block_index}/{result.precision_blocks} | "
                f"spectrogram_path={result.spectrogram_path}"
            )

    save_scan_events(
        event_dicts,
        runtime=runtime,
        cycle_index=cycle_index,
    )

    if verbose:
        print()
        print(f"=== Scan Cycle {cycle_index} Summary ===")
        print(f"scan range: {runtime.start_freq / 1e9:.3f} GHz ~ {runtime.stop_freq / 1e9:.3f} GHz")
        print(f"step: {runtime.step_freq / 1e6:.1f} MHz")
        print(f"num events: {len(event_dicts)}")
        print(f"triggered events: {sum(1 for e in event_dicts if e['triggered'])}")
        print(f"cnn_enabled: {runtime.cnn_enabled}")
        print(f"save_spectrogram: {runtime.save_spectrogram}")
        print(f"save_stft: {runtime.save_stft}")

    return event_dicts


def save_scan_events(
    event_dicts: list[dict[str, Any]],
    *,
    runtime: ScanRuntime,
    cycle_index: int,
) -> Path:
    latest_path = runtime.run_dir / "scan_events.json"

    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(event_dicts, f, indent=2, ensure_ascii=False)

    cycle_path = runtime.run_dir / f"scan_events_cycle_{cycle_index:06d}.json"

    with cycle_path.open("w", encoding="utf-8") as f:
        json.dump(event_dicts, f, indent=2, ensure_ascii=False)

    return latest_path


def _read_stop_key_if_available(stop_key: str) -> bool:
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)

    if not readable:
        return False

    user_input = sys.stdin.readline().strip().lower()

    if user_input == stop_key:
        return True

    if user_input:
        print(f"[SCAN] scan mode에서는 '{stop_key}'만 처리한다. 입력 무시: {user_input}")

    return False


def run_continuous_scan_loop(
    *,
    config_dir: str | Path = PROJECT_ROOT / "configs",
    stop_key: str = "q",
    cycle_delay_sec: float = 0.0,
    verbose: bool = True,
) -> int:
    runtime = setup_scan_runtime(config_dir=config_dir)

    cycle_index = 0

    if verbose:
        print()
        print("=== Continuous Scan Loop Start ===")
        print(f"stop key: {stop_key}")
        print(f"threshold: {runtime.threshold}")
        print("중단하려면 scan cycle 사이에 q 입력 후 Enter")
        print()

    try:
        while True:
            cycle_index += 1

            run_one_scan_cycle(
                runtime,
                cycle_index=cycle_index,
                verbose=verbose,
            )

            if _read_stop_key_if_available(stop_key):
                print()
                print(f"[SCAN] {stop_key} 입력 감지 → continuous scan 종료")
                break

            if cycle_delay_sec > 0:
                time.sleep(cycle_delay_sec)

    except KeyboardInterrupt:
        print()
        print("[SCAN] KeyboardInterrupt 감지 → continuous scan 종료")

    finally:
        runtime.close()

    if verbose:
        print()
        print("=== Continuous Scan Loop Finished ===")
        print(f"cycles: {cycle_index}")

    return 0
