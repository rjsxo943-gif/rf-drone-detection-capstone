from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_all_configs
from src.receiver.factory import build_receiver
from src.scan import FrequencyScanner, PrecisionAnalyzer


def main() -> None:
    cfg = load_all_configs(ROOT / "configs")

    receiver_cfg = cfg["receiver"]
    paths_cfg = cfg["paths"]

    receiver = build_receiver(receiver_cfg)

    # 임시 스캔 설정
    # 나중에 configs/scan.yaml로 뺄 예정
    start_freq = 2_400_000_000
    stop_freq = 2_485_000_000
    step_freq = 5_000_000

    num_samples = 16_384
    threshold = 1.0e7

    scan_blocks = 3
    min_pass_blocks = 2

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


    analyzer = PrecisionAnalyzer(
        receiver=receiver,
        num_samples=num_samples,
        sample_rate=receiver_cfg["sample_rate"],
        antenna_spacing_m=cfg["aoa"]["antenna_spacing_m"],
    )

    run_dir = ROOT / paths_cfg["outputs"]["runs"] / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)

    events = scanner.scan_once()
    event_dicts = []

    print("=== Scan Start (State Machine Mode) ===")

    for event in events:
        event_dict = asdict(event)

        if not event.triggered:
            event_dict["analysis"] = None
            event_dicts.append(event_dict)
            continue

        print(
            f"\n[TRIGGER] {event.center_freq / 1e9:.3f} GHz | "
            f"max_fft_power={event.max_fft_power:.3e} | "
            f"pass_count={event.pass_count}/{scan_blocks}"
        )

        result = analyzer.analyze(event.center_freq)
        analysis_dict = asdict(result)
        event_dict["analysis"] = analysis_dict
        event_dicts.append(event_dict)

        print(
            f"  stft_done={result.stft_done} | "
            f"coherence={result.coherence} | "
            f"angle={result.angle_deg} deg | "
            f"valid={result.angle_valid}"
        )

    save_path = run_dir / "scan_events.json"
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(event_dicts, f, indent=2, ensure_ascii=False)

    print("\n=== Scan Summary ===")
    print(f"scan range: {start_freq / 1e9:.3f} GHz ~ {stop_freq / 1e9:.3f} GHz")
    print(f"step: {step_freq / 1e6:.1f} MHz")
    print(f"num events: {len(event_dicts)}")
    print(f"triggered events: {sum(1 for e in event_dicts if e['triggered'])}")
    print(f"saved to: {save_path}")




if __name__ == "__main__":
    main()