import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_all_configs
from src.receiver.sim_receiver import SimReceiver
from src.preprocess.dc_blocker import remove_dc
from src.preprocess.iq_normalizer import normalize_iq
from src.preprocess.framing import frame_signal
from src.features.fft import compute_fft_magnitude
from src.detect.energy_detector import EnergyDetector
from src.ui.result_plotter import save_energy_plot


def main() -> None:
    cfg = load_all_configs(ROOT / "configs")

    receiver_cfg = cfg["receiver"]
    detect_cfg = cfg["detect"]
    paths_cfg = cfg["paths"]

    if receiver_cfg["source_type"] != "sim":
        raise NotImplementedError("현재 최소 동작 버전은 sim만 지원함")

    sim_cfg = receiver_cfg["sim"]

    receiver = SimReceiver(
        sample_rate=receiver_cfg["sample_rate"],
        center_freq=receiver_cfg["center_freq"],
        tone_freq_norm=sim_cfg["tone_freq_norm"],
        noise_std=sim_cfg["noise_std"],
        burst_amplitude=sim_cfg["burst_amplitude"],
        burst_period=sim_cfg["burst_period"],
        burst_length=sim_cfg["burst_length"],
        seed=sim_cfg["seed"],
    )

    iq = receiver.read_samples(receiver_cfg["num_samples"])
    iq = remove_dc(iq)
    iq = normalize_iq(iq)

    frames = frame_signal(
        iq=iq,
        frame_size=detect_cfg["frame_size"],
        hop_size=detect_cfg["hop_size"],
    )

    fft_mag = compute_fft_magnitude(
        frames=frames,
        window=detect_cfg["window"],
    )

    if len(fft_mag) == 0:
        raise RuntimeError("프레임이 0개입니다.")

    frame_energies = np.mean(fft_mag ** 2, axis=1).astype(np.float32)

    detector = EnergyDetector(
        threshold_multiplier=detect_cfg["threshold_multiplier"]
    )
    detections = detector.detect(frame_energies)

    run_dir = ROOT / paths_cfg["outputs"]["runs"] / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)

    np.save(run_dir / "frame_energies.npy", frame_energies)
    np.save(run_dir / "detections.npy", detections.astype(np.int32))
    np.save(run_dir / "fft_mag.npy", fft_mag)

    summary = {
        "num_samples": int(len(iq)),
        "num_frames": int(len(frames)),
        "num_detections": int(np.sum(detections)),
        "noise_floor": float(detector.noise_floor),
        "threshold": float(detector.threshold),
        "detection_ratio": float(np.mean(detections)) if len(detections) > 0 else 0.0,
    }

    save_energy_plot(
        energies=frame_energies,
        threshold=detector.threshold,
        detections=detections,
        save_path=run_dir / "energy_plot.png",
        title="Energy Detector Output",
    )

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== Pipeline Result ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"\nsaved to: {run_dir}")


if __name__ == "__main__":
    main()
