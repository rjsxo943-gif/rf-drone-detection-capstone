import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_all_configs
from src.receiver.factory import build_receiver
from src.preprocess.dc_blocker import remove_dc_offset
from src.preprocess.iq_normalizer import normalize_iq
from src.preprocess.framing import frame_signal
from src.features.fft import compute_fft_magnitude
from src.features.spectrogram import compute_dual_channel_stft_branch
from src.detect.energy_detector import EnergyDetector
from src.aoa.coherence import coherence_gate
from src.aoa.phase_diff import estimate_phase_diff
from src.aoa.angle_estimator import phase_diff_to_angle
from src.ui.result_plotter import save_energy_plot


def main() -> None:
    cfg = load_all_configs(ROOT / "configs")

    receiver_cfg = cfg["receiver"]
    detect_cfg = cfg["detect"]
    paths_cfg = cfg["paths"]
    aoa_cfg = cfg["aoa"]

    energy_cfg = detect_cfg["energy_detector"]

    receiver = build_receiver(receiver_cfg)

    iq = receiver.read_samples(receiver_cfg["num_samples"])
    iq = remove_dc_offset(iq)
    iq = normalize_iq(iq)

    # 2채널 원본은 STFT/AoA용으로 보관
    iq_for_aoa = iq

    # baseline energy detector는 현재 ch0만 사용
    if iq.ndim == 2:
        iq_energy = iq[0]
    else:
        iq_energy = iq

    frames = frame_signal(
        iq=iq_energy,
        frame_size=energy_cfg["frame_size"],
        hop_size=energy_cfg["hop_size"],
    )

    fft_mag = compute_fft_magnitude(
        frames=frames,
        window=energy_cfg["window"],
    )

    if len(fft_mag) == 0:
        raise RuntimeError("프레임이 0개입니다.")

    frame_energies = np.mean(fft_mag ** 2, axis=1).astype(np.float32)

    detector = EnergyDetector(
        threshold_multiplier=energy_cfg["threshold_multiplier"]
    )

    noise_floor = float(np.median(frame_energies))
    threshold = noise_floor * energy_cfg["threshold_multiplier"]
    detections = frame_energies > threshold

    detector.noise_floor = noise_floor
    detector.threshold = threshold

    # STFT / coherence / phase / AoA
    stft_done = False
    coherence_value = None
    coherence_passed = None
    phase_diff_rad = None
    phase_diff_deg = None
    angle_deg = None
    angle_valid = None
    cnn_spectrogram_shape = None

    if iq_for_aoa.ndim == 2 and iq_for_aoa.shape[0] >= 2:
        branch = compute_dual_channel_stft_branch(
            rx0_iq=iq_for_aoa[0],
            rx1_iq=iq_for_aoa[1],
            sample_rate=receiver_cfg["sample_rate"],
            nperseg=512,
            noverlap=384,
            nfft=512,
            window="hann",
            cnn_source="rx0",
        )

        coherence_result = coherence_gate(
            z0=branch.rx0.complex_stft,
            z1=branch.rx1.complex_stft,
            threshold=0.6,
            energy_percentile=75.0,
        )

        phase_result = estimate_phase_diff(
            iq_block=iq_for_aoa,
            ref_channel=0,
            target_channel=1,
        )

        carrier_freq = float(aoa_cfg.get("carrier_freq", receiver_cfg["center_freq"]))
        antenna_spacing_m = float(aoa_cfg["antenna_spacing_m"])

        angle_result = phase_diff_to_angle(
            phase_diff_rad=phase_result.phase_diff_rad,
            carrier_freq=carrier_freq,
            antenna_spacing_m=antenna_spacing_m,
            phase_offset_rad=0.0,
            clip_input=True,
        )

        stft_done = True
        coherence_value = float(coherence_result.coherence)
        coherence_passed = bool(coherence_result.passed)
        phase_diff_rad = float(phase_result.phase_diff_rad)
        phase_diff_deg = float(phase_result.phase_diff_deg)
        angle_deg = float(angle_result.angle_deg)
        angle_valid = bool(angle_result.valid)
        cnn_spectrogram_shape = list(branch.cnn_spectrogram.shape)

    run_dir = ROOT / paths_cfg["outputs"]["runs"] / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)

    np.save(run_dir / "frame_energies.npy", frame_energies)
    np.save(run_dir / "detections.npy", detections.astype(np.int32))
    np.save(run_dir / "fft_mag.npy", fft_mag)

    if stft_done:
        stage1_dir = run_dir / "stage1"
        stage1_dir.mkdir(parents=True, exist_ok=True)

        np.save(stage1_dir / "cnn_spectrogram.npy", branch.cnn_spectrogram)
        np.save(stage1_dir / "rx0_complex_stft.npy", branch.rx0.complex_stft)
        np.save(stage1_dir / "rx1_complex_stft.npy", branch.rx1.complex_stft)

    summary = {
        "source_type": receiver_cfg["source_type"],
        "num_samples": int(iq_for_aoa.shape[-1]),
        "iq_shape": list(iq_for_aoa.shape),
        "num_frames": int(len(frames)),
        "num_detections": int(np.sum(detections)),
        "noise_floor": float(detector.noise_floor),
        "threshold": float(detector.threshold),
        "detection_ratio": float(np.mean(detections)) if len(detections) > 0 else 0.0,
        "stft_done": stft_done,
        "cnn_spectrogram_shape": cnn_spectrogram_shape,
        "coherence": coherence_value,
        "coherence_passed": coherence_passed,
        "phase_diff_rad": phase_diff_rad,
        "phase_diff_deg": phase_diff_deg,
        "angle_deg": angle_deg,
        "angle_valid": angle_valid,
        "cnn_enabled": False,
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