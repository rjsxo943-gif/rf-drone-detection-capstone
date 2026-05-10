import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_all_configs
from src.receiver.factory import build_receiver
from src.preprocess import (
    remove_dc_offset,
    estimate_and_apply_gain_correction,
    estimate_phase_offset,
    remove_phase_offset,
    normalize_iq,
    frame_signal,
)
from src.features.fft import compute_fft_magnitude
from src.features.spectrogram import (
    compute_stft_branch,
    compute_dual_channel_stft_branch,
)
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
        # RF3 CNN 학습 데이터 기준 STFT 설정
    stft_nperseg = 128
    stft_noverlap = 96
    stft_nfft = 128
    stft_window = "hann"
    expected_cnn_shape = (128, 509)

    receiver = build_receiver(receiver_cfg)

    gain_estimate = None
    phase_estimate = None
    fixed_gain_correction = 1.0
    fixed_phase_offset_rad = 0.0

    # 1) Calibration block: 처음 한 block은 보정값 추정용
    cal_iq = receiver.read_samples(receiver_cfg["num_samples"])
    cal_iq = remove_dc_offset(cal_iq)

    if cal_iq.ndim == 2 and cal_iq.shape[0] >= 2:
        cal_iq, gain_estimate = estimate_and_apply_gain_correction(
            cal_iq,
            ref_channel=0,
            target_channel=1,
        )
        fixed_gain_correction = gain_estimate.gain_correction

        phase_estimate = estimate_phase_offset(
            cal_iq,
            ref_channel=0,
            target_channel=1,
        )
        fixed_phase_offset_rad = phase_estimate.phase_offset_rad

    # 2) Detection block: 실제 분석용 block은 새로 읽음
    iq = receiver.read_samples(receiver_cfg["num_samples"])
    iq = remove_dc_offset(iq)

    if iq.ndim == 2 and iq.shape[0] >= 2:
        # calibration에서 구한 gain correction 적용
        iq[:, :] = iq.astype(np.complex64, copy=False)
        iq[1] = iq[1] * float(fixed_gain_correction)

        # calibration에서 구한 phase offset 적용
        iq = remove_phase_offset(
            iq,
            phase_offset_rad=fixed_phase_offset_rad,
            ref_channel=0,
            target_channel=1,
        )

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
        mode=energy_cfg.get("mode", "block_median"),
        threshold_multiplier=energy_cfg["threshold_multiplier"],
        frame_size=energy_cfg["frame_size"],
        hop_size=energy_cfg["hop_size"],
        window=energy_cfg["window"],
        method=energy_cfg["method"],
        min_detection_ratio=energy_cfg["min_detection_ratio"],
        calibration_num_blocks=energy_cfg.get("calibration_blocks", 20),
        require_calibration=False,
    )

    frame_energies = detector.compute_frame_energies(iq_for_aoa)
    detections = detector.detect_frame_energies(frame_energies)
   


    # STFT / coherence / phase / AoA
    # STFT / coherence / phase / AoA
    stft_done = False
    stft_mode = None  # "single" or "dual"
    branch = None

    coherence_value = None
    coherence_passed = None
    phase_diff_rad = None
    phase_diff_deg = None
    angle_deg = None
    angle_valid = None
    cnn_spectrogram_shape = None

    # -----------------------------
    # 1채널 입력: CNN/STFT만 수행
    # -----------------------------
    if iq_for_aoa.ndim == 1 or (iq_for_aoa.ndim == 2 and iq_for_aoa.shape[0] == 1):
        if iq_for_aoa.ndim == 1:
            single_iq = iq_for_aoa
        else:
            single_iq = iq_for_aoa[0]

        branch = compute_stft_branch(
            iq_block=single_iq,
            sample_rate=receiver_cfg["sample_rate"],
            nperseg=stft_nperseg,
            noverlap=stft_noverlap,
            nfft=stft_nfft,
            window=stft_window,
        )

        stft_done = True
        stft_mode = "single"
        cnn_spectrogram_shape = list(branch.cnn_spectrogram.shape)

        # 1채널이므로 AoA 관련 값은 None 유지
        coherence_value = None
        coherence_passed = None
        phase_diff_rad = None
        phase_diff_deg = None
        angle_deg = None
        angle_valid = None

    # -----------------------------
    # 2채널 입력: CNN/STFT + AoA 수행
    # -----------------------------
    elif iq_for_aoa.ndim == 2 and iq_for_aoa.shape[0] >= 2:
        branch = compute_dual_channel_stft_branch(
            rx0_iq=iq_for_aoa[0],
            rx1_iq=iq_for_aoa[1],
            sample_rate=receiver_cfg["sample_rate"],
            nperseg=stft_nperseg,
            noverlap=stft_noverlap,
            nfft=stft_nfft,
            window=stft_window,
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
        stft_mode = "dual"

        coherence_value = float(coherence_result.coherence)
        coherence_passed = bool(coherence_result.passed)
        phase_diff_rad = float(phase_result.phase_diff_rad)
        phase_diff_deg = float(phase_result.phase_diff_deg)
        angle_deg = float(angle_result.angle_deg)
        angle_valid = bool(angle_result.valid)
        cnn_spectrogram_shape = list(branch.cnn_spectrogram.shape)

    if stft_done and branch is not None:
        actual_shape = tuple(branch.cnn_spectrogram.shape)

        if actual_shape != expected_cnn_shape:
            raise RuntimeError(
                f"Unexpected CNN spectrogram shape: {actual_shape}, "
                f"expected={expected_cnn_shape}. "
                f"Check STFT params: nperseg={stft_nperseg}, "
                f"noverlap={stft_noverlap}, nfft={stft_nfft}"
            )        

    run_dir = ROOT / paths_cfg["outputs"]["runs"] / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)

    np.save(run_dir / "frame_energies.npy", frame_energies)
    np.save(run_dir / "detections.npy", detections.astype(np.int32))
    np.save(run_dir / "fft_mag.npy", fft_mag)

    if stft_done and branch is not None:
        stage1_dir = run_dir / "stage1"
        stage1_dir.mkdir(parents=True, exist_ok=True)

        np.save(stage1_dir / "cnn_spectrogram.npy", branch.cnn_spectrogram)

        if stft_mode == "single":
            np.save(stage1_dir / "complex_stft.npy", branch.complex_stft)

        elif stft_mode == "dual":
            np.save(stage1_dir / "rx0_complex_stft.npy", branch.rx0.complex_stft)
            np.save(stage1_dir / "rx1_complex_stft.npy", branch.rx1.complex_stft)

        plt.figure(figsize=(12, 5))
        plt.imshow(branch.cnn_spectrogram, aspect="auto", origin="lower")
        plt.colorbar(label="normalized magnitude")
        plt.xlabel("time frame")
        plt.ylabel("frequency bin")
        plt.title(f"CNN Spectrogram ({stft_mode})")
        plt.tight_layout()
        plt.savefig(stage1_dir / "cnn_spectrogram.png", dpi=150)
        plt.close()

    summary = {
        "source_type": receiver_cfg["source_type"],
        "num_samples": int(iq_for_aoa.shape[-1]),
        "iq_shape": list(iq_for_aoa.shape),
        "gain_ref_rms": float(gain_estimate.ref_rms) if gain_estimate is not None else None,
        "gain_target_rms": float(gain_estimate.target_rms) if gain_estimate is not None else None,
        "gain_correction": float(gain_estimate.gain_correction) if gain_estimate is not None else None,
        "num_frames": int(len(frames)),
        "num_detections": int(np.sum(detections)),
        "noise_floor": float(detector.noise_floor),
        "threshold": float(detector.threshold),
        "detection_ratio": float(np.mean(detections)) if len(detections) > 0 else 0.0,
        "stft_done": stft_done,
        "stft_mode": stft_mode,
        "stft_nperseg": int(stft_nperseg),
        "stft_noverlap": int(stft_noverlap),
        "stft_nfft": int(stft_nfft),
        "stft_window": stft_window,
        "cnn_spectrogram_shape": cnn_spectrogram_shape,
        "coherence": coherence_value,
        "coherence_passed": coherence_passed,
        "phase_offset_rad": float(phase_estimate.phase_offset_rad) if phase_estimate is not None else None,
        "phase_offset_deg": float(phase_estimate.phase_offset_deg) if phase_estimate is not None else None,
        "phase_offset_coherence_like": float(phase_estimate.coherence_like) if phase_estimate is not None else None,
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
