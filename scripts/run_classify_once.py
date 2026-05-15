from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_all_configs
from src.receiver.factory import build_receiver
from src.preprocess import remove_dc_offset, normalize_iq, select_rx
from src.detect.energy_detector import EnergyDetector
from src.features.spectrogram import compute_stft_branch
from src.ml.inference import build_cnn_classifier
from src.ui.result_plotter import save_energy_plot


def _get_num_samples(receiver_cfg: dict) -> int:
    return int(receiver_cfg.get("num_samples", receiver_cfg.get("block_size", 16384)))


def _make_detector(detect_cfg: dict) -> EnergyDetector:
    energy_cfg = detect_cfg["energy_detector"]

    return EnergyDetector(
        mode=energy_cfg.get("mode", "block_median"),
        threshold_multiplier=energy_cfg.get("threshold_multiplier", 5.0),
        frame_size=energy_cfg.get("frame_size", 1024),
        hop_size=energy_cfg.get("hop_size", 512),
        window=energy_cfg.get("window", "hann"),
        method=energy_cfg.get("method", "time_power"),
        min_detection_ratio=energy_cfg.get("min_detection_ratio", 0.05),
        calibration_num_blocks=energy_cfg.get("calibration_blocks", 20),
        require_calibration=False,
    )


def _save_spectrogram_image(spec: np.ndarray, save_path: Path, title: str) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.imshow(spec, aspect="auto", origin="lower")
    plt.colorbar(label="normalized magnitude")
    plt.xlabel("time frame")
    plt.ylabel("frequency bin")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _probabilities_to_dict(class_names: list[str], probabilities: list[float]) -> dict[str, float]:
    return {
        class_name: float(prob)
        for class_name, prob in zip(class_names, probabilities)
    }


def main() -> None:
    cfg = load_all_configs(ROOT / "configs")

    receiver_cfg = cfg["receiver"]
    detect_cfg = cfg["detect"]
    ml_cfg = cfg["ml"]

    stft_cfg = ml_cfg["stft"]
    cnn_input_cfg = ml_cfg.get("cnn_input", {})
    inference_cfg = ml_cfg.get("inference", {})

    run_dir = ROOT / "outputs" / "runs" / "latest_classify"
    stage_dir = run_dir / "stage1"
    run_dir.mkdir(parents=True, exist_ok=True)
    stage_dir.mkdir(parents=True, exist_ok=True)

    num_samples = _get_num_samples(receiver_cfg)
    rx_index = int(cnn_input_cfg.get("rx_index", 0))

    receiver = build_receiver(receiver_cfg)

    try:
        # 1) IQ block 수신
        iq = receiver.read_samples(num_samples)
    finally:
        if hasattr(receiver, "close"):
            receiver.close()

    # 2) 기본 전처리
    iq = remove_dc_offset(iq)
    iq = normalize_iq(iq)

    # 3) CNN 입력용 RX 채널 선택
    #    단일 채널이면 RX0만 사용하고,
    #    2채널이면 configs/ml.yaml의 cnn_input.rx_index 기준으로 선택한다.
    cnn_iq = select_rx(iq, rx_index=rx_index)

    # 4) Energy Detection
    #    이번 classification-only pipeline에서는 energy 결과를 참고 정보로 저장하고,
    #    CNN 분류는 항상 수행한다.
    detector = _make_detector(detect_cfg)
    frame_energies = detector.compute_frame_energies(cnn_iq)
    frame_detections = detector.detect_frame_energies(frame_energies)

    if frame_detections.size == 0:
        detection_ratio = 0.0
        energy_detected = False
    else:
        detection_ratio = float(np.mean(frame_detections))
        energy_detected = detection_ratio >= detector.min_detection_ratio

    # 5) STFT spectrogram 생성
    branch = compute_stft_branch(
        iq_block=cnn_iq,
        sample_rate=float(receiver_cfg["sample_rate"]),
        nperseg=int(stft_cfg.get("nperseg", 128)),
        noverlap=int(stft_cfg.get("noverlap", 96)),
        nfft=int(stft_cfg.get("nfft", 128)),
        window=str(stft_cfg.get("window", "hann")),
        apply_fftshift=bool(stft_cfg.get("fftshift", True)),
    )

    cnn_spectrogram = branch.cnn_spectrogram.astype(np.float32)
    actual_shape = tuple(cnn_spectrogram.shape)

    expected_shape = (
        int(stft_cfg.get("expected_freq_bins", actual_shape[0])),
        int(stft_cfg.get("expected_time_frames", actual_shape[1])),
    )

    if actual_shape != expected_shape:
        raise RuntimeError(
            f"Unexpected CNN spectrogram shape: {actual_shape}, "
            f"expected={expected_shape}. "
            f"Check configs/ml.yaml STFT settings."
        )

    # 6) CNN 분류
    classifier = build_cnn_classifier(ml_cfg)
    cnn_result = classifier.predict(cnn_spectrogram)

    class_names = list(ml_cfg["class_names"])
    probability_dict = _probabilities_to_dict(
        class_names=class_names,
        probabilities=cnn_result.probabilities,
    )

    backend = str(inference_cfg.get("backend", "dummy")).lower().strip()
    model_path = inference_cfg.get("model_path", None)
    using_untrained_torch_model = backend == "torch" and model_path in [None, "", "null", "None"]

    # 7) 결과 저장
    np.save(run_dir / "frame_energies.npy", frame_energies.astype(np.float32))
    np.save(run_dir / "frame_detections.npy", frame_detections.astype(np.int32))
    np.save(stage_dir / "cnn_spectrogram.npy", cnn_spectrogram)
    np.save(stage_dir / "complex_stft.npy", branch.complex_stft)

    save_energy_plot(
        energies=frame_energies,
        threshold=detector.threshold,
        detections=frame_detections,
        save_path=run_dir / "energy_plot.png",
        title="Classification-only Energy Detector Output",
    )

    _save_spectrogram_image(
        spec=cnn_spectrogram,
        save_path=stage_dir / "cnn_spectrogram.png",
        title="CNN Spectrogram for RF Classification",
    )

    summary = {
        "mode": "classification_only",
        "source_type": receiver_cfg["source_type"],
        "center_freq": int(receiver_cfg.get("center_freq", 0)),
        "sample_rate": int(receiver_cfg["sample_rate"]),
        "num_samples": int(num_samples),
        "iq_shape": list(iq.shape),
        "rx_index": int(rx_index),
        "cnn_iq_shape": list(cnn_iq.shape),
        "energy_detected": bool(energy_detected),
        "detection_ratio": float(detection_ratio),
        "num_energy_frames": int(len(frame_energies)),
        "num_detected_frames": int(np.sum(frame_detections)),
        "noise_floor": float(detector.noise_floor),
        "threshold": float(detector.threshold),
        "stft_nperseg": int(stft_cfg.get("nperseg", 128)),
        "stft_noverlap": int(stft_cfg.get("noverlap", 96)),
        "stft_nfft": int(stft_cfg.get("nfft", 128)),
        "stft_window": str(stft_cfg.get("window", "hann")),
        "cnn_spectrogram_shape": list(cnn_spectrogram.shape),
        "cnn_backend": backend,
        "cnn_model_path": model_path,
        "using_untrained_torch_model": bool(using_untrained_torch_model),
        "class_name": cnn_result.class_name,
        "class_index": int(cnn_result.class_index),
        "confidence": float(cnn_result.confidence),
        "probabilities": probability_dict,
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== RF Classification-only Pipeline Result ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if using_untrained_torch_model:
        print()
        print("[WARN] Torch backend is running with model_path=None.")
        print("[WARN] This means the model is randomly initialized.")
        print("[WARN] Pipeline connection is valid, but classification result is not meaningful.")

    print()
    print(f"saved to: {run_dir}")


if __name__ == "__main__":
    main()
