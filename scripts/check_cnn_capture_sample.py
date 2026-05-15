from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROOT = Path("data/processed/cnn_capture")


def find_latest_sample(root: Path = DEFAULT_ROOT) -> Path:
    """
    data/processed/cnn_capture 아래에서 가장 최근 sample_*.npz 파일을 찾는다.
    """
    files = sorted(root.glob("*/*/sample_*.npz"), key=lambda p: p.stat().st_mtime)

    if not files:
        raise FileNotFoundError(
            f"No sample_*.npz found under: {root}\n"
            "먼저 CNN capture를 실행해서 sample 파일을 생성해야 한다."
        )

    return files[-1]


def resolve_input_path(path_text: str | None) -> Path:
    """
    입력이 없으면 최신 sample 파일을 찾고,
    입력이 파일이면 그대로 사용하고,
    입력이 디렉터리면 그 아래 최신 sample_*.npz를 찾는다.
    """
    if path_text is None:
        return find_latest_sample(DEFAULT_ROOT)

    path = Path(path_text)

    if path.is_file():
        return path

    if path.is_dir():
        files = sorted(path.glob("sample_*.npz"), key=lambda p: p.stat().st_mtime)
        if not files:
            files = sorted(path.glob("**/sample_*.npz"), key=lambda p: p.stat().st_mtime)

        if not files:
            raise FileNotFoundError(f"No sample_*.npz found under: {path}")

        return files[-1]

    raise FileNotFoundError(f"Input path does not exist: {path}")


def load_metadata(value: Any) -> dict[str, Any]:
    """
    npz 안의 metadata_json을 dict로 변환한다.
    """
    if isinstance(value, np.ndarray):
        value = value.item()

    if isinstance(value, bytes):
        value = value.decode("utf-8")

    if isinstance(value, str):
        return json.loads(value)

    raise TypeError(f"Unsupported metadata_json type: {type(value)}")


def array_report(name: str, arr: np.ndarray) -> None:
    """
    배열 shape, dtype, min/max, NaN/Inf 여부를 출력한다.
    """
    arr = np.asarray(arr)

    print()
    print(f"[{name}]")
    print(f"shape      : {arr.shape}")
    print(f"dtype      : {arr.dtype}")
    print(f"min        : {float(np.nanmin(arr)):.8g}")
    print(f"max        : {float(np.nanmax(arr)):.8g}")
    print(f"mean       : {float(np.nanmean(arr)):.8g}")
    print(f"std        : {float(np.nanstd(arr)):.8g}")
    print(f"has_nan    : {bool(np.isnan(arr).any())}")
    print(f"has_inf    : {bool(np.isinf(arr).any())}")


def check_expected_shapes(spectrogram: np.ndarray, cnn_input: np.ndarray) -> None:
    """
    현재 프로젝트 기준의 기본 shape를 가볍게 점검한다.
    STFT 설정을 바꾸면 time frame 수는 달라질 수 있으므로 warning만 출력한다.
    """
    print()
    print("[Shape Check]")

    if spectrogram.ndim != 2:
        print(f"[WARN] spectrogram should be 2D, got {spectrogram.ndim}D")
    else:
        print("[OK] spectrogram is 2D")

    if cnn_input.ndim != 3:
        print(f"[WARN] cnn_input should be 3D, got {cnn_input.ndim}D")
    else:
        print("[OK] cnn_input is 3D")

    if cnn_input.ndim == 3 and cnn_input.shape[-1] != 1:
        print(f"[WARN] cnn_input last dim should be 1, got {cnn_input.shape[-1]}")
    elif cnn_input.ndim == 3:
        print("[OK] cnn_input channel dim is 1")

    if spectrogram.shape + (1,) == cnn_input.shape:
        print("[OK] cnn_input shape matches spectrogram + channel dim")
    else:
        print(
            "[WARN] cnn_input shape does not match spectrogram + channel dim: "
            f"{spectrogram.shape} -> {cnn_input.shape}"
        )


def print_metadata_summary(metadata: dict[str, Any]) -> None:
    print()
    print("[Metadata]")
    print(f"label              : {metadata.get('label')}")
    print(f"center_freq        : {metadata.get('center_freq')}")
    print(f"center_freq_ghz    : {metadata.get('center_freq_ghz')}")
    print(f"sample_rate        : {metadata.get('sample_rate')}")
    print(f"block_size         : {metadata.get('block_size')}")
    print(f"rx_index           : {metadata.get('rx_index')}")
    print(f"detection_ratio    : {metadata.get('detection_ratio')}")
    print(f"scan_score_db      : {metadata.get('scan_score_db')}")
    print(f"spectrogram_shape  : {metadata.get('spectrogram_shape')}")
    print(f"cnn_input_shape    : {metadata.get('cnn_input_shape')}")
    print(f"created_at         : {metadata.get('created_at')}")


def save_preview_png(spectrogram: np.ndarray, output_path: Path) -> None:
    """
    spectrogram 확인용 PNG 저장.
    matplotlib이 없으면 이 옵션은 실패할 수 있다.
    """
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.imshow(spectrogram, aspect="auto", origin="lower")
    plt.title("CNN Capture Spectrogram")
    plt.xlabel("Time frame")
    plt.ylabel("Frequency bin")
    plt.colorbar(label="Normalized log magnitude")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check saved CNN capture .npz sample."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help=(
            "sample_*.npz 파일 경로 또는 세션 디렉터리. "
            "생략하면 data/processed/cnn_capture 아래 최신 sample을 검사한다."
        ),
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="spectrogram preview PNG를 sample 파일 옆에 저장한다.",
    )

    args = parser.parse_args()

    sample_path = resolve_input_path(args.path)

    print("=== CNN Capture Sample Check ===")
    print(f"file : {sample_path}")

    data = np.load(sample_path, allow_pickle=False)

    print()
    print("[Keys]")
    for key in data.files:
        print(f"- {key}")

    required_keys = ["spectrogram", "cnn_input", "metadata_json"]
    missing = [key for key in required_keys if key not in data.files]

    if missing:
        raise KeyError(f"Missing required keys: {missing}")

    spectrogram = data["spectrogram"]
    cnn_input = data["cnn_input"]
    metadata = load_metadata(data["metadata_json"])

    array_report("spectrogram", spectrogram)
    array_report("cnn_input", cnn_input)

    if "raw_iq" in data.files:
        raw_iq = data["raw_iq"]
        print()
        print("[raw_iq]")
        print(f"shape      : {raw_iq.shape}")
        print(f"dtype      : {raw_iq.dtype}")
        print(f"has_nan    : {bool(np.isnan(raw_iq).any())}")
        print(f"has_inf    : {bool(np.isinf(raw_iq).any())}")
    else:
        print()
        print("[raw_iq]")
        print("not saved")

    check_expected_shapes(spectrogram, cnn_input)
    print_metadata_summary(metadata)

    if args.save_png:
        png_path = sample_path.with_suffix(".png")
        save_preview_png(spectrogram, png_path)
        print()
        print(f"[PNG saved] {png_path}")

    print()
    print("=== Done ===")


if __name__ == "__main__":
    main()


"""최신 sample 자동 검사
PYTHONPATH=. python scripts/check_cnn_capture_sample.py

특정 파일 검사
PYTHONPATH=. python scripts/check_cnn_capture_sample.py data/processed/cnn_capture/test_sim/세션폴더/sample_000001.npz

특정 세션 폴더의 최신 sample 검사
PYTHONPATH=. python scripts/check_cnn_capture_sample.py data/processed/cnn_capture/test_sim/세션폴더

PNG까지 저장
PYTHONPATH=. python scripts/check_cnn_capture_sample.py --save-png

"""