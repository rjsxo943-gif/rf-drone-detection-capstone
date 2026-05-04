from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import resample_poly


DEFAULT_MAT_PATH = Path("data/external/mavic/Mavic_0.mat")
DEFAULT_OUT_DIR = Path("data/raw_iq/mavic")

ORIG_FS = 100_000_000      # 원본 Mavic mat 파일 sample rate: 100 MSPS라고 가정
TARGET_FS = 5_000_000      # 우리 파이프라인 기준 sample rate: 5 MSPS
BLOCK_SIZE = 16_384        # 우리 파이프라인 기준 block size

DECIM = ORIG_FS // TARGET_FS
WIDE_BLOCK_SIZE = BLOCK_SIZE * DECIM


def safe_offset_name(offset_mhz: float) -> str:
    """
    파일명에 넣기 좋은 offset 문자열 생성.
    예:
      -22.5 -> m22p5
      +17.5 -> p17p5
    """
    sign = "p" if offset_mhz >= 0 else "m"
    value = abs(offset_mhz)
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    text = text.replace(".", "p")
    return f"{sign}{text}"


def load_complex_iq_from_mat(dset, start_sample: int, num_samples: int) -> np.ndarray:
    """
    MATLAB v7.3 mat 내부의 uhd_samps에서 complex IQ를 읽는다.

    현재 Mavic_0.mat 기준:
    dset shape = (1, N)
    dtype fields = real, imag
    """
    end_sample = start_sample + num_samples
    raw = dset[0, start_sample:end_sample]

    if raw.dtype.fields is None:
        raise ValueError(
            f"지원하지 않는 dtype입니다: {raw.dtype}\n"
            "현재 코드는 real/imag field가 있는 MATLAB v7.3 complex 구조를 예상합니다."
        )

    if "real" not in raw.dtype.fields or "imag" not in raw.dtype.fields:
        raise ValueError(
            f"dtype fields에 real/imag가 없습니다: {raw.dtype.fields.keys()}"
        )

    iq = raw["real"].astype(np.float32) + 1j * raw["imag"].astype(np.float32)
    return iq.astype(np.complex64)


def extract_subband_block(
    wide_iq: np.ndarray,
    start_sample: int,
    offset_hz: float,
    orig_fs: int = ORIG_FS,
    target_fs: int = TARGET_FS,
) -> np.ndarray:
    """
    wideband IQ에서 특정 frequency offset 대역을 baseband로 이동한 뒤 5 MSPS로 다운샘플링한다.

    offset_hz:
      file center 기준으로 추출할 subband 중심.
      예: -22.5e6이면 원본의 -25~-20 MHz 대역이 0 Hz 근처로 내려온다.
    """
    if orig_fs % target_fs != 0:
        raise ValueError(f"orig_fs는 target_fs의 정수배여야 합니다: {orig_fs}, {target_fs}")

    decim = orig_fs // target_fs

    # DC offset 제거
    x = wide_iq.astype(np.complex64)
    x = x - np.mean(x)

    # 절대 sample index를 써야 block을 여러 개 이어도 mixer phase가 자연스럽게 이어짐
    n = np.arange(start_sample, start_sample + len(x), dtype=np.float64)

    # 원하는 offset 대역을 DC 근처로 이동
    mixer = np.exp(-1j * 2.0 * np.pi * offset_hz * n / orig_fs).astype(np.complex64)
    shifted = x * mixer

    # anti-aliasing filter 포함 decimation
    y = resample_poly(shifted, up=1, down=decim)

    return y.astype(np.complex64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a 5 MSPS subband from Mavic_0.mat wideband IQ."
    )

    parser.add_argument(
        "--mat-path",
        type=Path,
        default=DEFAULT_MAT_PATH,
        help="Path to Mavic_0.mat",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for extracted .npy and metadata .json",
    )

    parser.add_argument(
        "--offset-mhz",
        type=float,
        default=-22.5,
        help=(
            "Subband center offset from file center in MHz. "
            "Example: -22.5 means extracting approximately -25~-20 MHz."
        ),
    )

    parser.add_argument(
        "--start-sample",
        type=int,
        default=0,
        help="Start sample index in the original 100 MSPS file.",
    )

    parser.add_argument(
        "--num-output-blocks",
        type=int,
        default=100,
        help=(
            "Number of 5 MSPS pipeline blocks to generate. "
            "1 block = 16384 samples at 5 MSPS."
        ),
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional output .npy filename. If omitted, name is generated automatically.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mat_path: Path = args.mat_path
    out_dir: Path = args.out_dir
    offset_hz = args.offset_mhz * 1e6

    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.output_name is None:
        offset_name = safe_offset_name(args.offset_mhz)
        out_npy = out_dir / f"mavic_offset_{offset_name}MHz_5msps.npy"
    else:
        out_npy = out_dir / args.output_name

    out_json = out_npy.with_suffix(".json")

    num_output_blocks = int(args.num_output_blocks)
    total_wide_samples_needed = num_output_blocks * WIDE_BLOCK_SIZE

    print("=== Extract Mavic subband ===")
    print(f"mat_path              : {mat_path}")
    print(f"out_npy               : {out_npy}")
    print(f"out_json              : {out_json}")
    print(f"orig_fs               : {ORIG_FS}")
    print(f"target_fs             : {TARGET_FS}")
    print(f"decimation            : {DECIM}")
    print(f"block_size            : {BLOCK_SIZE}")
    print(f"wide_block_size       : {WIDE_BLOCK_SIZE}")
    print(f"offset_mhz            : {args.offset_mhz}")
    print(f"approx extracted band : {args.offset_mhz - 2.5:+.1f} ~ {args.offset_mhz + 2.5:+.1f} MHz")
    print(f"start_sample          : {args.start_sample}")
    print(f"num_output_blocks     : {num_output_blocks}")
    print(f"wide samples needed   : {total_wide_samples_needed}")
    print()

    extracted_blocks = []

    with h5py.File(mat_path, "r") as f:
        if "uhd_samps" not in f:
            raise KeyError(f"'uhd_samps' dataset not found. Available keys: {list(f.keys())}")

        dset = f["uhd_samps"]
        total_samples = dset.shape[1]

        print(f"dataset shape         : {dset.shape}")
        print(f"dataset dtype         : {dset.dtype}")
        print(f"total samples         : {total_samples}")
        print()

        for block_idx in range(num_output_blocks):
            wide_start = args.start_sample + block_idx * WIDE_BLOCK_SIZE
            wide_end = wide_start + WIDE_BLOCK_SIZE

            if wide_end > total_samples:
                print(
                    f"[stop] requested range exceeds file length: "
                    f"{wide_start} ~ {wide_end}, total={total_samples}"
                )
                break

            wide_iq = load_complex_iq_from_mat(
                dset=dset,
                start_sample=wide_start,
                num_samples=WIDE_BLOCK_SIZE,
            )

            narrow_iq = extract_subband_block(
                wide_iq=wide_iq,
                start_sample=wide_start,
                offset_hz=offset_hz,
                orig_fs=ORIG_FS,
                target_fs=TARGET_FS,
            )

            if len(narrow_iq) != BLOCK_SIZE:
                raise RuntimeError(
                    f"Unexpected narrow block length: {len(narrow_iq)} "
                    f"(expected {BLOCK_SIZE})"
                )

            extracted_blocks.append(narrow_iq)

            print(
                f"[block {block_idx:04d}] "
                f"wide {wide_start}~{wide_end} -> narrow {len(narrow_iq)} samples"
            )

    if len(extracted_blocks) == 0:
        raise RuntimeError("No blocks were extracted.")

    iq_out = np.concatenate(extracted_blocks).astype(np.complex64)

    np.save(out_npy, iq_out)

    metadata = {
        "source_mat_path": str(mat_path),
        "output_npy_path": str(out_npy),
        "original_sample_rate_hz": ORIG_FS,
        "target_sample_rate_hz": TARGET_FS,
        "decimation": DECIM,
        "pipeline_block_size": BLOCK_SIZE,
        "wide_block_size": WIDE_BLOCK_SIZE,
        "offset_hz": offset_hz,
        "offset_mhz": args.offset_mhz,
        "approx_band_start_mhz": args.offset_mhz - 2.5,
        "approx_band_end_mhz": args.offset_mhz + 2.5,
        "start_sample": args.start_sample,
        "num_extracted_blocks": len(extracted_blocks),
        "num_output_samples": int(iq_out.shape[0]),
        "output_dtype": str(iq_out.dtype),
        "output_shape": list(iq_out.shape),
    }

    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2, ensure_ascii=False)

    print()
    print("=== Done ===")
    print(f"saved npy : {out_npy}")
    print(f"saved json: {out_json}")
    print(f"output shape: {iq_out.shape}")
    print(f"output duration: {len(iq_out) / TARGET_FS:.6f} sec")
    print(f"pipeline blocks: {len(extracted_blocks)}")


if __name__ == "__main__":
    main()