from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np


EPS = 1e-12


def wrap_phase_rad(phase: float | np.ndarray) -> float | np.ndarray:
    """phase를 -pi ~ +pi 범위로 정리."""
    return (phase + np.pi) % (2.0 * np.pi) - np.pi


def circular_distance_rad(a: np.ndarray, b: float) -> np.ndarray:
    """
    원형 위상 거리 계산.
    예: +179도와 -179도는 358도가 아니라 2도 차이로 처리.
    """
    a = np.asarray(a, dtype=np.float64)
    return np.abs(np.angle(np.exp(1j * (a - b))))


def circular_weighted_mean_rad(phases: np.ndarray, weights: np.ndarray) -> float:
    """coherence 가중 원형 평균."""
    phases = np.asarray(phases, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if phases.size == 0:
        raise ValueError("phases is empty.")

    if np.sum(weights) <= 0:
        weights = np.ones_like(phases)

    z = np.sum(weights * np.exp(1j * phases))
    return float(np.angle(z))


def normalize_table_keys(table: dict) -> dict[int, dict]:
    """JSON 로드 후 gain key를 int로 변환."""
    return {int(k): v for k, v in table.items()}


def to_jsonable_table(table: dict[int, dict]) -> dict[str, dict]:
    """JSON 저장용으로 gain key를 str로 변환."""
    return {str(int(k)): v for k, v in table.items()}


def compute_block_phase_and_coherence(
    ch0: np.ndarray,
    ch1: np.ndarray,
) -> tuple[float, float]:
    """
    한 block에서 RX1 - RX0 phase offset과 coherence-like 계산.

    phase = angle(mean(ch1 * conj(ch0)))
    """
    ch0 = np.asarray(ch0, dtype=np.complex64)
    ch1 = np.asarray(ch1, dtype=np.complex64)

    if ch0.shape != ch1.shape:
        raise ValueError(
            f"ch0 and ch1 must have same shape. got {ch0.shape}, {ch1.shape}"
        )

    cross = np.mean(ch1 * np.conj(ch0))
    p0 = np.mean(np.abs(ch0) ** 2)
    p1 = np.mean(np.abs(ch1) ** 2)

    phase = float(np.angle(cross))
    coherence = float(np.abs(cross) / np.sqrt(p0 * p1 + EPS))

    return phase, coherence


def select_dominant_cluster(
    phases: np.ndarray,
    coherences: np.ndarray,
    cluster_window_deg: float = 5.0,
) -> np.ndarray:
    """
    phase 분포에서 가장 많은 블럭이 모인 dominant cluster 선택.

    기준:
    - 각 phase를 center 후보로 놓고 ±cluster_window_deg 안의 블럭 수 계산
    - count가 가장 큰 cluster 선택
    - count가 같으면 coherence 합이 큰 cluster 선택
    """
    phases = np.asarray(phases, dtype=np.float64)
    coherences = np.asarray(coherences, dtype=np.float64)

    if phases.size == 0:
        raise ValueError("phases is empty.")

    if phases.shape != coherences.shape:
        raise ValueError("phases and coherences must have same shape.")

    window_rad = np.deg2rad(cluster_window_deg)

    best_mask = None
    best_count = -1
    best_score = -1.0

    for center in phases:
        dist = circular_distance_rad(phases, center)
        mask = dist <= window_rad

        count = int(np.sum(mask))
        score = float(np.sum(coherences[mask]))

        if count > best_count or (count == best_count and score > best_score):
            best_count = count
            best_score = score
            best_mask = mask

    if best_mask is None:
        raise RuntimeError("Failed to select dominant phase cluster.")

    return best_mask


def dominant_cluster_phase(
    blocks_ch0: list[np.ndarray],
    blocks_ch1: list[np.ndarray],
    coherence_threshold: float = 0.50,
    cluster_window_deg: float = 5.0,
) -> dict:
    """
    단일 gain에서 phase offset 대표값 계산.

    단계:
    1. block별 phase/coherence 계산
    2. coherence_threshold 이상인 block만 1차 통과
    3. phase 분포에서 dominant cluster 선택
    4. cluster block만 coherence-weighted circular mean
    """
    phases: list[float] = []
    coherences: list[float] = []

    for ch0, ch1 in zip(blocks_ch0, blocks_ch1):
        phase, coherence = compute_block_phase_and_coherence(ch0, ch1)

        if coherence >= coherence_threshold:
            phases.append(phase)
            coherences.append(coherence)

    if len(phases) == 0:
        raise ValueError(
            "유효 블럭 없음 - coherence_threshold를 낮추거나 재측정 필요"
        )

    phases_np = np.asarray(phases, dtype=np.float64)
    coherences_np = np.asarray(coherences, dtype=np.float64)

    cluster_mask = select_dominant_cluster(
        phases=phases_np,
        coherences=coherences_np,
        cluster_window_deg=cluster_window_deg,
    )

    cluster_phases = phases_np[cluster_mask]
    cluster_coherences = coherences_np[cluster_mask]

    final_phase = circular_weighted_mean_rad(
        phases=cluster_phases,
        weights=cluster_coherences,
    )

    phase_errors = circular_distance_rad(cluster_phases, final_phase)
    phase_std = float(np.std(phase_errors))

    valid_blocks = int(len(phases_np))
    cluster_blocks = int(np.sum(cluster_mask))
    cluster_ratio = float(cluster_blocks / valid_blocks)

    return {
        "phase": float(final_phase),
        "phase_deg": float(np.rad2deg(final_phase)),
        "phase_std": phase_std,
        "phase_std_deg": float(np.rad2deg(phase_std)),
        "coherence_mean": float(np.mean(cluster_coherences)),
        "coherence_median": float(np.median(cluster_coherences)),
        "coherence_min": float(np.min(cluster_coherences)),
        "coherence_max": float(np.max(cluster_coherences)),
        "valid_blocks": valid_blocks,
        "cluster_blocks": cluster_blocks,
        "cluster_ratio": cluster_ratio,
        "cluster_window_deg": float(cluster_window_deg),
        "coherence_threshold": float(coherence_threshold),
    }


def evaluate_table_entry(meta: dict, gain: int) -> str:
    """
    gain별 phase table entry 품질 판정.

    OK:
    - phase_std < 3 deg
    - valid_blocks >= 100
    - cluster_ratio >= 0.80
    - coherence_median >= 0.70
    """
    std_deg = float(meta["phase_std_deg"])
    valid = int(meta["valid_blocks"])
    cluster = int(meta["cluster_blocks"])
    ratio = float(meta["cluster_ratio"])
    coh_med = float(meta["coherence_median"])

    if std_deg < 3.0 and valid >= 100 and ratio >= 0.80 and coh_med >= 0.70:
        status = "OK"
    elif std_deg < 7.0 and valid >= 50 and ratio >= 0.60 and coh_med >= 0.55:
        status = "WARNING - 재측정 권장"
    else:
        status = "ERROR - 재측정 필요"

    print(
        f"  gain={gain:3d} | phase={meta['phase_deg']:+7.2f}° "
        f"| std={std_deg:5.2f}° "
        f"| coh_med={coh_med:.3f} "
        f"| cluster={cluster}/{valid} ({ratio * 100:5.1f}%) "
        f"| {status}"
    )

    return status


def build_gain_phase_table(
    gain_list: list[int],
    collect_fn: Callable,
    output_path: str | Path = "configs/calibration/gain_phase_table.json",
    reference_gain: int = 30,
    total_blocks: int = 200,
    discard_blocks: int = 30,
    coherence_threshold: float = 0.50,
    cluster_window_deg: float = 5.0,
    metadata: dict | None = None,
) -> dict[int, dict]:
    """
    gain별 phase delta table 제작.

    collect_fn:
      collect_fn(gain, n_blocks) -> tuple[list[np.ndarray], list[np.ndarray]]
    """
    if reference_gain not in gain_list:
        raise ValueError(
            f"reference_gain={reference_gain} must be included in gain_list."
        )

    table: dict[int, dict] = {}
    errors: list[int] = []

    print("=" * 60)
    print("GAIN PHASE TABLE 제작 시작")
    print("=" * 60)
    print(f"gain_list           : {gain_list}")
    print(f"reference_gain      : {reference_gain}")
    print(f"total_blocks        : {total_blocks}")
    print(f"discard_blocks      : {discard_blocks}")
    print(f"used_blocks_target  : {total_blocks - discard_blocks}")
    print(f"coherence_threshold : {coherence_threshold}")
    print(f"cluster_window_deg  : {cluster_window_deg}")
    print("=" * 60)

    for gain in gain_list:
        gain = int(gain)
        print(f"\n[gain={gain}] 수집 중...")

        blocks_ch0, blocks_ch1 = collect_fn(gain, n_blocks=total_blocks)

        if len(blocks_ch0) != len(blocks_ch1):
            raise ValueError("blocks_ch0 and blocks_ch1 must have same length.")

        if len(blocks_ch0) <= discard_blocks:
            raise ValueError(
                f"Not enough blocks. got {len(blocks_ch0)}, "
                f"discard_blocks={discard_blocks}"
            )

        blocks_ch0_used = blocks_ch0[discard_blocks:]
        blocks_ch1_used = blocks_ch1[discard_blocks:]

        try:
            meta = dominant_cluster_phase(
                blocks_ch0=blocks_ch0_used,
                blocks_ch1=blocks_ch1_used,
                coherence_threshold=coherence_threshold,
                cluster_window_deg=cluster_window_deg,
            )
        except ValueError as exc:
            print(f"  [ERROR] {exc}")
            errors.append(gain)
            continue

        meta["gain"] = gain
        meta["total_blocks"] = int(total_blocks)
        meta["discard_blocks"] = int(discard_blocks)
        meta["used_blocks_target"] = int(total_blocks - discard_blocks)

        status = evaluate_table_entry(meta, gain)
        meta["quality"] = status

        if status.startswith("ERROR"):
            errors.append(gain)

        table[gain] = meta

    if reference_gain not in table:
        raise RuntimeError(
            f"reference_gain={reference_gain} 측정 실패 - 테이블 원점 없음"
        )

    ref_phase = float(table[reference_gain]["phase"])
    ref_phase_deg = float(table[reference_gain]["phase_deg"])

    for gain, meta in table.items():
        phase_delta = float(wrap_phase_rad(meta["phase"] - ref_phase))
        meta["phase_delta"] = phase_delta
        meta["phase_delta_deg"] = float(np.rad2deg(phase_delta))
        meta["reference_gain"] = int(reference_gain)
        meta["reference_phase_deg"] = ref_phase_deg

    output = {
        "table_type": "gain_dependent_phase_delta_table",
        "reference_gain": int(reference_gain),
        "reference_phase_deg": ref_phase_deg,
        "reference_phase_rad": ref_phase,
        "total_blocks": int(total_blocks),
        "discard_blocks": int(discard_blocks),
        "used_blocks_target": int(total_blocks - discard_blocks),
        "coherence_threshold": float(coherence_threshold),
        "cluster_window_deg": float(cluster_window_deg),
        "metadata": metadata or {},
        "gain_table": to_jsonable_table(table),
        "errors": errors,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    if errors:
        print(f"WARNING: 재측정 필요한 gain → {errors}")
    else:
        print("모든 gain 측정 완료")
    print(f"테이블 저장 완료: {output_path}")
    print("=" * 60)

    return table


def load_gain_phase_table(path: str | Path) -> dict:
    """gain phase table JSON 로드."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    if "gain_table" not in data:
        raise ValueError("Invalid gain phase table: missing 'gain_table'.")

    data["gain_table"] = normalize_table_keys(data["gain_table"])
    return data


def interpolate_phase_delta(
    table: dict,
    target_gain: int | float,
) -> tuple[float, float]:
    """
    target_gain의 phase_delta 반환.

    반환:
      phase_delta_rad, uncertainty_rad
    """
    table = normalize_table_keys(table)
    target_gain = float(target_gain)

    gains = sorted(table.keys())

    if target_gain < gains[0] or target_gain > gains[-1]:
        raise ValueError(
            f"gain={target_gain} 테이블 범위 밖 ({gains[0]}~{gains[-1]})"
        )

    target_gain_int = int(target_gain)

    if target_gain.is_integer() and target_gain_int in table:
        entry = table[target_gain_int]
        return float(entry["phase_delta"]), float(entry["phase_std"])

    lower = max(g for g in gains if g <= target_gain)
    upper = min(g for g in gains if g >= target_gain)

    if lower == upper:
        entry = table[lower]
        return float(entry["phase_delta"]), float(entry["phase_std"])

    delta_low = float(table[lower]["phase_delta"])
    delta_high = float(table[upper]["phase_delta"])
    std_low = float(table[lower]["phase_std"])
    std_high = float(table[upper]["phase_std"])

    w_low = (upper - target_gain) / (upper - lower)
    w_high = (target_gain - lower) / (upper - lower)

    phase_interp = np.angle(
        w_low * np.exp(1j * delta_low)
        + w_high * np.exp(1j * delta_high)
    )

    uncertainty = w_low * std_low + w_high * std_high

    if uncertainty > np.deg2rad(7.0):
        print(
            f"WARNING: gain={target_gain} 보간 불확실도 높음 "
            f"({np.rad2deg(uncertainty):.1f}°) - AoA 신뢰도 낮을 수 있음"
        )

    return float(phase_interp), float(uncertainty)


def get_phase_offset_to_apply(
    current_ref_phase_offset: float,
    table: dict,
    current_gain: int | float,
) -> tuple[float, float]:
    """
    현장 기준 gain 캘리브레이션값 + gain별 delta 적용.

    current_ref_phase_offset:
      현장 시작 시 reference_gain에서 얻은 phase offset [rad]

    반환:
      phase_offset_to_apply_rad, uncertainty_rad
    """
    delta, uncertainty = interpolate_phase_delta(
        table=table,
        target_gain=current_gain,
    )

    phase_offset_to_apply = wrap_phase_rad(current_ref_phase_offset + delta)

    return float(phase_offset_to_apply), float(uncertainty)
