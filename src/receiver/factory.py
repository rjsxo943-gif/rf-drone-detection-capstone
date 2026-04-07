from __future__ import annotations

from typing import Any, Mapping


def _to_plain_dict(cfg: Any) -> dict:
    """
    DictConfig / dict / 유사 객체를 최대한 dict처럼 변환
    """
    if isinstance(cfg, dict):
        return cfg

    try:
        return dict(cfg)
    except Exception:
        pass

    result = {}
    for key in dir(cfg):
        if key.startswith("_"):
            continue
        try:
            value = getattr(cfg, key)
        except Exception:
            continue
        if callable(value):
            continue
        result[key] = value
    return result


def build_receiver(receiver_cfg: Mapping[str, Any]):
    """
    receiver.yaml 설정을 보고 receiver 인스턴스를 생성한다.
    현재는 sim만 실제 동작.
    """
    cfg = _to_plain_dict(receiver_cfg)
    source_type = str(cfg.get("source_type", "sim")).strip().lower()

    if source_type == "sim":
        from src.receiver.sim_receiver import SimReceiver

        sim_cfg = cfg.get("sim", {}) or {}

        return SimReceiver(
            sample_rate=cfg["sample_rate"],
            center_freq=cfg["center_freq"],
            tone_freq_norm=sim_cfg.get("tone_freq_norm", 0.08),
            noise_std=sim_cfg.get("noise_std", 0.08),
            burst_amplitude=sim_cfg.get("burst_amplitude", 1.5),
            burst_period=sim_cfg.get("burst_period", 4096),
            burst_length=sim_cfg.get("burst_length", 768),
            seed=sim_cfg.get("seed", 42),
        )

    if source_type == "file":
        raise NotImplementedError("source_type: file 는 다음 단계에서 연결 예정")

    if source_type == "sdr":
        raise NotImplementedError("source_type: sdr 는 실제 SDR 연결 단계에서 구현 예정")

    raise ValueError(
        f"Unsupported source_type: {source_type}. "
        "Expected one of ['sim', 'file', 'sdr']"
    )
