from __future__ import annotations

import inspect
from typing import Any, Mapping


def _to_plain_dict(cfg: Any) -> dict[str, Any]:
    if isinstance(cfg, dict):
        return cfg

    try:
        return dict(cfg)
    except Exception:
        pass

    result: dict[str, Any] = {}

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


def _filter_kwargs_for_class(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(cls.__init__)
    accepted_params = set(signature.parameters.keys())
    accepted_params.discard("self")

    return {key: value for key, value in kwargs.items() if key in accepted_params}


def _build_instance(cls: type, kwargs: dict[str, Any]):
    filtered_kwargs = _filter_kwargs_for_class(cls, kwargs)
    return cls(**filtered_kwargs)


def _to_int_list(values: Any) -> list[int]:
    if values is None:
        return [0]

    if isinstance(values, (int, str)):
        return [int(values)]

    return [int(v) for v in values]


def build_receiver(receiver_cfg: Mapping[str, Any]):
    cfg = _to_plain_dict(receiver_cfg)

    source_type = str(cfg.get("source_type", "sim")).strip().lower()

    sample_rate = int(cfg.get("sample_rate", 5_000_000))
    center_freq = int(cfg.get("center_freq", 2_400_000_000))
    block_size = int(cfg.get("block_size", 16_384))
    num_samples = int(cfg.get("num_samples", block_size))

    if source_type == "sim":
        from src.receiver.sim_receiver import SimReceiver

        sim_cfg = _to_plain_dict(cfg.get("sim", {}) or {})

        num_channels = int(
            cfg.get(
                "num_channels",
                sim_cfg.get("num_channels", 1),
            )
        )

        kwargs = {
            "sample_rate": sample_rate,
            "center_freq": center_freq,
            "num_channels": num_channels,
            "block_size": block_size,
            "tone_freq_norm": sim_cfg.get("tone_freq_norm", 0.08),
            "noise_std": sim_cfg.get("noise_std", 0.08),
            "burst_amplitude": sim_cfg.get("burst_amplitude", 1.5),
            "burst_period": sim_cfg.get("burst_period", 4096),
            "burst_length": sim_cfg.get("burst_length", 768),
            "seed": sim_cfg.get("seed", 42),
            "channel_phase_offset_rad": sim_cfg.get("channel_phase_offset_rad", 0.0),
        }

        return _build_instance(SimReceiver, kwargs)

    if source_type == "file":
        from src.receiver.raw_file_receiver import RawFileReceiver

        file_path = cfg.get("file_path")

        if file_path is None:
            raise ValueError(
                "source_type is 'file', but 'file_path' is not set in receiver.yaml"
            )

        kwargs = {
            "file_path": file_path,
            "sample_rate": sample_rate,
            "center_freq": center_freq,
            "num_channels": int(cfg.get("num_channels", 1)),
            "block_size": block_size,
            "num_samples": num_samples,
        }

        return _build_instance(RawFileReceiver, kwargs)

    if source_type == "sdr":
        from src.receiver.pluto_receiver import PlutoReceiver

        sdr_cfg = _to_plain_dict(cfg.get("sdr", {}) or {})

        device = str(sdr_cfg.get("device", "pluto_plus")).strip().lower()

        if device not in {"pluto", "pluto+", "pluto_plus"}:
            raise ValueError(
                f"Unsupported SDR device: {device}. Currently expected Pluto+."
            )

        channels = _to_int_list(sdr_cfg.get("channels", [0, 1]))
        num_channels = len(channels)

        sdr_sample_rate = int(sdr_cfg.get("sample_rate", sample_rate))
        sdr_center_freq = int(sdr_cfg.get("center_freq", center_freq))
        sdr_block_size = int(sdr_cfg.get("block_size", block_size))
        sdr_num_samples = int(sdr_cfg.get("num_samples", num_samples))

        kwargs = {
            "uri": sdr_cfg.get("uri", "ip:192.168.2.1"),
            "sample_rate": sdr_sample_rate,
            "center_freq": sdr_center_freq,
            "num_channels": num_channels,
            "channels": channels,
            "gain_control_mode": sdr_cfg.get("gain_control_mode", "manual"),
            "gain": float(sdr_cfg.get("gain", 20)),
            "block_size": sdr_block_size,
            "num_samples": sdr_num_samples,
            "rf_bandwidth": int(
                sdr_cfg.get("rf_bandwidth", sdr_cfg.get("sample_rate", sample_rate))
            ),
            "warmup_reads": int(sdr_cfg.get("warmup_reads", 1)),
        }

        return _build_instance(PlutoReceiver, kwargs)

    raise ValueError(
        f"Unsupported source_type: {source_type}. "
        "Expected one of ['sim', 'file', 'sdr']."
    )