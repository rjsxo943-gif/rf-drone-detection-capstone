from __future__ import annotations

import argparse
from collections import Counter

from src.core import load_all_configs
from src.ml import RF4Classifier
from src.preprocess.dc_blocker import remove_dc_offset

from scripts.debug_rf4_live_capture import (
    build_runtime_receiver,
    close_receiver,
    compute_canonical01_spectrogram,
    ensure_2d_iq,
    get_block_size_from_configs,
    get_sample_rate_from_configs,
    read_block,
    set_center_freq,
    spec_stats,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/ml/rf4_cnn_live2450_v2/best_model.pt")
    parser.add_argument("--center-freq", type=int, default=2450000000)
    parser.add_argument("--num-blocks", type=int, default=50)
    parser.add_argument("--rx-index", type=int, default=0)
    parser.add_argument("--drone-vote-threshold", type=int, default=3)
    parser.add_argument("--min-p99", type=float, default=0.65)
    parser.add_argument("--min-max", type=float, default=0.80)
    args = parser.parse_args()

    configs = load_all_configs()
    receiver = build_runtime_receiver(configs)

    block_size = get_block_size_from_configs(configs)
    sample_rate = get_sample_rate_from_configs(configs)

    classifier = RF4Classifier(
        checkpoint_path=args.model,
        general_threshold=0.50,
        drone_threshold=0.70,
    )

    final_votes = []
    raw_votes = []

    print("=== RF4 Voting Live Detection ===")
    print(f"model       : {args.model}")
    print(f"center_freq : {args.center_freq} Hz")
    print(f"sample_rate : {sample_rate}")
    print(f"block_size  : {block_size}")
    print(f"num_blocks  : {args.num_blocks}")
    print(f"vote_th     : {args.drone_vote_threshold}")
    print()

    try:
        set_center_freq(receiver, args.center_freq)

        # warmup
        for _ in range(5):
            read_block(receiver, block_size)

        for i in range(args.num_blocks):
            iq = read_block(receiver, block_size)
            iq = remove_dc_offset(iq, axis=-1)
            iq = ensure_2d_iq(iq)

            iq_1d = iq[args.rx_index]

            spec = compute_canonical01_spectrogram(
                iq_1d,
                nperseg=128,
                noverlap=96,
                nfft=128,
                window="hann",
                vmin=-40.0,
                vmax=40.0,
            )

            stats = spec_stats(spec)
            result = classifier.predict_array(spec)

            is_strong_drone = (
                result.class_name == "Drone-like"
                and stats["p99"] >= args.min_p99
                and stats["max"] >= args.min_max
            )

            if result.class_name == "Drone-like":
                final_class = "Drone-like" if is_strong_drone else "Background"
            else:
                final_class = result.final_class

            raw_votes.append(result.class_name)
            final_votes.append(final_class)

            print(
                f"[{i:04d}] "
                f"raw={result.class_name:10s} "
                f"final={final_class:10s} "
                f"conf={result.confidence:.4f} "
                f"p99={stats['p99']:.4f} "
                f"max={stats['max']:.4f}"
            )

    finally:
        close_receiver(receiver)

    raw_counts = Counter(raw_votes)
    final_counts = Counter(final_votes)

    drone_votes = final_counts.get("Drone-like", 0)
    detected = drone_votes >= args.drone_vote_threshold

    print()
    print("=== Voting Summary ===")
    print("raw votes:")
    for k, v in raw_counts.most_common():
        print(f"  {k}: {v}")

    print("final votes:")
    for k, v in final_counts.most_common():
        print(f"  {k}: {v}")

    print()
    print(f"drone_votes : {drone_votes}/{args.num_blocks}")
    print(f"threshold   : {args.drone_vote_threshold}")
    print(f"decision    : {'DRONE_CANDIDATE' if detected else 'NO_DRONE'}")


if __name__ == "__main__":
    main()
