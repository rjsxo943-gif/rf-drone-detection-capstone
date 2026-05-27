from __future__ import annotations

import argparse
from pathlib import Path

from src.ml import RF4Classifier


LABEL_DIRS = {
    "Background": "Background",
    "WiFi": "Wifi",
    "Bluetooth": "Bluetooth",
    "Drone-like": "Drone-like",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="outputs/ml/rf4_cnn_baseline_v2/best_model.pt",
    )
    parser.add_argument(
        "--root",
        default="data/processed/cnn_capture",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--general-threshold",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--drone-threshold",
        type=float,
        default=0.70,
    )

    args = parser.parse_args()
    

    root = Path(args.root)

    classifier = RF4Classifier(
        checkpoint_path=args.model,
        general_threshold=args.general_threshold,
        drone_threshold=args.drone_threshold,
    )

    total = 0
    correct = 0

    print("=== RF4 Classifier Smoke Test ===")
    print(f"model             : {args.model}")
    print(f"root              : {root}")
    print(f"general_threshold : {args.general_threshold}")
    print(f"drone_threshold   : {args.drone_threshold}")
    print()

    for true_label, folder_name in LABEL_DIRS.items():
        folder = root / folder_name
        files = sorted(folder.glob("*.npy"))[: args.max_per_class]

        class_total = 0
        class_correct = 0

        print(f"[{true_label}] files={len(files)}")

        for path in files:
            result = classifier.predict_file(path)

            is_correct = result.final_class == true_label

            total += 1
            class_total += 1

            if is_correct:
                correct += 1
                class_correct += 1

            mark = "OK" if is_correct else "MISS"

            print(
                f"  {mark} "
                f"pred={result.final_class:10s} "
                f"raw={result.class_name:10s} "
                f"conf={result.confidence:.4f} "
                f"thr={result.applied_threshold:.2f} "
                f"file={path.name}"
            )

        acc = class_correct / class_total if class_total else 0.0
        print(f"  class_acc={acc:.4f}")
        print()

    overall_acc = correct / total if total else 0.0

    print("=== Summary ===")
    print(f"correct: {correct}/{total}")
    print(f"acc    : {overall_acc:.4f}")


if __name__ == "__main__":
    main()


