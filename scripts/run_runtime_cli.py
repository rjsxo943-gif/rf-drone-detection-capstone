from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.cli import run_cli  # noqa: E402


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()