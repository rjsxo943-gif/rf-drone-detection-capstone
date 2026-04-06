from pathlib import Path


def list_npy_files(root: str):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(root.rglob("*.npy"))
