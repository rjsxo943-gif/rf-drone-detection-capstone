from pathlib import Path
import csv

ROOT = Path("outputs/datasets/rf_binary_v1_att10_20260528")
OUT = Path("data/processed/external_binary_20260528_manifest.csv")

def label_from_path(path: Path):
    s = str(path).lower()
    if "_drone_" in s:
        return "Drone", 1
    if any(k in s for k in ["notdrone", "background", "wifi", "bluetooth"]):
        return "NonDrone", 0
    return None, None

rows = []
for p in sorted(ROOT.rglob("*.npy")):
    label, label_id = label_from_path(p)
    if label is None:
        continue

    rows.append({
        "path": str(p.resolve()),
        "label_id": label_id,
        "label": label,
        "session": p.parent.name,
        "split": "external",
    })

OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["path", "label_id", "label", "session", "split"],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"[OK] saved: {OUT}")
print(f"total: {len(rows)}")
print(f"Drone: {sum(r['label']=='Drone' for r in rows)}")
print(f"NonDrone: {sum(r['label']=='NonDrone' for r in rows)}")