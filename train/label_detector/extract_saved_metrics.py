import argparse
import csv
from pathlib import Path


METRIC_COLUMNS = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict, key: str) -> float:
    return float(row[key])


def find_best_row(rows: list[dict], sort_key: str) -> dict:
    return max(rows, key=lambda row: as_float(row, sort_key))


def print_row(label: str, row: dict) -> None:
    print(f"  {label}: epoch={row['epoch']}")
    print(f"    precision     = {as_float(row, 'metrics/precision(B)'):.5f}")
    print(f"    recall        = {as_float(row, 'metrics/recall(B)'):.5f}")
    print(f"    mAP@0.5       = {as_float(row, 'metrics/mAP50(B)'):.5f}")
    print(f"    mAP@0.5:0.95  = {as_float(row, 'metrics/mAP50-95(B)'):.5f}")


def find_result_files(root: Path) -> list[Path]:
    return sorted(root.glob("runs/detect_*/*/results.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Показывает уже сохранённые метрики YOLO без повторного обучения."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Корень проекта, где находится папка runs",
    )
    parser.add_argument(
        "--sort-by",
        default="metrics/mAP50-95(B)",
        choices=METRIC_COLUMNS,
        help="Метрика, по которой выбирать лучшую эпоху",
    )
    args = parser.parse_args()

    result_files = find_result_files(args.root)
    if not result_files:
        print("Файлы results.csv не найдены.")
        return

    for csv_path in result_files:
        rows = load_rows(csv_path)
        if not rows:
            continue

        run_dir = csv_path.parent
        best_weights = run_dir / "weights" / "best.pt"
        args_yaml = run_dir / "args.yaml"

        print(f"\n{run_dir}")
        print("-" * len(str(run_dir)))
        print_row("last epoch", rows[-1])
        print_row(f"best by {args.sort_by}", find_best_row(rows, args.sort_by))

        if best_weights.exists():
            print("  best.pt:", best_weights)
        if args_yaml.exists():
            print("  args.yaml:", args_yaml)


if __name__ == "__main__":
    main()
