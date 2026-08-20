"""Collect atomic benchmark case results into CSV and JSONL tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _arguments() -> argparse.Namespace:
    """Read locations for case JSON records and the two summary tables.

    Returns:
        Parsed input root plus CSV and JSONL output paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--csv", default="results/summary.csv")
    parser.add_argument("--jsonl", default="results/summary.jsonl")
    return parser.parse_args()


def main() -> None:
    """Read all case JSON files and write combined CSV and JSONL summaries."""
    args = _arguments()
    records = []
    for path in sorted((Path(args.output_root) / "cases").rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["result_path"] = str(path)
        records.append(record)
    if not records:
        raise SystemExit("no case result files found")

    jsonl_path = Path(args.jsonl)
    csv_path = Path(args.csv)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    fields = sorted({key for record in records for key in record})
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in record.items()
                }
            )
    passed = sum(record.get("status") == "passed" for record in records)
    print(f"records={len(records)} passed={passed} failed={len(records) - passed}")


if __name__ == "__main__":
    main()
