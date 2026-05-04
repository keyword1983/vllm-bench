#!/usr/bin/env python3
"""
vllm Benchmark Results Summarizer.

Scans an output directory for benchmark result JSON files and prints a
formatted performance metrics table. Optionally exports to CSV.

Usage:
    # Print table from default output/ directory
    python scripts/summarize_results.py

    # Specify a different directory
    python scripts/summarize_results.py --output-dir /tmp/bench_results

    # Export to CSV
    python scripts/summarize_results.py --csv output/summary.csv

    # Sort by throughput (descending)
    python scripts/summarize_results.py --sort-by throughput
"""
import argparse
import csv
import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NA = "N/A"

# Benchmark result JSON files follow this pattern:
#   {safe_model_name}_{input_len}-{output_len}_{num_prompts}.json
# The last two numeric groups separated by '-' and '_' are input/output/prompts.
_FILENAME_RE = re.compile(
    r"^.+_(\d+)-(\d+)_(\d+)\.json$"
)

# Non-result files to skip
_SKIP_FILES = {"engine_params.txt", "metrics_before.txt", "metrics_after.txt"}


# ---------------------------------------------------------------------------
# Group inference from input/output lengths
# ---------------------------------------------------------------------------

def infer_group(input_len: int, output_len: int) -> str:
    """Infer benchmark group name from input/output token lengths."""
    if input_len >= 10000:
        return "long_context"
    if input_len == 200 and output_len == 250:
        return "throughput"
    if input_len == 100 and output_len == 100:
        return "latency"
    return "custom"


# ---------------------------------------------------------------------------
# Metric formatting helpers
# ---------------------------------------------------------------------------

def _fmt(value, fmt: str = ".1f") -> str:
    """Format a numeric value, or return N/A if missing."""
    if value is None:
        return _NA
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return _NA


def _sort_key(row: dict, sort_by: str):
    """Return a sortable key for the given column; put N/A rows last."""
    if sort_by == "throughput":
        v = row.get("throughput")
        return (0, -float(v)) if v is not None else (1, 0)
    if sort_by == "input_len":
        return (0, row["input_len"])
    if sort_by == "output_len":
        return (0, row["output_len"])
    if sort_by == "num_prompts":
        return (0, row["n"])
    # default: group alphabetically, then input/output/n
    return (0, row["group"], row["input_len"], row["output_len"], row["n"])


# ---------------------------------------------------------------------------
# Core: scan directory and build rows
# ---------------------------------------------------------------------------

def load_results(output_dir: str) -> list:
    """
    Scan output_dir for benchmark result JSON files.
    Returns a list of row dicts with parsed metadata and metrics.
    """
    if not os.path.isdir(output_dir):
        print(f"❌ Directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for fname in sorted(os.listdir(output_dir)):
        # Skip non-JSON and known non-result files
        if not fname.endswith(".json"):
            continue
        if fname in _SKIP_FILES:
            continue

        m = _FILENAME_RE.match(fname)
        if not m:
            continue  # filename doesn't match expected pattern

        input_len  = int(m.group(1))
        output_len = int(m.group(2))
        n          = int(m.group(3))

        fpath = os.path.join(output_dir, fname)
        data: dict = {}
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ⚠️  Could not read {fname}: {e}", file=sys.stderr)

        # Group: prefer field in JSON, fall back to inference from lengths
        group = data.get("benchmark_id") or infer_group(input_len, output_len)

        rows.append({
            "filename":   fname,
            "group":      group,
            "input_len":  input_len,
            "output_len": output_len,
            "n":          n,
            "throughput": data.get("output_throughput"),
            "ttft":       data.get("mean_ttft_ms"),
            "tpot":       data.get("mean_tpot_ms"),
            "itl":        data.get("mean_itl_ms"),
            "e2el":       data.get("mean_e2el_ms"),
        })

    return rows


# ---------------------------------------------------------------------------
# Output: print table
# ---------------------------------------------------------------------------

def print_table(rows: list) -> None:
    """Print a formatted performance metrics table to stdout."""
    if not rows:
        print("⚠️  No benchmark result files found.")
        return

    header = (
        f"{'Group':<14} {'in':>7} {'out':>5} {'n':>4}  "
        f"{'Tput(tok/s)':>11}  {'TTFT(ms)':>9}  "
        f"{'TPOT(ms)':>9}  {'ITL(ms)':>8}  {'E2EL(ms)':>9}"
    )
    sep  = "=" * 85
    dash = "-" * 85

    print(f"\n📊 Performance Metrics Summary:")
    print(sep)
    print(header)
    print(dash)
    for row in rows:
        print(
            f"{row['group']:<14} {row['input_len']:>7} {row['output_len']:>5} {row['n']:>4}  "
            f"{_fmt(row['throughput']):>11}  {_fmt(row['ttft']):>9}  "
            f"{_fmt(row['tpot']):>9}  {_fmt(row['itl']):>8}  {_fmt(row['e2el']):>9}"
        )
    print(sep)
    print(f"\nTotal: {len(rows)} result(s)")


# ---------------------------------------------------------------------------
# Output: export CSV
# ---------------------------------------------------------------------------

def export_csv(rows: list, csv_path: str) -> None:
    """Export rows to a CSV file."""
    fieldnames = [
        "group", "input_len", "output_len", "num_prompts",
        "output_throughput_tok_s",
        "mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms", "mean_e2el_ms",
        "filename",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "group":                    row["group"],
                "input_len":                row["input_len"],
                "output_len":               row["output_len"],
                "num_prompts":              row["n"],
                "output_throughput_tok_s":  row["throughput"] if row["throughput"] is not None else "",
                "mean_ttft_ms":             row["ttft"]       if row["ttft"]       is not None else "",
                "mean_tpot_ms":             row["tpot"]       if row["tpot"]       is not None else "",
                "mean_itl_ms":              row["itl"]        if row["itl"]        is not None else "",
                "mean_e2el_ms":             row["e2el"]       if row["e2el"]       is not None else "",
                "filename":                 row["filename"],
            })
    print(f"\n💾 CSV exported → {os.path.abspath(csv_path)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize vllm benchmark result JSON files into a metrics table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/summarize_results.py
  python scripts/summarize_results.py --output-dir /tmp/bench_results
  python scripts/summarize_results.py --csv output/summary.csv
  python scripts/summarize_results.py --sort-by throughput
""",
    )
    p.add_argument(
        "--output-dir", default="output",
        help="Directory containing benchmark result JSON files (default: output)",
    )
    p.add_argument(
        "--csv", default=None, metavar="CSV_PATH",
        help="Export results to this CSV file path (optional)",
    )
    p.add_argument(
        "--sort-by",
        choices=["group", "input_len", "output_len", "num_prompts", "throughput"],
        default="group",
        help="Sort results by this column (default: group)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rows = load_results(args.output_dir)

    # Normalize sort_by to match row dict keys
    sort_col = "n" if args.sort_by == "num_prompts" else args.sort_by
    rows.sort(key=lambda r: _sort_key(r, sort_col))

    print_table(rows)

    if args.csv:
        export_csv(rows, args.csv)


if __name__ == "__main__":
    main()