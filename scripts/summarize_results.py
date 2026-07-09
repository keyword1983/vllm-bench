#!/usr/bin/env python3
"""
vllm Benchmark Results Summarizer.

Scans an output directory for benchmark result JSON files and prints a
formatted performance metrics table. Optionally exports to CSV, HTML report, and Markdown.
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

_FILENAME_RE = re.compile(
    r"^.+_(\d+)-(\d+)_(\d+)\.json$"
)

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
    return (0, row["group"], row["input_len"], row["output_len"], row["n"])


# ---------------------------------------------------------------------------
# Core: scan directory and load results
# ---------------------------------------------------------------------------

def load_results(output_dir: str) -> list:
    """Scan output_dir for benchmark result JSON files."""
    if not os.path.isdir(output_dir):
        print(f"❌ Directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for fname in sorted(os.listdir(output_dir)):
        if not fname.endswith(".json"):
            continue
        if fname in _SKIP_FILES:
            continue

        m = _FILENAME_RE.match(fname)
        if not m:
            continue

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
# Output: export Markdown Report with Mermaid charts
# ---------------------------------------------------------------------------

def export_markdown_report(rows: list, md_path: str, output_dir: str) -> None:
    """Generates a premium Markdown report featuring table and Mermaid line chart."""
    os.makedirs(os.path.dirname(os.path.abspath(md_path)), exist_ok=True)
    
    # Try to load diagnostics and engine params
    diagnostics = ""
    diag_path = os.path.join(output_dir, "diagnostics.txt")
    if os.path.exists(diag_path):
        with open(diag_path, "r", encoding="utf-8") as f:
            diagnostics = f.read()

    engine_params = {}
    engine_path = os.path.join(output_dir, "engine_params.txt")
    if os.path.exists(engine_path):
        try:
            with open(engine_path, "r", encoding="utf-8") as f:
                # First line is a comment in run_benchmark.py
                lines = f.readlines()
                json_str = "".join([l for l in lines if not l.strip().startswith("#")])
                engine_params = json.loads(json_str)
        except Exception:
            pass

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# vLLM Benchmark Performance Report\n\n")
        
        # Section 1: Server Config
        f.write("## 📡 Model & Engine configuration\n\n")
        if engine_params:
            f.write(f"- **Model ID**: `{engine_params.get('model_name', 'N/A')}`\n")
            f.write(f"- **Tokenizer**: `{engine_params.get('tokenizer', 'N/A')}`\n")
            f.write(f"- **Context Length limit**: `{engine_params.get('max_model_len', 'N/A')}` tokens\n\n")
        else:
            f.write("No server engine information was available during the test.\n\n")

        # Section 2: Diagnostics
        if diagnostics:
            f.write("## 🔍 Prometheus metrics analysis & diagnostics\n\n")
            f.write("> [!NOTE]\n")
            f.write("> Below are observations based on vLLM server metrics collected before and after benchmark:\n>\n")
            for line in diagnostics.splitlines():
                if line.strip() and not line.startswith("#"):
                    f.write(f"> {line}\n")
            f.write("\n")

        # Section 3: Performance chart (using Mermaid xychart-beta)
        # Filters rows where throughput is not None
        valid_rows = [r for r in rows if r["throughput"] is not None]
        if valid_rows:
            f.write("## 📊 Performance visualization (Throughput)\n\n")
            f.write("```mermaid\n")
            f.write("xychart-beta\n")
            f.write("    title \"Throughput vs Concurrency (num_prompts)\"\n")
            
            x_labels = [f"\"{r['group']}(n={r['n']})\"" for r in valid_rows]
            f.write(f"    x-axis [{', '.join(x_labels)}]\n")
            f.write("    y-axis \"Throughput (tokens/s)\"\n")
            
            tput_values = [f"{r['throughput']:.1f}" for r in valid_rows]
            f.write(f"    bar [{', '.join(tput_values)}]\n")
            f.write("```\n\n")

        # Section 4: Performance Table
        f.write("## 📋 Detailed Performance results\n\n")
        f.write("| Group | Input | Output | Concurrency (n) | Throughput (tok/s) | TTFT (ms) | TPOT (ms) | ITL (ms) | E2EL (ms) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in rows:
            f.write(
                f"| {r['group']} | {r['input_len']} | {r['output_len']} | {r['n']} "
                f"| {_fmt(r['throughput'])} | {_fmt(r['ttft'])} | {_fmt(r['tpot'])} | {_fmt(r['itl'])} | {_fmt(r['e2el'])} |\n"
            )
            
    print(f"💾 Markdown report generated → {os.path.abspath(md_path)}")


# ---------------------------------------------------------------------------
# Output: export HTML Chart Report
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>vLLM Benchmark serving report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background-color: #0f172a; color: #f8fafc; padding: 40px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #38bdf8; border-bottom: 1px solid #334155; padding-bottom: 10px; margin-bottom: 30px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; margin-top: 30px; }
        .card { background-color: #1e293b; border-radius: 12px; padding: 24px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); border: 1px solid #334155; }
        .card h2 { margin-top: 0; color: #e2e8f0; font-size: 1.25rem; border-bottom: 1px solid #475569; padding-bottom: 8px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; text-align: left; }
        th, td { padding: 12px; border-bottom: 1px solid #334155; }
        th { color: #94a3b8; font-weight: 600; }
        tr:hover { background-color: #334155; }
        .alert { background: rgba(56, 189, 248, 0.1); border-left: 4px solid #38bdf8; padding: 16px; border-radius: 4px; margin-bottom: 30px; line-height: 1.6; }
        .alert-title { font-weight: bold; color: #38bdf8; margin-bottom: 8px; }
        pre { background-color: #020617; padding: 16px; border-radius: 8px; overflow-x: auto; color: #38bdf8; border: 1px solid #1e293b; font-family: monospace; font-size: 0.9rem; line-height: 1.5; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 vLLM Benchmark Performance Report</h1>
        <div class="alert">
            <div class="alert-title">📡 Server & Model information</div>
            <div id="model-info">No details available.</div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>Throughput (tokens/s)</h2>
                <canvas id="throughputChart"></canvas>
            </div>
            <div class="card">
                <h2>Latency metrics (ms)</h2>
                <canvas id="latencyChart"></canvas>
            </div>
        </div>

        <div class="card" style="margin-top: 40px;">
            <h2>📋 Detailed Performance results</h2>
            <table id="resultsTable">
                <thead>
                    <tr>
                        <th>Group</th>
                        <th>Input Len</th>
                        <th>Output Len</th>
                        <th>Concurrency (n)</th>
                        <th>Throughput (tok/s)</th>
                        <th>TTFT (ms)</th>
                        <th>TPOT (ms)</th>
                        <th>ITL (ms)</th>
                        <th>E2EL (ms)</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>

        <div class="card" style="margin-top: 40px;" id="diagnostics-card">
            <h2>🔍 Prometheus metrics analysis & diagnostics</h2>
            <pre id="diagnostics-text">No diagnostics file found.</pre>
        </div>
    </div>

    <script>
        const data = /*DATA_PLACEHOLDER*/;
        const diagnostics = /*DIAGNOSTICS_PLACEHOLDER*/;
        const engineParams = /*ENGINE_PARAMS_PLACEHOLDER*/;

        // Populate Model Info
        if (engineParams && Object.keys(engineParams).length > 0) {
            document.getElementById('model-info').innerHTML = `
                <strong>Model ID:</strong> ${engineParams.model_name || 'N/A'}<br>
                <strong>Tokenizer:</strong> ${engineParams.tokenizer || 'N/A'}<br>
                <strong>Context Length limit:</strong> ${engineParams.max_model_len || 'N/A'} tokens
            `;
        }

        if (diagnostics) {
            document.getElementById('diagnostics-text').innerText = diagnostics;
        }

        // Render Table
        const tbody = document.querySelector('#resultsTable tbody');
        data.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${row.group}</td>
                <td>${row.input_len}</td>
                <td>${row.output_len}</td>
                <td>${row.n}</td>
                <td>${row.throughput ? row.throughput.toFixed(1) : 'N/A'}</td>
                <td>${row.ttft ? row.ttft.toFixed(1) : 'N/A'}</td>
                <td>${row.tpot ? row.tpot.toFixed(1) : 'N/A'}</td>
                <td>${row.itl ? row.itl.toFixed(1) : 'N/A'}</td>
                <td>${row.e2el ? row.e2el.toFixed(1) : 'N/A'}</td>
            `;
            tbody.appendChild(tr);
        });

        // Charts
        const validData = data.filter(r => r.throughput !== null);
        const throughputLabels = validData.map(r => `${r.group} (n=${r.n})`);
        const throughputValues = validData.map(r => r.throughput);

        const ttftValues = validData.map(r => r.ttft || 0);
        const tpotValues = validData.map(r => r.tpot || 0);
        const itlValues = validData.map(r => r.itl || 0);

        // Throughput Chart
        new Chart(document.getElementById('throughputChart'), {
            type: 'bar',
            data: {
                labels: throughputLabels,
                datasets: [{
                    label: 'Throughput (tok/s)',
                    data: throughputValues,
                    backgroundColor: '#38bdf8',
                    borderColor: '#0ea5e9',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                    x: { ticks: { color: '#94a3b8' } }
                },
                plugins: {
                    legend: { labels: { color: '#f8fafc' } }
                }
            }
        });

        // Latency Chart
        new Chart(document.getElementById('latencyChart'), {
            type: 'line',
            data: {
                labels: throughputLabels,
                datasets: [
                    {
                        label: 'TTFT (ms)',
                        data: ttftValues,
                        borderColor: '#f43f5e',
                        backgroundColor: 'rgba(244, 63, 94, 0.1)',
                        tension: 0.1
                    },
                    {
                        label: 'TPOT (ms)',
                        data: tpotValues,
                        borderColor: '#fbbf24',
                        backgroundColor: 'rgba(251, 191, 36, 0.1)',
                        tension: 0.1
                    },
                    {
                        label: 'ITL (ms)',
                        data: itlValues,
                        borderColor: '#34d399',
                        backgroundColor: 'rgba(52, 211, 153, 0.1)',
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                    x: { ticks: { color: '#94a3b8' } }
                },
                plugins: {
                    legend: { labels: { color: '#f8fafc' } }
                }
            }
        });
    </script>
</body>
</html>"""


def export_html_report(rows: list, html_path: str, output_dir: str) -> None:
    """Export benchmark data and metrics to a self-contained interactive HTML page."""
    os.makedirs(os.path.dirname(os.path.abspath(html_path)), exist_ok=True)
    
    # Try to load diagnostics and engine params
    diagnostics = ""
    diag_path = os.path.join(output_dir, "diagnostics.txt")
    if os.path.exists(diag_path):
        with open(diag_path, "r", encoding="utf-8") as f:
            diagnostics = f.read()

    engine_params = {}
    engine_path = os.path.join(output_dir, "engine_params.txt")
    if os.path.exists(engine_path):
        try:
            with open(engine_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                json_str = "".join([l for l in lines if not l.strip().startswith("#")])
                engine_params = json.loads(json_str)
        except Exception:
            pass

    content = HTML_TEMPLATE.replace("/*DATA_PLACEHOLDER*/", json.dumps(rows))
    content = content.replace("/*DIAGNOSTICS_PLACEHOLDER*/", json.dumps(diagnostics))
    content = content.replace("/*ENGINE_PARAMS_PLACEHOLDER*/", json.dumps(engine_params))

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"💾 HTML report generated → {os.path.abspath(html_path)}")


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
        "--report-md", default=None, metavar="MD_PATH",
        help="Path to export the Markdown report (optional, defaults to output_dir/summary_report.md)",
    )
    p.add_argument(
        "--report-html", default=None, metavar="HTML_PATH",
        help="Path to export the HTML report (optional, defaults to output_dir/report.html)",
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

    sort_col = "n" if args.sort_by == "num_prompts" else args.sort_by
    rows.sort(key=lambda r: _sort_key(r, sort_col))

    print_table(rows)

    if args.csv:
        export_csv(rows, args.csv)

    # Automatically output md and html reports to output_dir if not specified
    md_path = args.report_md or os.path.join(args.output_dir, "summary_report.md")
    export_markdown_report(rows, md_path, args.output_dir)

    html_path = args.report_html or os.path.join(args.output_dir, "report.html")
    export_html_report(rows, html_path, args.output_dir)


if __name__ == "__main__":
    main()