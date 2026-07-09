#!/usr/bin/env python3
"""
vllm Benchmark Runner using `vllm bench serve` CLI.

Usage:
    # List available models first
    python scripts/run_benchmark.py --base-url https://litellm-xxx.sslip.io --api-key sk-xxx --list-models

    # Run benchmark with a specific model
    python scripts/run_benchmark.py --base-url https://litellm-xxx.sslip.io --api-key sk-xxx \
        --model vllm-a40-qwen36-27b --max-model-len 262144

    # Direct vllm server (auto-detect everything)
    python scripts/run_benchmark.py --host 10.110.134.151 --port 5000
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import requests

# ---------------------------------------------------------------------------
# Benchmark group definitions
# ---------------------------------------------------------------------------
BENCHMARK_GROUPS = {
    "long_context": {
        "random_input_len": None,   # None = dynamic: max_model_len - 2048
        "random_output_len": 2048,
        "num_prompts_list": [1, 5, 10],
    },
    "throughput": {
        "random_input_len": 200,
        "random_output_len": 250,
        "num_prompts_list": [8, 16, 32, 64, 128],
    },
    "latency": {
        "random_input_len": 100,
        "random_output_len": 100,
        "num_prompts_list": [1, 8, 16, 32],
    },
}


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run vllm benchmark tests using `vllm bench serve` CLI."
    )
    p.add_argument("--base-url",      default=None,
                   help="Full server base URL, e.g. https://litellm-xxx.sslip.io "
                        "(overrides --host and --port if provided)")
    p.add_argument("--host",         default="127.0.0.1",
                   help="vllm server host (default: 127.0.0.1)")
    p.add_argument("--port",         type=int, default=5000,
                   help="vllm server port (default: 5000)")
    p.add_argument("--backend",      default="openai",
                   help="benchmark backend (default: openai)")
    p.add_argument("--endpoint",     default="/v1/completions",
                   help="API endpoint (default: /v1/completions)")
    p.add_argument("--dataset-name", default="random",
                   help="dataset name (default: random)")
    p.add_argument("--dataset-path", default=None,
                   help="Path to the dataset file (required for non-random datasets if server-side doesn't have it)")
    p.add_argument("--random-range-ratio", type=float, default=0.0,
                   help="Range ratio for random dataset generation (default: 0.0)")
    p.add_argument("--request-rate", default="inf",
                   help="requests per second, 'inf' sends all at once (default: inf)")
    p.add_argument("--output-dir",   default="output",
                   help="directory to save results (default: output)")
    p.add_argument("--groups",       nargs="+",
                   choices=["long_context", "throughput", "latency", "all"],
                   default=["all"],
                   help="benchmark groups to run (default: all)")
    p.add_argument("--api-key",      default=None,
                   help="API key for Authorization: Bearer header (optional)")
    p.add_argument("--models-path",   default=None,
                   help="Override path for models endpoint "
                        "(default: /v1/models for host:port, /models for base-url)")
    p.add_argument("--model",          default=None,
                   help="Model name to benchmark. Required when using --base-url "
                        "(LiteLLM proxy does not return tokenizer/max_model_len).")
    p.add_argument("--tokenizer",      default=None,
                   help="Tokenizer path or name. Auto-detected from /v1/models when "
                        "using --host/--port. Required when server does not return 'root' field.")
    p.add_argument("--max-model-len",  type=int, default=None,
                   help="Override max model length. Auto-detected from /v1/models when "
                        "using --host/--port. Required when server does not return 'max_model_len' field.")
    p.add_argument("--list-models",    action="store_true",
                   help="List all available models from the server and exit.")
    p.add_argument("--print-results",  action="store_true",
                   help="Print all result JSON files to stdout after benchmark "
                        "(used by K8s Job launcher to retrieve results via pod logs).")
    p.add_argument("--no-warmup",      action="store_true",
                   help="Disable the warm-up request before starting actual benchmark runs.")
    p.add_argument("--skip-models",    action="store_true",
                   help="Skip /v1/models auto-detection. Requires --model, --tokenizer, and --max-model-len.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Server info via HTTP API (no ps/ssh needed)
# ---------------------------------------------------------------------------
def build_base_url(args: argparse.Namespace) -> str:
    """Resolve base URL from --base-url or --host/--port."""
    if args.base_url:
        return args.base_url.rstrip("/")
    return f"http://{args.host}:{args.port}"


def detect_models_path(base_url: str, api_key: str = None) -> str:
    """
    Dynamically detect whether to use /v1/models or /models.
    Tries /v1/models first, then falls back to /models if /v1/models returns 404.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    # Try /v1/models first
    try:
        url = f"{base_url}/v1/models"
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code != 404:
            return "/v1/models"
    except Exception:
        pass

    # Try /models as fallback
    try:
        url = f"{base_url}/models"
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code != 404:
            return "/models"
    except Exception:
        pass

    # Fallback to standard OpenAI if both probes fail or timeout
    return "/models" if ("litellm" in base_url.lower() or "proxy" in base_url.lower()) else "/v1/models"


def get_litellm_model_info(base_url: str, api_key: str, model_name: str) -> dict:
    """
    Fetch detailed model info from LiteLLM GET /model/info.
    Returns max_tokens and hf_model_name for the specified model.
    """
    url = f"{base_url}/model/info"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        for m in models:
            if m.get("model_name") == model_name:
                return m.get("model_info", {})
    except Exception as e:
        print(f"  ⚠️  Could not fetch /model/info: {e}")
    return {}


def get_server_info(
    base_url: str,
    api_key: str = None,
    models_path: str = "/v1/models",
    model_name: str = None,
) -> dict:
    """
    Fetch model info from GET {models_path}.
    """
    url = f"{base_url}{models_path}"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    models = resp.json()["data"]

    # If model_name is specified, find it; otherwise use the first model
    if model_name:
        match = next((m for m in models if m["id"] == model_name), None)
        if match is None:
            available = [m["id"] for m in models]
            raise ValueError(
                f"Model '{model_name}' not found. Available: {available}"
            )
        model = match
    else:
        model = models[0]

    return {
        "model_name":      model["id"],
        "safe_model_name": model["id"].replace("/", "_").replace(" ", "_"),
        # LiteLLM proxy may not return 'root' or 'max_model_len'
        "tokenizer":       model.get("root"),
        "max_model_len":   model.get("max_model_len"),
    }


def get_vllm_metrics(base_url: str, api_key: str = None) -> str:
    """
    Fetch Prometheus metrics from GET /metrics.
    Only retains lines relevant to vllm (filters noise).
    """
    url = f"{base_url}/metrics"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        lines = [
            line for line in resp.text.splitlines()
            if (line.startswith("vllm:")
                or line.startswith("# HELP vllm:")
                or line.startswith("# TYPE vllm:"))
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"# ERROR: failed to fetch /metrics: {e}"


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------
def build_vllm_bench_cmd(
    args: argparse.Namespace,
    model_name: str,
    tokenizer: str,
    input_len: int,
    output_len: int,
    num_prompts: int,
    seed: int,
    result_file: str = None,
    save_result: bool = True,
) -> list:
    """Build the `vllm bench serve` command list."""
    backend = args.backend
    if "chat" in args.endpoint and backend == "openai":
        backend = "openai-chat"

    cmd = [
        "vllm", "bench", "serve",
        "--backend",           backend,
        "--model",             model_name,
        "--tokenizer",         tokenizer,
        "--endpoint",          args.endpoint,
        "--dataset-name",      args.dataset_name,
        "--request-rate",      args.request_rate,
        "--num-prompts",       str(num_prompts),
        "--seed",              str(seed),
    ]

    if args.dataset_name == "random":
        cmd += [
            "--random-input-len",  str(input_len),
            "--random-output-len", str(output_len),
        ]
        if args.random_range_ratio > 0.0:
            cmd += ["--random-range-ratio", str(args.random_range_ratio)]
    
    if args.dataset_path:
        cmd += ["--dataset-path", args.dataset_path]

    if save_result and result_file:
        cmd += [
            "--save-result",
            "--result-dir",        args.output_dir,
            "--result-filename",   result_file,
        ]

    # Use --base-url when provided, otherwise use --host/--port
    if args.base_url:
        cmd += ["--base-url", build_base_url(args)]
    else:
        cmd += ["--host", args.host, "--port", str(args.port)]
    # Append API key header if provided
    if args.api_key:
        cmd += ["--header", f"Authorization=Bearer {args.api_key}"]
    return cmd


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------
def run_single(
    args: argparse.Namespace,
    model_name: str,
    safe_model_name: str,
    tokenizer: str,
    group_name: str,
    input_len: int,
    output_len: int,
    num_prompts: int,
    is_warmup: bool = False,
) -> dict:
    """Execute one benchmark run and return result metadata."""
    seed = int(time.time() * 1e9) % 4294967295
    result_file = f"{safe_model_name}_{input_len}-{output_len}_{num_prompts}.json"

    cmd = build_vllm_bench_cmd(
        args, model_name, tokenizer,
        input_len, output_len, num_prompts,
        seed, result_file, save_result=not is_warmup,
    )

    if is_warmup:
        print(f"  🔥 Warm-up run: num_prompts={num_prompts} (results will not be saved)")
    else:
        print(f"  🚀 num_prompts={num_prompts:4d} | seed={seed} -> {result_file}")
    
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        print(f"  ❌ Failed (returncode={proc.returncode})")
        print(proc.stderr[:400])
    else:
        print(f"  ✅ Done")

    return {
        "group":       group_name,
        "num_prompts": num_prompts,
        "input_len":   input_len,
        "output_len":  output_len,
        "seed":        seed,
        "result_file": result_file if not is_warmup else None,
        "returncode":  proc.returncode,
    }


# ---------------------------------------------------------------------------
# Metrics analysis / diagnostics
# ---------------------------------------------------------------------------
def parse_metric_value(metrics_text: str, name: str) -> float:
    """Parse a specific metric value from prometheus text output."""
    pattern = re.compile(rf'^{name}(?:\{{[^}}]*\}})?\s+([0-9eE\.\+-]+)', re.MULTILINE)
    match = pattern.search(metrics_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return 0.0


def diagnose_metrics(before_text: str, after_text: str) -> str:
    """Analyze vllm metrics changes and output diagnostic observations."""
    if before_text.startswith("# ERROR") or after_text.startswith("# ERROR"):
        return "  • ⚠️ Prometheus /metrics endpoint was not accessible. Diagnostic checks skipped."
    if "vllm:" not in before_text and "vllm:" not in after_text:
        return "  • ⚠️ No vLLM-specific metrics found. Diagnostic checks skipped."

    diagnostics = []
    
    # 1. GPU cache usage factor (from after)
    gpu_cache_after = parse_metric_value(after_text, "vllm:gpu_cache_usage_factor")
    if gpu_cache_after > 0.0:
        diagnostics.append(f"  • GPU KV Cache Usage Factor: {gpu_cache_after:.2%}")
        if gpu_cache_after > 0.95:
            diagnostics.append("    ⚠️  GPU Cache is almost fully utilized (>95%). This can trigger preemption under heavy load.")
    
    # 2. Preemptions (diff between before and after)
    preempt_before = (parse_metric_value(before_text, "vllm:num_requests_preempted") or 
                      parse_metric_value(before_text, "vllm:num_requests_preempted_total"))
    preempt_after = (parse_metric_value(after_text, "vllm:num_requests_preempted") or 
                     parse_metric_value(after_text, "vllm:num_requests_preempted_total"))
    
    preempt_diff = max(0.0, preempt_after - preempt_before)
    if preempt_diff > 0:
        diagnostics.append(f"  • 🚨 Preempted Requests detected during test: {int(preempt_diff)}")
        diagnostics.append("    ⚠️  Preemptions occur when vllm runs out of GPU memory for KV Cache. "
                           "Consider decreasing max_num_seqs or increasing gpu_memory_utilization on the server.")
    else:
        diagnostics.append("  • Preemptions: 0 (No requests were preempted. GPU memory allocation was healthy).")

    # 3. Waiting / Pending queue
    waiting_after = parse_metric_value(after_text, "vllm:num_requests_waiting")
    if waiting_after > 0:
        diagnostics.append(f"  • Waiting Requests after test: {int(waiting_after)}")
        diagnostics.append("    ℹ️  There are still requests in the waiting queue, indicating scheduler saturation.")

    if not diagnostics:
        return "  • No significant performance bottleneck detected in prometheus metrics."
    
    return "\n".join(diagnostics)


def try_resolve_hf_tokenizer_id(tokenizer_path: str) -> str:
    """
    If the tokenizer path is a local path (starts with /), try to parse a potential HuggingFace Model ID from it.
    Example: '/models/proj-707c659f/models/qwen/qwen3-0.6b/latest' -> 'Qwen/qwen3-0.6b'
    """
    if not tokenizer_path or not tokenizer_path.startswith("/"):
        return tokenizer_path
        
    # Pattern: .../models/<org>/<model_name>(/latest)?
    match = re.search(r'models/([^/]+)/([^/]+)(?:/latest)?$', tokenizer_path)
    if match:
        org = match.group(1)
        model = match.group(2)
        if org.lower() == "qwen":
            org = "Qwen"
        elif org.lower() == "meta-llama":
            org = "meta-llama"
        elif org.lower() == "google":
            org = "google"
        return f"{org}/{model}"
        
    return tokenizer_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def list_models(base_url: str, api_key: str, models_path: str) -> None:
    """Fetch and print all available models from the server."""
    url = f"{base_url}{models_path}"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        models = resp.json()["data"]
    except Exception as e:
        print(f"❌ Failed to fetch models: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n📋 Available models from {url}:")
    print(f"{'='*55}")
    for i, m in enumerate(models, 1):
        print(f"  {i:2d}. {m['id']}")
    print(f"{'='*55}")
    print(f"Total: {len(models)} model(s)")
    print(f"\nTo benchmark a specific model, run:")
    if "https" in base_url or not base_url.startswith("http://127"):
        print(f"  python scripts/run_benchmark.py \\")
        print(f"      --base-url {base_url} \\")
        if api_key:
            print(f"      --api-key <your_key> \\")
        print(f"      --model <model_id>")
        print(f"")
        print(f"  ℹ️  max_model_len will be auto-detected from /model/info")
    else:
        print(f"  python scripts/run_benchmark.py \\")
        print(f"      --host <host> --port <port>")
        print(f"")
        print(f"  ℹ️  model / tokenizer / max_model_len will be auto-detected")


# ---------------------------------------------------------------------------
# Performance metrics table
# ---------------------------------------------------------------------------

_NA = "N/A"


def _fmt(value, fmt=".1f") -> str:
    """Format a numeric value or return N/A if None."""
    if value is None:
        return _NA
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return _NA


def _print_metrics_table(all_results: list, output_dir: str) -> None:
    """Read result JSON files and print a performance metrics summary table."""
    rows = []
    for r in all_results:
        if r["returncode"] != 0:
            continue
        json_path = os.path.join(output_dir, r["result_file"])
        data: dict = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass
        rows.append({
            "group":      r["group"],
            "input_len":  r["input_len"],
            "output_len": r["output_len"],
            "n":          r["num_prompts"],
            "throughput": data.get("output_throughput"),
            "ttft":       data.get("mean_ttft_ms"),
            "tpot":       data.get("mean_tpot_ms"),
            "itl":        data.get("mean_itl_ms"),
            "e2el":       data.get("mean_e2el_ms"),
        })

    if not rows:
        return

    header = (
        f"{'Group':<14} {'in':>6} {'out':>5} {'n':>4}  "
        f"{'Tput(tok/s)':>11}  {'TTFT(ms)':>9}  "
        f"{'TPOT(ms)':>9}  {'ITL(ms)':>8}  {'E2EL(ms)':>9}"
    )
    sep = "=" * 85
    dash = "-" * 85

    print(f"\n📊 Performance Metrics Summary:")
    print(sep)
    print(header)
    print(dash)
    for row in rows:
        print(
            f"{row['group']:<14} {row['input_len']:>6} {row['output_len']:>5} {row['n']:>4}  "
            f"{_fmt(row['throughput']):>11}  {_fmt(row['ttft']):>9}  "
            f"{_fmt(row['tpot']):>9}  {_fmt(row['itl']):>8}  {_fmt(row['e2el']):>9}"
        )
    print(sep)


def main() -> None:
    args = parse_args()

    # Handle --list-models: just print available models and exit
    if args.list_models:
        base_url    = build_base_url(args)
        models_path = args.models_path or detect_models_path(base_url, args.api_key)
        list_models(base_url, args.api_key, models_path)
        sys.exit(0)

    os.makedirs(args.output_dir, exist_ok=True)

    engine_params_file  = os.path.join(args.output_dir, "engine_params.txt")
    metrics_before_file = os.path.join(args.output_dir, "metrics_before.txt")
    metrics_after_file  = os.path.join(args.output_dir, "metrics_after.txt")
    diagnostics_file    = os.path.join(args.output_dir, "diagnostics.txt")

    # ------------------------------------------------------------------
    # Step 1: Fetch server info from /v1/models (or skip if all params provided)
    # ------------------------------------------------------------------
    base_url = build_base_url(args)
    models_path = args.models_path or detect_models_path(base_url, args.api_key)

    if args.skip_models:
        # All info must come from CLI args
        if not args.model:
            print("❌ --skip-models requires --model", file=sys.stderr); sys.exit(1)
        if not args.tokenizer:
            print("❌ --skip-models requires --tokenizer", file=sys.stderr); sys.exit(1)
        if not args.max_model_len:
            print("❌ --skip-models requires --max-model-len", file=sys.stderr); sys.exit(1)
        model_name = args.model
        safe_model_name = model_name.replace("/", "_").replace(" ", "_")
        tokenizer = args.tokenizer
        max_model_len = args.max_model_len
        print(f"\n⏭️  Skipping /v1/models auto-detection (--skip-models)")
        info = {
            "model_name":      model_name,
            "safe_model_name": safe_model_name,
            "tokenizer":       tokenizer,
            "max_model_len":   max_model_len,
        }
    else:
        print(f"\n📡 Fetching server info from {base_url}{models_path} ...")
        try:
            info = get_server_info(base_url, args.api_key, models_path, args.model)
        except Exception as e:
            print(f"❌ Cannot connect to vllm server: {e}", file=sys.stderr)
            sys.exit(1)

    model_name      = info["model_name"]
    safe_model_name = info["safe_model_name"]
    detected_tokenizer = info.get("tokenizer") or model_name
    max_model_len = args.max_model_len or info.get("max_model_len")

    # If max_model_len or tokenizer not found and using base-url, try /model/info
    if not args.skip_models and (max_model_len is None or detected_tokenizer == model_name) and args.base_url:
        print(f"  🔍 Fetching model details from {base_url}/model/info ...")
        litellm_info = get_litellm_model_info(base_url, args.api_key, model_name)
        if max_model_len is None:
            max_model_len = litellm_info.get("max_tokens")
            if max_model_len:
                print(f"  ✅ max_tokens from /model/info: {max_model_len}")
        # Use hf_model_name as tokenizer if not manually specified
        if detected_tokenizer == model_name:
            hf_model_name = litellm_info.get("hf_model_name")
            if hf_model_name:
                detected_tokenizer = hf_model_name
                print(f"  ✅ hf_model_name from /model/info: {detected_tokenizer}")

    # Try to resolve HF ID from detected path
    hf_tokenizer = try_resolve_hf_tokenizer_id(detected_tokenizer)

    tokenizer = args.tokenizer
    if not tokenizer:
        if hf_tokenizer != detected_tokenizer:
            print(f"  🔍 Testing connection to HuggingFace for tokenizer '{hf_tokenizer}' ...")
            try:
                # Test connection by trying to load the tokenizer config
                from transformers import AutoTokenizer
                AutoTokenizer.from_pretrained(hf_tokenizer, local_files_only=False)
                tokenizer = hf_tokenizer
                print(f"  ✅ HuggingFace tokenizer '{tokenizer}' is available and preferred.")
            except Exception as e:
                tokenizer = detected_tokenizer
                print(f"  ⚠️  HuggingFace connection failed or repo unavailable ({e}).\n"
                      f"     Falling back to local tokenizer path: '{tokenizer}'")
        else:
            tokenizer = detected_tokenizer

    if max_model_len is None:
        print(
            "❌ max_model_len could not be detected from server.\n"
            "   Please specify it manually: --max-model-len <value>\n"
            "   Tip: check the model's context length and use that value.\n"
            "   e.g. --max-model-len 262144",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  model_name:    {model_name}")
    print(f"  tokenizer:     {tokenizer}")
    print(f"  max_model_len: {max_model_len}")

    # Save engine params
    with open(engine_params_file, "w", encoding="utf-8") as f:
        f.write("# Engine params (from /v1/models)\n")
        f.write(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    print(f"  ✅ engine_params.txt saved")

    # ------------------------------------------------------------------
    # Step 1b: Execute Warm-up (Unless --no-warmup)
    # ------------------------------------------------------------------
    if not args.no_warmup:
        print(f"\n🔥 Executing Warm-up request to initialize server KV cache...")
        run_single(
            args, model_name, safe_model_name, tokenizer,
            "warmup", 128, 128, 1, is_warmup=True
        )

    # ------------------------------------------------------------------
    # Step 2: Collect metrics snapshot BEFORE benchmark
    # ------------------------------------------------------------------
    print(f"\n📊 Collecting pre-benchmark metrics ...")
    metrics_before = get_vllm_metrics(base_url, args.api_key)
    with open(metrics_before_file, "w", encoding="utf-8") as f:
        f.write("# Snapshot: before benchmark\n")
        f.write(f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(metrics_before + "\n")
    print(f"  ✅ metrics_before.txt saved")

    # ------------------------------------------------------------------
    # Step 3: Determine groups to run
    # ------------------------------------------------------------------
    groups = (
        list(BENCHMARK_GROUPS.keys())
        if "all" in args.groups
        else args.groups
    )
    total = sum(len(BENCHMARK_GROUPS[g]["num_prompts_list"]) for g in groups)
    print(f"\n🏃 Running {len(groups)} group(s), {total} benchmark(s) total")

    # ------------------------------------------------------------------
    # Step 4: Execute benchmarks
    # ------------------------------------------------------------------
    all_results = []
    for group_name in groups:
        group      = BENCHMARK_GROUPS[group_name]
        input_len  = (
            (max_model_len - 2048)
            if group["random_input_len"] is None
            else group["random_input_len"]
        )
        output_len = group["random_output_len"]

        print(f"\n--- [{group_name}] input={input_len} output={output_len} ---")

        for num_prompts in group["num_prompts_list"]:
            result = run_single(
                args, model_name, safe_model_name, tokenizer,
                group_name, input_len, output_len, num_prompts,
            )
            all_results.append(result)

    # ------------------------------------------------------------------
    # Step 5: Collect metrics snapshot AFTER benchmark
    # ------------------------------------------------------------------
    print(f"\n📊 Collecting post-benchmark metrics ...")
    metrics_after = get_vllm_metrics(base_url, args.api_key)
    with open(metrics_after_file, "w", encoding="utf-8") as f:
        f.write("# Snapshot: after benchmark\n")
        f.write(f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(metrics_after + "\n")
    print(f"  ✅ metrics_after.txt saved")

    # ------------------------------------------------------------------
    # Step 5b: Perform Metrics Diagnostics
    # ------------------------------------------------------------------
    print(f"\n🔍 Diagnosing system metrics ...")
    diag_results = diagnose_metrics(metrics_before, metrics_after)
    with open(diagnostics_file, "w", encoding="utf-8") as f:
        f.write(f"# System Metrics Diagnostics\n")
        f.write(f"# Analyzed on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(diag_results + "\n")
    print(diag_results)
    print(f"  ✅ diagnostics.txt saved")

    # ------------------------------------------------------------------
    # Step 6: Print summary
    # ------------------------------------------------------------------
    ok = sum(1 for r in all_results if r["returncode"] == 0)
    print(f"\n{'=' * 65}")
    print(f"📋 Benchmark Summary: {ok}/{len(all_results)} succeeded")
    print(f"{'=' * 65}")
    for r in all_results:
        icon = "✅" if r["returncode"] == 0 else "❌"
        print(
            f"  {icon} [{r['group']:12s}] "
            f"in={r['input_len']:6d} out={r['output_len']:4d} "
            f"n={r['num_prompts']:3d}  ->  {r['result_file']}"
        )
    print(f"\nOutput directory: {os.path.abspath(args.output_dir)}/")

    # ------------------------------------------------------------------
    # Step 6b: Print performance metrics table
    # ------------------------------------------------------------------
    _print_metrics_table(all_results, args.output_dir)

    # ------------------------------------------------------------------
    # Step 7: Print results to stdout (for K8s Job log retrieval)
    # ------------------------------------------------------------------
    if args.print_results:
        print("\n===BENCHMARK_RESULTS_START===")
        for r in all_results:
            json_path = os.path.join(args.output_dir, r["result_file"])
            if os.path.exists(json_path):
                print(f"===FILE:{r['result_file']}===")
                with open(json_path, "r", encoding="utf-8") as f:
                    print(f.read())
        print("===FILE:diagnostics.txt===")
        with open(diagnostics_file, "r", encoding="utf-8") as f:
            print(f.read())
        print("===FILE:engine_params.txt===")
        with open(engine_params_file, "r", encoding="utf-8") as f:
            print(f.read())
        print("===BENCHMARK_RESULTS_END===")


if __name__ == "__main__":
    main()