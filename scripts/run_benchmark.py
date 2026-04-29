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
    return p.parse_args()


# ---------------------------------------------------------------------------
# Server info via HTTP API (no ps/ssh needed)
# ---------------------------------------------------------------------------
def build_base_url(args: argparse.Namespace) -> str:
    """Resolve base URL from --base-url or --host/--port."""
    if args.base_url:
        return args.base_url.rstrip("/")
    return f"http://{args.host}:{args.port}"


def get_litellm_model_info(base_url: str, api_key: str, model_name: str) -> dict:
    """
    Fetch detailed model info from LiteLLM GET /model/info.
    Returns max_tokens and hf_model_name for the specified model.

    Response structure:
        {"data": [{"model_name": "...",
                   "model_info": {"max_tokens": 262144,
                                  "hf_model_name": "Qwen/Qwen3.6-27B"}, ...}]}
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

    Returns:
        {
          "model_name":      str,  # used for --model arg
          "safe_model_name": str,  # sanitized for filenames
          "tokenizer":       str,  # used for --tokenizer arg
          "max_model_len":   int,
        }

    Example response field:
        {"id": "vllm-a40-qwen36-27b", "root": "/model/Qwen3.6-27B",
         "max_model_len": 262144}
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
    result_file: str,
) -> list:
    """Build the `vllm bench serve` command list."""
    cmd = [
        "vllm", "bench", "serve",
        "--backend",           args.backend,
        "--model",             model_name,
        "--tokenizer",         tokenizer,
        "--endpoint",          args.endpoint,
        "--dataset-name",      args.dataset_name,
        "--request-rate",      args.request_rate,
        "--num-prompts",       str(num_prompts),
        "--random-input-len",  str(input_len),
        "--random-output-len", str(output_len),
        "--save-result",
        "--result-dir",        args.output_dir,
        "--result-filename",   result_file,
        "--seed",              str(seed),
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
) -> dict:
    """Execute one benchmark run and return result metadata."""
    # Generate random seed (mirrors original bash: $(($(date +%s%N) % 4294967295)))
    seed = int(time.time() * 1e9) % 4294967295
    result_file = f"{safe_model_name}_{input_len}-{output_len}_{num_prompts}.json"

    cmd = build_vllm_bench_cmd(
        args, model_name, tokenizer,
        input_len, output_len, num_prompts,
        seed, result_file,
    )

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
        "result_file": result_file,
        "returncode":  proc.returncode,
    }


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


def main() -> None:
    args = parse_args()

    # Handle --list-models: just print available models and exit
    if args.list_models:
        base_url    = build_base_url(args)
        models_path = args.models_path or ("/models" if args.base_url else "/v1/models")
        list_models(base_url, args.api_key, models_path)
        sys.exit(0)

    os.makedirs(args.output_dir, exist_ok=True)

    engine_params_file  = os.path.join(args.output_dir, "engine_params.txt")
    metrics_before_file = os.path.join(args.output_dir, "metrics_before.txt")
    metrics_after_file  = os.path.join(args.output_dir, "metrics_after.txt")

    # ------------------------------------------------------------------
    # Step 1: Fetch server info from /v1/models
    # ------------------------------------------------------------------
    base_url = build_base_url(args)
    # Auto-detect models path: use /models when --base-url is provided, else /v1/models
    models_path = args.models_path or ("/models" if args.base_url else "/v1/models")
    print(f"\n📡 Fetching server info from {base_url}{models_path} ...")
    try:
        info = get_server_info(base_url, args.api_key, models_path, args.model)
    except Exception as e:
        print(f"❌ Cannot connect to vllm server: {e}", file=sys.stderr)
        sys.exit(1)

    model_name      = info["model_name"]
    safe_model_name = info["safe_model_name"]
    # --tokenizer / --max-model-len CLI args override auto-detected values
    tokenizer     = args.tokenizer or info["tokenizer"] or model_name
    max_model_len = args.max_model_len or info["max_model_len"]

    # If max_model_len or tokenizer not found and using base-url, try /model/info
    if (max_model_len is None or tokenizer == model_name) and args.base_url:
        print(f"  🔍 Fetching model details from {base_url}/model/info ...")
        litellm_info = get_litellm_model_info(base_url, args.api_key, model_name)
        if max_model_len is None:
            max_model_len = litellm_info.get("max_tokens")
            if max_model_len:
                print(f"  ✅ max_tokens from /model/info: {max_model_len}")
        # Use hf_model_name as tokenizer if not manually specified
        if tokenizer == model_name and not args.tokenizer:
            hf_model_name = litellm_info.get("hf_model_name")
            if hf_model_name:
                tokenizer = hf_model_name
                print(f"  ✅ hf_model_name from /model/info: {tokenizer}")

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
    # Step 2: Collect metrics snapshot BEFORE benchmark
    # ------------------------------------------------------------------
    print(f"\n📊 Collecting pre-benchmark metrics ...")
    with open(metrics_before_file, "w", encoding="utf-8") as f:
        f.write("# Snapshot: before benchmark\n")
        f.write(f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(get_vllm_metrics(base_url, args.api_key) + "\n")
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
        # Use explicit None check to avoid bug where 0 would be treated as None
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
    with open(metrics_after_file, "w", encoding="utf-8") as f:
        f.write("# Snapshot: after benchmark\n")
        f.write(f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(get_vllm_metrics(base_url, args.api_key) + "\n")
    print(f"  ✅ metrics_after.txt saved")

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


if __name__ == "__main__":
    main()