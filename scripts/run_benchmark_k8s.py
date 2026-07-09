#!/usr/bin/env python3
"""
vllm Benchmark K8s Job Launcher.

Runs run_benchmark.py inside a K8s Job using the vllm image,
allowing benchmark execution from an Agent container that does not have vllm installed.

The script:
  1. Packages run_benchmark.py into a ConfigMap
  2. Creates a K8s Job (no GPU, uses vllm/vllm-openai image)
  3. Polls Job until completion
  4. Reads Pod logs and extracts JSON results
  5. Saves results to local output/ directory
  6. Cleans up Job + ConfigMap
"""
import argparse
import json
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# K8s curl-based client (mirrors pattern from intelligent-model-deployer)
# ---------------------------------------------------------------------------

class K8sCurlClient:
    """Lightweight K8s client using curl via subprocess with a kubectl fallback for local testing."""

    def __init__(self):
        self.token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
        self.ca_path    = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
        self.api_server = "https://kubernetes.default.svc"

        if not os.path.exists(self.token_path):
            self.token_path = None
            self.ca_path    = None
            self.use_kubectl = True
            print("ℹ️  Running outside cluster. Will use local 'kubectl' CLI for K8s API operations.",
                  file=sys.stderr)
        else:
            self.use_kubectl = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_token(self) -> str:
        if self.token_path and os.path.exists(self.token_path):
            with open(self.token_path, "r") as f:
                return f.read().strip()
        return ""

    def _run_curl(self, method: str, url: str, data: dict = None) -> str:
        """Execute a curl command against the K8s API server."""
        cmd = ["curl", "-s", "-k", "-X", method, url]

        token = self._read_token()
        if token:
            cmd.extend(["-H", f"Authorization: Bearer {token}"])
        if self.ca_path and os.path.exists(self.ca_path):
            cmd.extend(["--cacert", self.ca_path])
        if data is not None:
            cmd.extend(["-H", "Content-Type: application/json",
                        "-d", json.dumps(data)])

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"curl failed (rc={result.returncode}): {result.stderr}")
        return result.stdout

    def _parse(self, raw: str) -> dict:
        """Parse JSON response; raise on K8s error status."""
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse K8s response: {e}\nRaw: {raw[:300]}")
        if isinstance(data, dict) and data.get("kind") == "Status" and data.get("status") == "Failure":
            raise RuntimeError(f"K8s API error: {data.get('message', raw[:200])}")
        return data

    # ------------------------------------------------------------------
    # core/v1 – ConfigMap
    # ------------------------------------------------------------------

    def create_configmap(self, namespace: str, body: dict) -> dict:
        if self.use_kubectl:
            proc = subprocess.run(["kubectl", "apply", "-f", "-", "-n", namespace],
                                  input=json.dumps(body), capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"kubectl apply configmap failed: {proc.stderr}")
            return body
        url = f"{self.api_server}/api/v1/namespaces/{namespace}/configmaps"
        return self._parse(self._run_curl("POST", url, data=body))

    def delete_configmap(self, namespace: str, name: str) -> None:
        if self.use_kubectl:
            subprocess.run(["kubectl", "delete", "configmap", name, "-n", namespace],
                           capture_output=True)
            return
        url = f"{self.api_server}/api/v1/namespaces/{namespace}/configmaps/{name}"
        self._run_curl("DELETE", url)

    # ------------------------------------------------------------------
    # batch/v1 – Job
    # ------------------------------------------------------------------

    def create_job(self, namespace: str, body: dict) -> dict:
        if self.use_kubectl:
            proc = subprocess.run(["kubectl", "create", "-f", "-", "-n", namespace],
                                  input=json.dumps(body), capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"kubectl create job failed: {proc.stderr}")
            return body
        url = f"{self.api_server}/apis/batch/v1/namespaces/{namespace}/jobs"
        return self._parse(self._run_curl("POST", url, data=body))

    def get_job(self, namespace: str, name: str) -> dict:
        if self.use_kubectl:
            proc = subprocess.run(["kubectl", "get", "job", name, "-n", namespace, "-o", "json"],
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                return {}
            return json.loads(proc.stdout)
        url = f"{self.api_server}/apis/batch/v1/namespaces/{namespace}/jobs/{name}"
        raw = self._run_curl("GET", url)
        return json.loads(raw) if raw else {}

    def delete_job(self, namespace: str, name: str) -> None:
        if self.use_kubectl:
            subprocess.run(["kubectl", "delete", "job", name, "-n", namespace, "--cascade=foreground"],
                           capture_output=True)
            return
        url = f"{self.api_server}/apis/batch/v1/namespaces/{namespace}/jobs/{name}"
        body = {"propagationPolicy": "Foreground"}
        self._run_curl("DELETE", url, data=body)

    # ------------------------------------------------------------------
    # core/v1 – Pod
    # ------------------------------------------------------------------

    def list_pods(self, namespace: str, label_selector: str) -> dict:
        if self.use_kubectl:
            proc = subprocess.run(["kubectl", "get", "pods", "-n", namespace, "-l", label_selector, "-o", "json"],
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                return {}
            return json.loads(proc.stdout)
        url = (f"{self.api_server}/api/v1/namespaces/{namespace}/pods"
               f"?labelSelector={label_selector}")
        raw = self._run_curl("GET", url)
        return json.loads(raw) if raw else {}

    def get_pod_log(self, namespace: str, pod_name: str) -> str:
        if self.use_kubectl:
            proc = subprocess.run(["kubectl", "logs", pod_name, "-n", namespace],
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"kubectl logs failed: {proc.stderr}")
            return proc.stdout
        url = f"{self.api_server}/api/v1/namespaces/{namespace}/pods/{pod_name}/log"
        return self._run_curl("GET", url)


# ---------------------------------------------------------------------------
# Benchmark K8s runner
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RUN_BENCHMARK_PY = os.path.join(_SCRIPT_DIR, "run_benchmark.py")

_RESULTS_START = "===BENCHMARK_RESULTS_START==="
_RESULTS_END   = "===BENCHMARK_RESULTS_END==="
_FILE_PREFIX   = "===FILE:"
_FILE_SUFFIX   = "==="


class BenchmarkK8sRunner:
    """Orchestrates the full K8s Job lifecycle for running vllm benchmarks."""

    def __init__(
        self,
        namespace: str = "frank-dev",
        vllm_image: str = "vllm/vllm-openai:v0.20.1",
        timeout: int = 3600,
        no_cleanup: bool = False,
        output_dir: str = "output",
        cpu_request: str = "500m",
        mem_request: str = "1Gi",
        cpu_limit: str = "2",
        mem_limit: str = "4Gi",
    ):
        self.namespace   = namespace
        self.vllm_image  = vllm_image
        self.timeout     = timeout
        self.no_cleanup  = no_cleanup
        self.output_dir  = output_dir
        self.cpu_request = cpu_request
        self.mem_request = mem_request
        self.cpu_limit   = cpu_limit
        self.mem_limit   = mem_limit
        self.k8s         = K8sCurlClient()

    def _unique_name(self) -> str:
        """Generate a unique short name based on timestamp."""
        ts = int(time.time())
        return f"vllm-bench-{ts}"

    def build_configmap_body(self, cm_name: str) -> dict:
        """Read run_benchmark.py and package it as a ConfigMap."""
        if not os.path.exists(_RUN_BENCHMARK_PY):
            raise FileNotFoundError(
                f"run_benchmark.py not found at {_RUN_BENCHMARK_PY}"
            )
        with open(_RUN_BENCHMARK_PY, "r", encoding="utf-8") as f:
            script_content = f.read()

        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name":      cm_name,
                "namespace": self.namespace,
                "labels":    {"app": "vllm-bench", "managed-by": "run_benchmark_k8s"},
            },
            "data": {
                "run_benchmark.py": script_content,
            },
        }

    def build_job_body(self, job_name: str, cm_name: str, bench_args: list) -> dict:
        """Build a batch/v1 Job spec with resource limits."""
        command = ["python3", "/scripts/run_benchmark.py"] + bench_args + ["--print-results"]

        return {
            "apiVersion": "batch/v1",
            "kind":       "Job",
            "metadata": {
                "name":      job_name,
                "namespace": self.namespace,
                "labels":    {"app": "vllm-bench", "managed-by": "run_benchmark_k8s"},
            },
            "spec": {
                "ttlSecondsAfterFinished": 600,
                "backoffLimit":            0,   # no retry on failure
                "template": {
                    "metadata": {
                        "labels": {"app": "vllm-bench", "job-name": job_name},
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [{
                            "name":    "vllm-bench",
                            "image":   self.vllm_image,
                            "command": command,
                            "env": [
                                {"name": "NVIDIA_VISIBLE_DEVICES", "value": "none"},
                                {"name": "VLLM_TARGET_DEVICE",      "value": "cpu"},
                                {"name": "VLLM_NO_USAGE_STATS",    "value": "1"},
                                {"name": "VLLM_DO_NOT_TRACK",      "value": "1"},
                            ],
                            "resources": {
                                "requests": {"cpu": self.cpu_request, "memory": self.mem_request},
                                "limits":   {"cpu": self.cpu_limit, "memory": self.mem_limit},
                            },
                            "volumeMounts": [
                                {
                                    "name":      "scripts",
                                    "mountPath": "/scripts",
                                },
                                {
                                    "name":      "models-volume",
                                    "mountPath": "/models",
                                }
                            ],
                        }],
                        "volumes": [
                            {
                                "name": "scripts",
                                "configMap": {
                                    "name": cm_name,
                                    "defaultMode": 0o755,
                                },
                            },
                            {
                                "name": "models-volume",
                                "hostPath": {
                                    "path": "/var/lib/afsbox/models",
                                    "type": "Directory"
                                }
                            }
                        ],
                    },
                },
            },
        }

    def submit(self, bench_args: list) -> tuple:
        """Create ConfigMap and Job; return (job_name, cm_name)."""
        name    = self._unique_name()
        cm_name = f"{name}-scripts"

        print(f"\n📦 Creating ConfigMap: {cm_name} ...")
        cm_body = self.build_configmap_body(cm_name)
        self.k8s.create_configmap(self.namespace, cm_body)
        print(f"  ✅ ConfigMap created")

        print(f"\n🚀 Creating K8s Job: {name} ...")
        job_body = self.build_job_body(name, cm_name, bench_args)
        self.k8s.create_job(self.namespace, job_body)
        print(f"  ✅ Job created  (namespace={self.namespace}, image={self.vllm_image})")
        print(f"  📋 resources: requests(cpu={self.cpu_request}, mem={self.mem_request}) limits(cpu={self.cpu_limit}, mem={self.mem_limit})")
        print(f"  📋 bench args: {' '.join(bench_args)}")

        return name, cm_name

    def wait_for_completion(self, job_name: str) -> bool:
        """Poll Job status every 15 seconds until succeeded/failed or timeout."""
        print(f"\n⏳ Waiting for Job '{job_name}' to complete (timeout={self.timeout}s) ...")
        deadline = time.time() + self.timeout
        poll_interval = 15
        elapsed = 0

        while time.time() < deadline:
            job = self.k8s.get_job(self.namespace, job_name)
            status = job.get("status", {})

            succeeded = status.get("succeeded", 0) or 0
            failed    = status.get("failed",    0) or 0
            active    = status.get("active",    0) or 0

            print(f"  [{elapsed:5d}s] active={active}  succeeded={succeeded}  failed={failed}")

            if succeeded > 0:
                print(f"  ✅ Job succeeded!")
                return True
            if failed > 0:
                print(f"  ❌ Job failed!")
                return False

            time.sleep(poll_interval)
            elapsed += poll_interval

        print(f"  ⏰ Timeout ({self.timeout}s) reached — Job may still be running.",
              file=sys.stderr)
        return False

    def fetch_logs(self, job_name: str) -> str:
        """Find the Pod created by the Job and return its full log."""
        print(f"\n📄 Fetching logs for Job '{job_name}' ...")
        selector = f"job-name={job_name}"
        pods_resp = self.k8s.list_pods(self.namespace, selector)
        pods = pods_resp.get("items", [])

        if not pods:
            print("  ⚠️  No pods found for this Job.", file=sys.stderr)
            return ""

        pod_name = pods[0]["metadata"]["name"]
        print(f"  📌 Pod: {pod_name}")
        log = self.k8s.get_pod_log(self.namespace, pod_name)
        print(f"  ✅ Log retrieved ({len(log)} chars)")
        return log

    def parse_and_save_results(self, log_text: str) -> list:
        """Extract JSON and other files from the log and save them."""
        os.makedirs(self.output_dir, exist_ok=True)
        saved = []

        start_idx = log_text.find(_RESULTS_START)
        end_idx   = log_text.find(_RESULTS_END)

        if start_idx == -1 or end_idx == -1:
            print("  ⚠️  No benchmark results block found in logs.", file=sys.stderr)
            print("\n📋 Last 50 lines of Pod logs for diagnostic purposes:")
            print("-" * 65)
            lines = log_text.splitlines()
            for line in lines[-50:]:
                print(line)
            print("-" * 65)
            return saved

        block = log_text[start_idx + len(_RESULTS_START):end_idx]
        lines = block.splitlines()

        current_filename = None
        current_lines: list = []

        def _flush():
            if current_filename and current_lines:
                content = "\n".join(current_lines).strip()
                dest = os.path.join(self.output_dir, current_filename)
                with open(dest, "w", encoding="utf-8") as fh:
                    fh.write(content + "\n")
                saved.append(current_filename)
                print(f"  💾 Saved: {current_filename}")

        for line in lines:
            if line.startswith(_FILE_PREFIX) and line.endswith(_FILE_SUFFIX):
                _flush()
                current_filename = line[len(_FILE_PREFIX):-len(_FILE_SUFFIX)]
                current_lines    = []
            else:
                if current_filename is not None:
                    current_lines.append(line)

        _flush()  # save the last file
        return saved

    def cleanup(self, job_name: str, cm_name: str) -> None:
        """Delete the K8s Job (cascade to Pods) and ConfigMap."""
        print(f"\n🧹 Cleaning up Job '{job_name}' and ConfigMap '{cm_name}' ...")
        try:
            self.k8s.delete_job(self.namespace, job_name)
            print(f"  ✅ Job deleted")
        except Exception as e:
            print(f"  ⚠️  Could not delete Job: {e}", file=sys.stderr)
        try:
            self.k8s.delete_configmap(self.namespace, cm_name)
            print(f"  ✅ ConfigMap deleted")
        except Exception as e:
            print(f"  ⚠️  Could not delete ConfigMap: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run vllm benchmark via a K8s Job.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Benchmark target (mirrors run_benchmark.py) ───────────────────
    p.add_argument("--base-url",      default=None,
                   help="Full server base URL, e.g. https://litellm-xxx.sslip.io")
    p.add_argument("--host",          default="127.0.0.1",
                   help="vllm server host (default: 127.0.0.1)")
    p.add_argument("--port",          type=int, default=5000,
                   help="vllm server port (default: 5000)")
    p.add_argument("--api-key",       default=None,
                   help="API key for Authorization: Bearer header")
    p.add_argument("--model",         default=None,
                   help="Model name to benchmark (required for LiteLLM proxy)")
    p.add_argument("--tokenizer",     default=None,
                   help="Tokenizer path or HuggingFace model name")
    p.add_argument("--max-model-len", type=int, default=None,
                   help="Override max model length")
    p.add_argument("--groups",        nargs="+",
                   choices=["long_context", "throughput", "latency", "all"],
                   default=["all"],
                   help="Benchmark groups to run (default: all)")
    p.add_argument("--backend",       default="openai",
                   help="Benchmark backend (default: openai)")
    p.add_argument("--endpoint",      default="/v1/completions",
                   help="API endpoint (default: /v1/completions)")
    p.add_argument("--dataset-name",  default="random",
                   help="Dataset name (default: random)")
    p.add_argument("--dataset-path",  default=None,
                   help="Path to the dataset file")
    p.add_argument("--random-range-ratio", type=float, default=0.0,
                   help="Range ratio for random dataset generation")
    p.add_argument("--request-rate",  default="inf",
                   help="Requests per second (default: inf)")
    p.add_argument("--models-path",   default=None,
                   help="Override path for models endpoint")
    p.add_argument("--list-models",   action="store_true",
                   help="List available models from the server and exit")
    p.add_argument("--output-dir",    default="output",
                   help="Local directory to save retrieved results (default: output)")
    p.add_argument("--no-warmup",      action="store_true",
                   help="Disable warm-up request")
    p.add_argument("--skip-models",    action="store_true",
                   help="Skip /v1/models auto-detection. Requires --model, --tokenizer, and --max-model-len.")

    # ── K8s-specific ──────────────────────────────────────────────────
    p.add_argument("--namespace",     default="frank-dev",
                   help="K8s namespace to create Job in (default: frank-dev)")
    p.add_argument("--vllm-image",    default="vllm/vllm-openai:v0.20.1",
                   help="vllm Docker image for the Job (default: vllm/vllm-openai:v0.20.1)")
    p.add_argument("--timeout",       type=int, default=3600,
                   help="Seconds to wait for Job completion (default: 3600)")
    p.add_argument("--no-cleanup",    action="store_true",
                   help="Keep Job and ConfigMap after completion")
    p.add_argument("--k8s-cpu-request", default="500m",
                   help="K8s CPU request for the Job container (default: 500m)")
    p.add_argument("--k8s-mem-request", default="1Gi",
                   help="K8s memory request for the Job container (default: 1Gi)")
    p.add_argument("--k8s-cpu-limit",   default="2",
                   help="K8s CPU limit for the Job container (default: 2)")
    p.add_argument("--k8s-mem-limit",   default="4Gi",
                   help="K8s memory limit for the Job container (default: 4Gi)")

    return p.parse_args()


def detect_models_path(base_url: str, api_key: str = None) -> str:
    """
    Dynamically detect whether to use /v1/models or /models.
    Tries /v1/models first, then falls back to /models if /v1/models returns 404.
    """
    import requests
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


def _list_models_local(args: argparse.Namespace) -> None:
    """Fetch and print models directly from this process."""
    try:
        import requests
    except ImportError:
        print("❌ 'requests' package not found. Install it: pip install requests",
              file=sys.stderr)
        sys.exit(1)

    base_url    = args.base_url.rstrip("/") if args.base_url else f"http://{args.host}:{args.port}"
    models_path = args.models_path or detect_models_path(base_url, args.api_key)
    url         = f"{base_url}{models_path}"
    headers     = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        models = resp.json()["data"]
    except Exception as e:
        print(f"❌ Failed to fetch models: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n📋 Available models from {url}:")
    print("=" * 55)
    for i, m in enumerate(models, 1):
        print(f"  {i:2d}. {m['id']}")
    print("=" * 55)
    print(f"Total: {len(models)} model(s)")
    print("\nTo benchmark via K8s Job, run:")
    print(f"  python scripts/run_benchmark_k8s.py \\")
    print(f"      --base-url {base_url} \\")
    if args.api_key:
        print(f"      --api-key <your_key> \\")
    print(f"      --model <model_id>")


def build_bench_args(args: argparse.Namespace) -> list:
    """Convert parsed args into a list of CLI flags for run_benchmark.py."""
    bench = []

    if args.base_url:
        bench += ["--base-url", args.base_url]
    else:
        bench += ["--host", args.host, "--port", str(args.port)]

    if args.api_key:
        bench += ["--api-key", args.api_key]
    if args.model:
        bench += ["--model", args.model]
    if args.tokenizer:
        bench += ["--tokenizer", args.tokenizer]
    if args.max_model_len:
        bench += ["--max-model-len", str(args.max_model_len)]
    if args.groups:
        bench += ["--groups"] + args.groups
    if args.backend and args.backend != "openai":
        bench += ["--backend", args.backend]
    if args.endpoint and args.endpoint != "/v1/completions":
        bench += ["--endpoint", args.endpoint]
    if args.dataset_name and args.dataset_name != "random":
        bench += ["--dataset-name", args.dataset_name]
    if args.dataset_path:
        bench += ["--dataset-path", args.dataset_path]
    if args.random_range_ratio > 0.0:
        bench += ["--random-range-ratio", str(args.random_range_ratio)]
    if args.request_rate and args.request_rate != "inf":
        bench += ["--request-rate", args.request_rate]
    if args.models_path:
        bench += ["--models-path", args.models_path]
    if args.no_warmup:
        bench += ["--no-warmup"]
    if args.skip_models:
        bench += ["--skip-models"]

    bench += ["--output-dir", "/output"]

    return bench


def main() -> None:
    args = parse_args()

    if args.list_models:
        _list_models_local(args)
        sys.exit(0)

    runner = BenchmarkK8sRunner(
        namespace   = args.namespace,
        vllm_image  = args.vllm_image,
        timeout     = args.timeout,
        no_cleanup  = args.no_cleanup,
        output_dir  = args.output_dir,
        cpu_request = args.k8s_cpu_request,
        mem_request = args.k8s_mem_request,
        cpu_limit   = args.k8s_cpu_limit,
        mem_limit   = args.k8s_mem_limit,
    )

    job_name = cm_name = None
    succeeded = False

    try:
        bench_args = build_bench_args(args)
        job_name, cm_name = runner.submit(bench_args)
        succeeded = runner.wait_for_completion(job_name)
        log_text = runner.fetch_logs(job_name)

        # Save the raw pod logs to output/pod_log.txt for troubleshooting
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "pod_log.txt"), "w", encoding="utf-8") as fh:
            fh.write(log_text)
        print("  💾 Saved: pod_log.txt")

        print(f"\n💾 Parsing and saving results to '{args.output_dir}/' ...")
        saved = runner.parse_and_save_results(log_text)

        print(f"\n{'=' * 65}")
        if succeeded:
            print(f"✅ Benchmark Job completed successfully")
        else:
            print(f"❌ Benchmark Job failed or timed out")
        print(f"   Job:       {job_name}")
        print(f"   Namespace: {args.namespace}")
        print(f"   Saved:     {len(saved)} result file(s) → {os.path.abspath(args.output_dir)}/")
        for fname in saved:
            print(f"             • {fname}")
        print(f"{'=' * 65}")

    finally:
        if job_name and not args.no_cleanup:
            runner.cleanup(job_name, cm_name)
        elif args.no_cleanup:
            print(f"\n🔍 --no-cleanup set. Resources retained for inspection:")
            print(f"   Job:       {job_name}  (namespace={args.namespace})")
            print(f"   ConfigMap: {cm_name}")

    sys.exit(0 if succeeded else 1)


if __name__ == "__main__":
    main()