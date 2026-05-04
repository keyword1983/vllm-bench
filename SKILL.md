---
name: vllm-bench
description: 對已運行中的 vllm OpenAI-compatible server 執行 benchmark 效能測試。自動從 /v1/models API 偵測 model 名稱、tokenizer 及 max_model_len，使用 `vllm bench serve` CLI 執行三組標準測試（長文本壓力、吞吐量、延遲），並從 /metrics endpoint 收集 benchmark 前後的系統指標。適用於評估 vllm 推論服務的 throughput、TTFT、TPOT、ITL 等效能指標。
---

# vllm Benchmark Serving Skill

## 可用腳本

- `scripts/run_benchmark.py` — 執行完整 benchmark 流程（需要 vllm 環境）
- `scripts/run_benchmark_k8s.py` — 透過 K8s Job 執行 benchmark（**Agent 環境無 vllm 時使用**）

**永遠先執行 `--help` 確認用法，請勿直接讀取原始碼。**

```bash
python scripts/run_benchmark.py --help
python scripts/run_benchmark_k8s.py --help
```

## 決策樹

```
使用者要求執行 benchmark
├── Agent 環境有安裝 vllm？
│   ├── YES → 直接執行（run_benchmark.py）
│   │   ├── 直連 vllm server（host:port）
│   │   │   └── python scripts/run_benchmark.py --host <host> --port <port>
│   │   └── 透過 LiteLLM proxy（base-url）
│   │       ├── 步驟 1：查詢可用模型
│   │       │         python scripts/run_benchmark.py \
│   │       │             --base-url <url> --api-key <key> --list-models
│   │       └── 步驟 2：指定 model 執行
│   │                 python scripts/run_benchmark.py \
│   │                     --base-url <url> --api-key <key> \
│   │                     --model <model_id> --max-model-len <value>
│   └── NO → 透過 K8s Job 執行（run_benchmark_k8s.py）
│       ├── 步驟 1：查詢可用模型（本地執行，不需 K8s）
│       │         python scripts/run_benchmark_k8s.py \
│       │             --base-url <url> --api-key <key> --list-models
│       └── 步驟 2：建立 K8s Job 執行 benchmark
│                 python scripts/run_benchmark_k8s.py \
│                     --base-url <url> --api-key <key> \
│                     --model <model_id> --namespace frank-dev
```

## 快速使用

**查詢可用模型清單（先做這步）：**
```bash
python scripts/run_benchmark.py \
    --base-url https://litellm-xxx.sslip.io \
    --api-key sk-xxxxx \
    --list-models
```

**執行完整預設測試（共 14 次）：**
```bash
python scripts/run_benchmark.py --host 10.110.134.151 --port 5000
```

**帶 API key 執行：**
```bash
python scripts/run_benchmark.py --host 10.110.134.151 --port 5000 --api-key sk-xxxxx
```

**透過 LiteLLM proxy（base-url 模式）：**
```bash
# tokenizer 指定 HuggingFace model name（推薦）
python scripts/run_benchmark.py \
    --base-url https://litellm-xxx.sslip.io \
    --api-key sk-xxxxx \
    --model 5glab-a40-qwen36-27b \
    --tokenizer Qwen/Qwen3.6-27B \
    --max-model-len 262144
```

**僅執行特定組別：**
```bash
python scripts/run_benchmark.py --host 10.110.134.151 --port 5000 --groups throughput latency
```

**查看所有參數：**
```bash
python scripts/run_benchmark.py --help
```

## 三組標準測試

| 組別 | input_len | output_len | num_prompts | 說明 |
|------|-----------|------------|-------------|------|
| `long_context` | max_model_len - 2048（動態） | 2048 | 1, 5, 10 | 長文本壓力測試 |
| `throughput` | 200 | 250 | 8, 16, 32, 64, 128 | 吞吐量測試 |
| `latency` | 100 | 100 | 1, 8, 16, 32 | 延遲測試 |

全部測試使用相同設定：
- `--backend openai`
- `--endpoint /v1/completions`
- `--dataset-name random`
- `--request-rate inf`

## 自動偵測（來自 models API）

| 連線模式 | model_name | tokenizer | max_model_len |
|---------|------------|-----------|---------------|
| `--host/--port`（直連 vllm） | ✅ 自動（`/v1/models` `id`）| ✅ 自動（`/v1/models` `root`）| ✅ 自動（`/v1/models` `max_model_len`）|
| `--base-url`（LiteLLM proxy） | ✅ 自動（`/models`，需 `--model` 指定目標）| ✅ 自動（`/model/info` `hf_model_name`）| ✅ 自動（`/model/info` `max_tokens`）|

**自動偵測優先序（LiteLLM proxy 模式）：**
1. CLI 參數 `--tokenizer` / `--max-model-len`（最優先）
2. `/model/info` 的 `hf_model_name` / `max_tokens`（自動查詢）
3. 若 `/model/info` 403 → tokenizer fallback 為 model_name，max_model_len 需手動指定

## 輸出檔案（存於 `output/`）

| 檔案 | 說明 |
|------|------|
| `{safe_model_name}_{input_len}-{output_len}_{num_prompts}.json` | 每次測試的詳細效能結果 |
| `metrics_before.txt` | Benchmark 前的 vllm `/metrics` 快照 |
| `metrics_after.txt` | Benchmark 後的 vllm `/metrics` 快照 |
| `engine_params.txt` | 從 `/v1/models` 取得的 engine 資訊（JSON） |

## 前置條件

- `vllm` 已安裝，且包含 `vllm bench serve` CLI
- vllm server 已在指定 host:port 運行（支援跨容器，純 HTTP 存取）
- Python 套件：`requests`（`pip install requests`）
- （選填）若 server 啟用 API key 驗證，透過 `--api-key` 參數傳入

## 注意事項

⚠️ `long_context` 組的 `input_len` 會動態計算為 `max_model_len - 2048`。
例如 `max_model_len=262144` 時，`input_len=260096`，每次請求的 token 數極大，
**執行時間可能非常長**，請確認 server 資源充足再執行此組別。

若只想快速測試，可使用 `--groups throughput latency` 跳過此組別。

## 常見問題

**❌ dataset-name 不支援錯誤**
→ 改用 `--dataset-name random`（`vllm bench serve` 穩定版支援的標準資料集）

**❌ 無法連線到 vllm server**
→ 確認 server 已啟動，並確認 host/port 正確

**❌ 401 Unauthorized 錯誤**
→ server 啟用了 API key 驗證，請加上 `--api-key <your_key>` 參數

**❌ max_model_len could not be detected**
→ 腳本會自動嘗試從 `/model/info` 取得。若該端點回傳 403（API key 無權限），
  請手動指定：`--max-model-len 262144`

**❌ /model/info 403 Forbidden**
→ 目前 API key 無權限存取 `/model/info`。請直接加上 `--max-model-len <value>`，
  可從模型文件或管理員處確認 context length。
  例如：`--max-model-len 262144`

**❌ tokenizer 找不到（tokenizer: <model_id>）**
→ 腳本會自動從 `/model/info` 的 `hf_model_name` 欄位取得正確的 HuggingFace tokenizer。
  若 `/model/info` 回傳 403，請用 `--tokenizer` 手動指定：
  - `--tokenizer Qwen/Qwen3.6-27B`
  - `--tokenizer google/gemma-3-27b-it`
  - `--tokenizer meta-llama/Llama-3.1-8B-Instruct`

**❌ Model 'xxx' not found**
→ 指定的 `--model` 不在清單中，腳本會印出所有可用模型供選擇

**❌ vllm bench serve 指令不存在**
→ 確認 vllm 版本，執行 `vllm --version` 及 `vllm bench --help`

## K8s Job 執行方式

### 使用時機

Agent 容器環境**無法安裝 vllm**（例如：CPU-only 環境、套件衝突、CUDA 依賴問題）時，
使用 `run_benchmark_k8s.py` 將 benchmark 工作委派給 K8s Job 執行。

**運作原理：**
```
Agent Pod（無 vllm）
  │
  ├─ 1. 將 run_benchmark.py 打包成 K8s ConfigMap
  ├─ 2. 建立 K8s Job（image: vllm/vllm-openai:v0.20.1，無 GPU）
  │       Job 內執行：python /scripts/run_benchmark.py ... --print-results
  ├─ 3. Poll Job 狀態直到完成
  ├─ 4. 讀取 Pod logs，解析 JSON 結果
  ├─ 5. 存回 Agent 本地 output/ 目錄
  └─ 6. 清理 Job + ConfigMap
```

### 快速使用（K8s 模式）

**查詢可用模型（本地執行，不建立 Job）：**
```bash
python scripts/run_benchmark_k8s.py \
    --base-url https://litellm-xxx.sslip.io \
    --api-key sk-xxxxx \
    --list-models
```

**執行 benchmark（透過 K8s Job）：**
```bash
python scripts/run_benchmark_k8s.py \
    --base-url https://litellm-xxx.sslip.io \
    --api-key sk-xxxxx \
    --model 5glab-a40-qwen36-27b \
    --namespace frank-dev
```

**僅執行特定組別（跳過長文本）：**
```bash
python scripts/run_benchmark_k8s.py \
    --base-url https://litellm-xxx.sslip.io \
    --api-key sk-xxxxx \
    --model 5glab-a40-qwen36-27b \
    --groups throughput latency \
    --namespace frank-dev
```

**保留 Job/ConfigMap 供 debug（不自動清理）：**
```bash
python scripts/run_benchmark_k8s.py \
    --base-url https://litellm-xxx.sslip.io \
    --api-key sk-xxxxx \
    --model 5glab-a40-qwen36-27b \
    --no-cleanup
```

### K8s 專用參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--namespace` | `frank-dev` | 建立 Job 的 K8s namespace |
| `--vllm-image` | `vllm/vllm-openai:v0.20.1` | Job 使用的 vllm Docker image |
| `--timeout` | `3600` | 等待 Job 完成的最長秒數 |
| `--no-cleanup` | `false` | 加上此 flag 保留 Job 和 ConfigMap（供 debug） |

其餘參數（`--base-url`, `--api-key`, `--model`, `--tokenizer`, `--max-model-len`, `--groups` 等）
與 `run_benchmark.py` 完全相同。

### ServiceAccount RBAC 權限需求

Agent Pod 的 ServiceAccount 需要以下權限才能建立/管理 K8s 資源：

```yaml
rules:
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["create", "delete"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["list", "get"]
- apiGroups: [""]
  resources: ["pods/log"]
  verbs: ["get"]
- apiGroups: ["batch"]
  resources: ["jobs"]
  verbs: ["create", "get", "delete"]
```

### K8s 模式常見問題

**❌ ServiceAccount token not found**
→ Agent Pod 未掛載 ServiceAccount token，確認 Pod spec 未設定 `automountServiceAccountToken: false`

**❌ K8s API error: jobs is forbidden**
→ ServiceAccount 缺少 `batch/jobs` 的 RBAC 權限，請聯繫管理員授權

**❌ No benchmark results block found in logs**
→ Job 內的 `run_benchmark.py` 執行失敗，使用 `--no-cleanup` 後手動查看 Pod logs：
  ```bash
  kubectl logs -n frank-dev -l job-name=<job-name>
  ```

**❌ Job timeout**
→ 預設 timeout 為 3600 秒，long_context 組測試時間極長，可加大：`--timeout 7200`
  或跳過：`--groups throughput latency`

## 參考資料

- `references/vllm_bench_serve_args.md`：`vllm bench serve` 完整參數說明Model Info