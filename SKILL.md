---
name: vllm-bench
description: 對已運行中支援 OpenAI-compatible API 的任何 LLM 推論服務（如 vLLM、sglang、Ollama、TGI 或 LiteLLM Proxy 等）執行 benchmark 效能測試。自動從 /v1/models API 偵測 model 名稱、tokenizer 及 max_model_len，使用 `vllm bench serve` CLI 執行三組標準測試（長文本壓力、吞吐量、延遲），並在可能時從 /metrics endpoint 收集系統指標進行分析診斷。適用於評估推論服務的 throughput、TTFT、TPOT、ITL 等效能指標。
---

# vllm Benchmark Serving Skill

本 Skill 專門用於對運作中且支援 OpenAI-compatible API 的任何 LLM 推論服務（如 vLLM、sglang、Ollama、TGI，以及經由 LiteLLM 等 Proxy 封裝的 API）進行吞吐量、延遲、長文本上下文的基準測試，並在服務端支援時分析 Prometheus 監控指標以判定瓶頸。

## 📂 腳本結構與工具

- [scripts/run_benchmark.py](file:///home/asus/.gemini/skills/vllm-bench/scripts/run_benchmark.py) — 執行完整 benchmark 流程。**（本機裝有 vllm 時使用）**
- [scripts/run_benchmark_k8s.py](file:///home/asus/.gemini/skills/vllm-bench/scripts/run_benchmark_k8s.py) — 透過輕量級 K8s Job 遠端執行 benchmark。**（Agent 本地無 vllm 時使用）**
- [scripts/summarize_results.py](file:///home/asus/.gemini/skills/vllm-bench/scripts/summarize_results.py) — 整理 `output/` 數據，產生 CLI 表格、CSV、Markdown Mermaid 報告與 Chart.js HTML 報告。

---

## 🧭 決策樹與工作流程

當 Agent 被要求進行 vLLM Benchmark 時，應遵循以下決策與執行步驟：

```
[開始：使用者要求執行 Benchmark]
  │
  ├── 步驟 1：檢測連線模式與可用模型
  │     └── 目的：確認連線可用，並選定目標模型。
  │     └── 指令：python scripts/run_benchmark.py --base-url <url> --api-key <key> --list-models
  │
  ├── 步驟 2：判斷執行環境
  │     │
  │     ├── [環境 A：Agent 容器內有安裝 vllm CLI]
  │     │     └── 使用 run_benchmark.py
  │     │     └── 指令：python scripts/run_benchmark.py --base-url <url> --model <model>
  │     │
  │     └── [環境 B：Agent 容器內無 vllm CLI（如 CPU-only 環境）]
  │           └── 使用 run_benchmark_k8s.py (委派給輕量 K8s Job 容器)
  │           └── 指令：python scripts/run_benchmark_k8s.py --base-url <url> --model <model> --namespace <ns>
  │
  └── 步驟 3：產生與分析報告 (自動執行)
        ├── 1. 產生 CLI 效能表格與 `output/summary.csv`
        ├── 2. 產生 `output/summary_report.md` (包含 Mermaid 折線圖)
        ├── 3. 產生 `output/report.html` (互動式 Chart.js 網頁)
        └── 4. 產生 `output/diagnostics.txt` (對比 /metrics 分析 KV Cache 搶占與瓶頸)
```

---

## 🚀 核心功能與參數說明

### 1. 三組預設標準測試
 benchmark 自動執行以下三組測試（可使用 `--groups` 參數指定特定組別）：

| 組別 | 預設 input_len | 預設 output_len | 並發並行數 (num_prompts) | 說明 |
| :--- | :--- | :--- | :--- | :--- |
| `long_context` | `max_model_len - 2048` | `2048` | `1`, `5`, `10` | 評估長文本上下文對記憶體與首字延遲 (TTFT) 的壓力 |
| `throughput` | `200` | `250` | `8`, `16`, `32`, `64`, `128` | 測試高並發下的系統最大輸出吞吐量 |
| `latency` | `100` | `100` | `1`, `8`, `16`, `32` | 評估在不同負載下相鄰 Token 產生的平滑度 (ITL/TPOT) |

### 2. 暖機機制 (Warm-up)
*   **預設開啟**：在收集 pre-benchmark metrics 前，腳本會自動送出一筆輕量請求（1 prompts, 128 in, 128 out）來初始化 vLLM 的 KV Cache 與 Tokenizer，防止「冷啟動」效能偏低影響數據。
*   **關閉方式**：在命令中加上 `--no-warmup` 旗標。

### 3. 資料集 (Dataset) 彈性
*   預設使用 `--dataset-name random`。
*   **擴充選項**：支援 `sharegpt`、`sonnet`、`hf` 等官方資料集。
*   **自訂資料集**：可使用 `--dataset-path <path>` 載入本地自訂的 JSON 格式資料集。
*   **隨機變化長度**：可加上 `--random-range-ratio 0.2`，使隨機產生的輸入/輸出長度有 +/-20% 波動，更貼近真實情境。

### 4. 智慧 Metrics 診斷 (Prometheus 整合)
在測試前後，腳本會存取 vLLM 服務的 `/metrics`，並比對兩者差異：
*   **GPU 快取飽和度**：分析 `gpu_cache_usage_factor` 是否超過 95%。
*   **搶占 (Preemption) 檢測**：比對前後 `num_requests_preempted` 累計數。若發生搶占，報告會顯示警告並建議調整參數（如調小 `max_num_seqs` 或降低並發）。
*   **排隊飽和**：分析 `num_requests_waiting` 來判定調度器是否過載。

### 5. 預設使用 Completions API 評測純粹性
*   **預設端點**：預設使用 `--endpoint /v1/completions`（非 Chat API）。
*   **為什麼不建議用 Chat API (`/v1/chat/completions`)？**
    *   **避開 Chat Template 雜訊**：Chat API 會強行在伺服器端套用模型的 Chat Template 模板（如 `<|im_start|>user\n...`），多出來的模板 Token 會導致實際處理 Token 數大於設定的 `input_len`，造成 Throughput 與 TTFT 指標計算失真。
    *   **精準衡量推理極限**：Completions API 能繞過模板字串解析與對話角色包裝的 Overhead，確保客戶端發送的 Token 數與引擎實際處理的 Token 數 100% 精準吻合，從而測出最純粹的 Engine 推理極限。
*   **⚠️ 格式對應與自適應 Backend 切換 (防呆機制)**：
    *   **格式差異**：`/v1/completions` 接收 `{"prompt": "..."}` 格式（對應 `--backend openai`）；而 `/v1/chat/completions` 則強制要求 `{"messages": [...]}` 對話格式（對應 `--backend openai-chat`）。若格式傳錯，API 伺服器會立刻回報 `400 Bad Request`。
    *   **自動防護**：若因其他 Agent 的限制或特定框架需求，必須將端點設定為 `/v1/chat/completions`，本 Skill 已內建防呆邏輯。一旦偵測到 Chat 端點且 `--backend` 為預設的 `openai` 時，**會自動將後端切換為 `openai-chat`**，確保發送對齊的 JSON 格式，完全免去格式不對造成的連線崩潰。

---

## 🛠️ K8s 模式進階設定

當在 Agent 本地無 vllm 時，使用 `run_benchmark_k8s.py`。

### 1. 輕量級資源自適應 (自動配適)
本腳本只作為客戶端發送請求，因此預設被調配為非常輕量級的資源：
- **CPU Request**: `500m` (半核) / **Limit**: `2`
- **Memory Request**: `1Gi` / **Limit**: `4Gi` (載入大 tokenizer 緩衝)
- **自訂參數**：若需調整（例如測試超大規模並發以防發送端 CPU 飽和），可傳入以下參數：
  ```bash
  python scripts/run_benchmark_k8s.py \
      --k8s-cpu-request 1 --k8s-mem-request 2Gi \
      --k8s-cpu-limit 4 --k8s-mem-limit 8Gi \
      --model <model_id> --base-url <url>
  ```

### 2. Job 失敗防護與崩潰擷取
若 K8s Job 失敗或超時，腳本不會只回傳「找不到 JSON」，而是會**自動擷取並印出 Pod Log 的最後 50 行**，供您立刻定位是否為連線超時、API Key 錯誤或 Pod OOM 等根本原因。

---

## 📊 結果輸出與報告

每次測試完成後，`output/` 目錄將會包含：

| 檔案 | 說明 |
| :--- | :--- |
| `*.json` | 各個並發點測試的原始詳細指標 JSON。 |
| `diagnostics.txt` | 系統指標分析（如 GPU Cache 使用率與 Preemption 診斷）。 |
| `engine_params.txt` | 自動偵測到的模型名稱、Tokenizer 與最大長度限制。 |
| `summary.csv` | （選填）以 CSV 格式匯出的簡短數據。 |
| **`summary_report.md`** | **效能 Markdown 報告。內嵌 Mermaid 柱狀圖以渲染 Concurrency vs Throughput 曲線。** |
| **`report.html`** | **互動式網頁報告。內嵌 Chart.js 以可視化 TTFT、TPOT、ITL 延遲曲線與吞吐量。** |

---

## 💡 給 Agent 的調用範例與引導提示

若你是正在處理 benchmark 任務的 Agent：
1. **第一步**：先用 `list-models` 指令檢測並回報使用者伺服器上目前有哪些模型可選。
2. **第二步**：詢問使用者希望測試的模型與測試範疇（例如是否跳過長文本 `long_context`，或是否指定真實資料集）。
3. **第三步**：執行測試。如果本機缺少 `vllm` CLI，一律自動改為使用 `run_benchmark_k8s.py`。
4. **第四步**：測試結束後，使用 `summarize_results.py` 產出報告。
5. **第五步**：提供 [summary_report.md](file:///path/to/output/summary_report.md) 和 [report.html](file:///path/to/output/report.html) 的絕對路徑點擊連結給使用者。
6. **第六步**：在回覆中簡短總結 Throughput (最佳與最差) 以及 `/metrics` 診斷中是否有搶占警告，並給予使用者伺服器端配置的調整建議。