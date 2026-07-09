# vLLM Benchmark Serving Skill

本專案提供了一套針對運作中且支援 OpenAI 相容 API (OpenAI-compatible API) 的任何 LLM 推論服務（例如 vLLM、sglang、Ollama、TGI，以及經由 LiteLLM 等 Proxy 封裝的 API）進行吞吐量 (Throughput)、延遲 (Latency) 以及長文本上下文 (Long Context) 壓測的基準測試工具。

除了執行測試外，本工具還支援自動抓取與比對服務端的 Prometheus 指標 (`/metrics`)，用以智慧診斷系統瓶頸，如 GPU KV Cache 搶占 (Preemption) 及 scheduler 飽和度。

---

## 📂 專案結構

- **[SKILL.md](file:///home/asus/.gemini/skills/vllm-bench/SKILL.md)** — Antigravity Agent 的 Skill 描述與導引規範。
- **[scripts/run_benchmark.py](file:///home/asus/.gemini/skills/vllm-bench/scripts/run_benchmark.py)** — 本機端執行完整測試流程的主腳本。
- **[scripts/run_benchmark_k8s.py](file:///home/asus/.gemini/skills/vllm-bench/scripts/run_benchmark_k8s.py)** — 透過輕量級 Kubernetes Job 遠端執行基準測試的腳本（適用於測試發起端環境無 vLLM 套件時）。
- **[scripts/summarize_results.py](file:///home/asus/.gemini/skills/vllm-bench/scripts/summarize_results.py)** — 分析並彙整測試結果，自動產生 Markdown/Mermaid 報告及互動式 HTML Chart 報告。
- **[references/](file:///home/asus/.gemini/skills/vllm-bench/references/)** — 包含參數說明文件與額外參考資料。

---

## 🚀 核心功能與設計特色

### 1. 三組標準測試群組 (Standard Groups)
預設會對以下三種情境執行全方位壓測：
- **`long_context`**: 測試長文本（預設為 `max_model_len - 2048` 輸入 / `2048` 輸出，並發數為 `1`, `5`, `10`）。評估長上下文下的首字延遲 (TTFT) 與記憶體極限。
- **`throughput`**: 測試高並發最大輸出能力（預設為 `200` 輸入 / `250` 輸出，並發數 `8` 到 `128`）。
- **`latency`**: 測試低負載下 Token 產生的平滑度（預設為 `100` 輸入 / `100` 輸出，並發數 `1` 到 `32`），專注於衡量 ITL (Inter-token Latency) 與 TPOT (Time-per-output-token)。

### 2. 動態端點偵測 (Adaptive Endpoint Detection)
- 支援自動探測底層 API 機制。在未指定 `--models-path` 時，腳本會自動探測 `base-url` 上的 `/v1/models` (OpenAI/vLLM 標準) 與 `/models` (LiteLLM 標準)，無縫適應不同推論框架。

### 3. API 防護與自適應切換 (Auto Backend Switch)
- 預設推薦使用 Completions API (`/v1/completions`) 以避免 Chat Template 產生的多餘 Token 雜訊。
- 若指定端點為 Chat API (`/v1/chat/completions`) 且後端設為預設的 `openai`，腳本會自動切換為 `openai-chat` 格式，確保請求的 JSON 格式對齊，防止 API 400 報錯。

### 4. 智慧監控指標診斷 (Prometheus Diagnostics)
- 在測試前後會抓取伺服器的 `/metrics`。
- 自動分析並診斷是否有發生 **GPU 快取飽和度過高 (>95%)**、**KV Cache 搶占 (Preemption) 次數** 以及 **等待調度器飽和度 (Waiting Requests)**。
- 具備防呆機制，若檢測到非 vLLM 服務或端點不支援，將自動跳過診斷而不會輸出誤導性的數據。

### 5. 暖機機制 (Warm-up)
- 預設在正式測試前發送一筆輕量請求（1 Prompts, 128 in, 128 out）以初始化 KV Cache，避免「冷啟動」影響測試精度。

---

## 🛠️ 安裝與準備工作

1. 安裝必要依賴：
   ```bash
   pip install requests pandas
   ```
2. （僅限本地運行模式）需要安裝 `vllm` 作為壓測發送端：
   ```bash
   pip install vllm
   ```

---

## 📖 使用說明

### 1. 快速列出伺服器上所有可用模型
```bash
python scripts/run_benchmark.py --base-url http://<YOUR_LLM_HOST>:<PORT> --list-models
```

### 2. 執行本機基準測試 (Local Mode)
```bash
python scripts/run_benchmark.py \
    --base-url http://<YOUR_LLM_HOST>:<PORT> \
    --model <MODEL_NAME> \
    --api-key <API_KEY_IF_NEEDED>
```
*如需測試特定組別（例如僅測試 Throughput 與 Latency）：*
```bash
python scripts/run_benchmark.py --base-url http://localhost:8000 --model qwen --groups throughput latency
```

### 3. 透過 K8s 遠端執行測試 (K8s Job Mode)
當壓測發起端沒有安裝 GPU 或無法安裝 `vllm` 依賴時，使用此腳本在 Kubernetes 叢集建立一個臨時的 Job 容器執行壓測，完成後會自動拉回測試結果：
```bash
python scripts/run_benchmark_k8s.py \
    --base-url http://<YOUR_LLM_HOST>:<PORT> \
    --model <MODEL_NAME> \
    --namespace <K8s_NAMESPACE>
```

---

## 📊 結果輸出與視覺化報告

測試結果會預設輸出至 `output/` 目錄：
- **`summary_report.md`**: Markdown 效能總結報告，內嵌 Mermaid 柱狀圖（可視化並發數 vs Throughput）。
- **`report.html`**: 互動式網頁報告，使用 Chart.js 呈現 TTFT、TPOT、ITL 延遲曲線與吞吐量。
- **`diagnostics.txt`**: GPU KV Cache 的搶占與瓶頸診斷分析報告。
- **`engine_params.txt`**: 記錄自動偵測到的模型長度與 Tokenizer 資訊。
