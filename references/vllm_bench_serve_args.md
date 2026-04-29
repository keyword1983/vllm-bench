# vllm bench serve 參數速查

## 核心參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--backend` | `openai` | 後端類型。可選：`vllm`, `openai`, `openai-chat` 等 |
| `--host` | `127.0.0.1` | vllm server 的 host |
| `--port` | `8000` | vllm server 的 port |
| `--model` | （自動偵測） | 模型名稱，對應 `/v1/models` 回傳的 `id` |
| `--tokenizer` | — | Tokenizer 名稱或路徑，對應 `/v1/models` 回傳的 `root` |
| `--endpoint` | `/v1/completions` | API 端點 |
| `--num-prompts` | `1000` | 測試請求總數量 |
| `--request-rate` | `inf` | 每秒請求數。`inf` 表示所有請求同時送出 |
| `--seed` | `0` | 隨機種子，確保可重現性 |
| `--save-result` | `False` | 加上此旗標以儲存結果至 JSON 檔案 |
| `--result-dir` | `./` | 結果 JSON 檔案的輸出目錄 |
| `--result-filename` | （自動產生） | 結果 JSON 的檔案名稱。若未指定，自動以時間戳命名 |

## random dataset 參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--dataset-name` | `random` | 資料集名稱。可選：`random`, `sharegpt`, `sonnet`, `hf` 等 |
| `--random-input-len` | `1024` | 每個請求的輸入 token 數（僅 random dataset 使用）|
| `--random-output-len` | `128` | 每個請求的輸出 token 數（僅 random dataset 使用）|
| `--random-range-ratio` | `0.0` | 輸入/輸出長度的變化範圍比例，值域 `[0, 1)`。`0` 表示固定長度 |
| `--random-prefix-len` | `0` | 每個請求固定前綴的 token 數 |

## 效能指標說明

| 指標 | 全名 | 說明 |
|------|------|------|
| **TTFT** | Time To First Token | 從送出請求到收到第一個 token 的時間（毫秒）。反映 prefill 效率 |
| **TPOT** | Time Per Output Token | 每產生一個輸出 token 的平均時間（毫秒）。反映 decode 效率 |
| **ITL** | Inter-Token Latency | 相鄰兩個 token 之間的間隔時間（毫秒）。串流模式下的平滑度指標 |
| **E2EL** | End-to-End Latency | 從送出請求到接收完整回應的總時間（毫秒）|
| **Throughput** | — | 每秒處理的 token 總數（tokens/s）或每秒完成的請求數（req/s）|

## 常用指令範例

```bash
# 執行全部三組標準測試
python scripts/run_benchmark.py --host 10.110.134.151 --port 5000

# 僅執行吞吐量和延遲測試（跳過長文本組）
python scripts/run_benchmark.py --host 10.110.134.151 --port 5000 --groups throughput latency

# 指定自訂輸出目錄
python scripts/run_benchmark.py --host 10.110.134.151 --port 5000 --output-dir /tmp/bench_results
```