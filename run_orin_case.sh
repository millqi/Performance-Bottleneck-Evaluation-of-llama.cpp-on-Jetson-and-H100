#!/usr/bin/env bash
# run_orin_case.sh — Orin deep profiling orchestrator
#
# Handles:
#   1. Power mode switching via nvpmodel
#   2. tegrastats full time-series log
#   3. llama-server startup / health check
#   4. bench_client_orin.py
#   5. Clean shutdown (always, even on error)
#
# Usage:
#   ./run_orin_case.sh \
#       --case-id      Orin_Q4_S1_50W_npl64 \
#       --power-mode   3 \
#       --model        /home/yifan/llama.cpp/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
#       --quantization Q4_K_M \
#       --scenario     S1 \
#       --prompt-tokens 128 \
#       --gen-tokens    128 \
#       --concurrency   64 \
#       --repeats       3
#
# Power mode IDs (Jetson AGX Orin):
#   0 = MAXN   1 = 15W   2 = 30W   3 = 50W

set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# Defaults
# ══════════════════════════════════════════════════════════════════════════════
CASE_ID=""
OUTPUT_DIR="results/orin"
POWER_MODE=0          # 0=MAX, 1=15W, 2=30W, 3=50W
MODEL=""
QUANTIZATION="Q4_K_M"
SCENARIO="S1"
FLASH_ATTN=1          # 1=on (fa1), 0=off (fa0)
PROMPT_TOKENS=128
GEN_TOKENS=128
CONCURRENCY=1
REPEATS=3
CTX_SIZE=""           # auto-calculated if empty
URL="http://localhost:8080"
PORT=8080

BIN_DIR="$(dirname "$0")"
SERVER_PID=""
TSTAT_PID=""

# ══════════════════════════════════════════════════════════════════════════════
# Cleanup — always runs on exit
# ══════════════════════════════════════════════════════════════════════════════
cleanup() {
    echo ""
    echo "[cleanup] Stopping processes..."
    [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true
    [[ -n "$TSTAT_PID" ]]  && kill "$TSTAT_PID"  2>/dev/null || true
    sleep 2
    [[ -n "$SERVER_PID" ]] && kill -9 "$SERVER_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    echo "[cleanup] Done."
}
trap cleanup EXIT

# ══════════════════════════════════════════════════════════════════════════════
# Parse args
# ══════════════════════════════════════════════════════════════════════════════
while [[ $# -gt 0 ]]; do
    case "$1" in
        --case-id)       CASE_ID="$2";       shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2";    shift 2 ;;
        --power-mode)    POWER_MODE="$2";    shift 2 ;;
        --model)         MODEL="$2";         shift 2 ;;
        --quantization)  QUANTIZATION="$2";  shift 2 ;;
        --scenario)      SCENARIO="$2";      shift 2 ;;
        --flash-attn)    FLASH_ATTN="$2";    shift 2 ;;
        --prompt-tokens) PROMPT_TOKENS="$2"; shift 2 ;;
        --gen-tokens)    GEN_TOKENS="$2";    shift 2 ;;
        --concurrency)   CONCURRENCY="$2";   shift 2 ;;
        --repeats)       REPEATS="$2";       shift 2 ;;
        --ctx-size)      CTX_SIZE="$2";      shift 2 ;;
        --port)          PORT="$2";          shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$CASE_ID" || -z "$MODEL" ]]; then
    echo "ERROR: --case-id and --model are required."
    exit 1
fi

# Auto ctx-size: concurrency × (prompt + gen + 64) with a floor of 4096
if [[ -z "$CTX_SIZE" ]]; then
    NEEDED=$(( CONCURRENCY * (PROMPT_TOKENS + GEN_TOKENS + 64) ))
    CTX_SIZE=$(( NEEDED > 4096 ? NEEDED : 4096 ))
fi

# Power mode name for display
case "$POWER_MODE" in
    0) POWER_NAME="MAX" ;;
    1) POWER_NAME="15W" ;;
    2) POWER_NAME="30W" ;;
    3) POWER_NAME="50W" ;;
    *) POWER_NAME="mode${POWER_MODE}" ;;
esac

CASE_DIR="${OUTPUT_DIR}/${CASE_ID}"
mkdir -p "$CASE_DIR"

# ══════════════════════════════════════════════════════════════════════════════
# Print config
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Case:        ${CASE_ID}"
echo "  Output:      ${CASE_DIR}"
echo "  Power mode:  ${POWER_NAME} (id=${POWER_MODE})"
echo "  Model:       ${QUANTIZATION}"
echo "  Scenario:    ${SCENARIO}  prompt=${PROMPT_TOKENS} gen=${GEN_TOKENS}"
echo "  Concurrency: ${CONCURRENCY}  ctx=${CTX_SIZE}  fa=${FLASH_ATTN}"
echo "  Repeats:     ${REPEATS}"
echo "════════════════════════════════════════════════════════════"
echo ""

# Write run_meta.json
python3 - << EOF
import json, time
meta = {
    "case_id":       "${CASE_ID}",
    "platform":      "Jetson AGX Orin",
    "model":         "${QUANTIZATION}",
    "model_path":    "${MODEL}",
    "scenario":      "${SCENARIO}",
    "power_mode_id": ${POWER_MODE},
    "power_mode":    "${POWER_NAME}",
    "flash_attn":    ${FLASH_ATTN},
    "prompt_tokens": ${PROMPT_TOKENS},
    "gen_tokens":    ${GEN_TOKENS},
    "concurrency":   ${CONCURRENCY},
    "ctx_size":      ${CTX_SIZE},
    "repeats":       ${REPEATS},
    "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
}
with open("${CASE_DIR}/run_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("  run_meta.json written.")
EOF

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Set power mode
# ══════════════════════════════════════════════════════════════════════════════
echo "  Setting power mode → ${POWER_NAME} (nvpmodel -m ${POWER_MODE})"
sudo nvpmodel -m "${POWER_MODE}"
sleep 5   # let clocks stabilize after mode switch
echo "  Power mode set."

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Start tegrastats full time-series log
# ══════════════════════════════════════════════════════════════════════════════
TSTAT_LOG="${CASE_DIR}/tegrastats.log"
tegrastats --interval 500 > "${TSTAT_LOG}" &
TSTAT_PID=$!
echo "  tegrastats started (PID ${TSTAT_PID}) → ${TSTAT_LOG}"

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Start llama-server
# ══════════════════════════════════════════════════════════════════════════════
SERVER_LOG="${CASE_DIR}/server.log"
"${BIN_DIR}/llama-server" \
    -m "${MODEL}" \
    --n-gpu-layers 99 \
    -fa "${FLASH_ATTN}" \
    -c "${CTX_SIZE}" \
    --parallel "${CONCURRENCY}" \
    --port "${PORT}" \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "  Server PID: ${SERVER_PID} → ${SERVER_LOG}"

# Wait until /health returns 200
echo -n "  Waiting for server"
DEADLINE=$(( $(date +%s) + 120 ))
READY=false
while [[ $(date +%s) -lt $DEADLINE ]]; do
    if curl -sf "${URL}/health" > /dev/null 2>&1; then
        READY=true
        break
    fi
    echo -n "."
    sleep 2
done
echo ""

if [[ "$READY" != true ]]; then
    echo "  ERROR: server did not become ready in 120s. Check ${SERVER_LOG}."
    exit 1
fi
echo "  Server ready."

# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Run benchmark client
# ══════════════════════════════════════════════════════════════════════════════
CLIENT_JSON="${CASE_DIR}/client_metrics.json"
echo ""
echo "  Running bench_client_orin.py..."

python3 "${BIN_DIR}/bench_client_orin.py" \
    --url            "${URL}" \
    --prompt-tokens  "${PROMPT_TOKENS}" \
    --gen-tokens     "${GEN_TOKENS}" \
    --concurrency    "${CONCURRENCY}" \
    --repeats        "${REPEATS}" \
    --output         "${CLIENT_JSON}"

echo "  bench_client_orin.py done → ${CLIENT_JSON}"

# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Stop server and tegrastats
# ══════════════════════════════════════════════════════════════════════════════
kill "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""
kill "$TSTAT_PID"  2>/dev/null || true
TSTAT_PID=""
sleep 2
echo "  Server and tegrastats stopped."

# ══════════════════════════════════════════════════════════════════════════════
# Step 6: Summary
# ══════════════════════════════════════════════════════════════════════════════
echo ""
python3 - << EOF
import json

def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict): return None
        d = d.get(k)
    return d

try:
    with open("${CLIENT_JSON}") as f:
        r = json.load(f)
except Exception as e:
    print(f"  Could not read results: {e}")
    exit()

cm = r
en = r.get("energy", {})
runs = r.get("all_runs", [])

# Aggregate prefill/decode from runs
pre_list = [run.get("server_prefill_tok_per_s", {}).get("median")
            for run in runs if run.get("server_prefill_tok_per_s", {}).get("median")]
dec_list = [run.get("server_decode_tok_per_s", {}).get("median")
            for run in runs if run.get("server_decode_tok_per_s", {}).get("median")]

import statistics
pre_med = round(statistics.median(pre_list), 2) if pre_list else None
dec_med = round(statistics.median(dec_list), 2) if dec_list else None

print("────────────────────────────────────────────────────────────")
print(f"  ${CASE_ID} complete")
print("────────────────────────────────────────────────────────────")
print(f"  [Power]   mode       : ${POWER_NAME}")
print(f"  [Client]  throughput : {cm.get('median_throughput_tok_per_sec')} tok/s")
print(f"  [Client]  TTFT p50   : {cm.get('median_ttft_s')} s")
print(f"  [Server]  prefill    : {pre_med} tok/s (median across runs)")
print(f"  [Server]  decode     : {dec_med} tok/s (median across runs)")
print(f"  [Energy]  avg power  : {en.get('avg_power_w')} W (VDD_GPU_SOC)")
print(f"  [Energy]  J/token    : {en.get('energy_j_per_token')}")
print("────────────────────────────────────────────────────────────")
EOF

echo ""
