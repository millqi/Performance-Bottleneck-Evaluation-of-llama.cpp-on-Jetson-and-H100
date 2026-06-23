#!/bin/bash
# ==============================================================================
# Jetson Orin Thesis Baseline Profiling Script
#
# Phase 1: llama-bench  — Single-sequence profiling (pp & tg speed)
# Phase 2: llama-server — Concurrent request profiling (throughput & latency)
#
# [JETSON] Key differences from h100_baseline.sh:
#   - No CUDA_VISIBLE_DEVICES (Jetson has one unified GPU)
#   - Power monitoring via tegrastats instead of nvidia-smi
#   - Reduced concurrency levels (unified memory, OOM risk at high concurrency)
#   - Adjusted paths for Jetson filesystem
# ==============================================================================

# --- 1. Paths ---
BIN_DIR="/home/yifan/llama.cpp/build/bin"
MODEL_DIR="/home/yifan/llama.cpp/models"

MODEL_Q4="${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
MODEL_Q8="${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
MODEL_BF16="${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-bf16.gguf"

MODELS=("$MODEL_Q4" "$MODEL_Q8" "$MODEL_BF16")
MODEL_NAMES=("Q4_K_M" "Q8_0" "BF16")

# --- 2. Scenarios (Prompt:Gen) ---
# S1: Short QA          — balanced workload
# S2: RAG/Summarization — prefill-heavy (compute-bound)
# S3: Pure Prefill      — extreme compute-bound test
# S4: Long Generation   — extreme memory-bound test
SCENARIOS=("128:128" "2048:128" "4096:1" "128:1024")
SCENARIO_NAMES=("S1_ShortQA" "S2_RAG" "S3_PurePrefill" "S4_LongGen")

# --- 3. Common Parameters ---
REPEATS=5
RESULTS_DIR="${BIN_DIR}/results_baseline_jetson"
mkdir -p "$RESULTS_DIR"

# ==============================================================================
# Power monitoring helpers (tegrastats version)
#
# tegrastats outputs lines like:
#   ... VDD_GPU_SOC 15000mW/15000mW ...
# We extract VDD_GPU_SOC (GPU + SoC power) as the closest equivalent
# to nvidia-smi's power.draw on H100.
#
# Usage: same interface as h100_baseline.sh for easy comparison.
# ==============================================================================

start_power_log() {
    local log_file="$1"
    # Sample every 1000ms; grep out the GPU power field
    tegrastats --interval 1000 \
        | grep --line-buffered -o 'VDD_GPU_SOC [0-9]*mW' \
        > "$log_file" &
    echo $!
}

stop_power_log() {
    local pid="$1"
    local log_file="$2"
    local json_file="$3"
    local n_tokens="$4"

    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null

    python3 - "$log_file" "$json_file" "$n_tokens" << 'PYEOF'
import sys, json, statistics, os, re

log_path  = sys.argv[1]
json_path = sys.argv[2]
n_tokens  = int(sys.argv[3]) if sys.argv[3].isdigit() else 0

# Parse tegrastats power readings (mW → W)
readings = []
if os.path.exists(log_path):
    with open(log_path) as f:
        for line in f:
            m = re.search(r'(\d+)mW', line)
            if m:
                readings.append(float(m.group(1)) / 1000)  # mW → W

avg_power_w = round(statistics.mean(readings), 2) if readings else None

# Inject into existing JSON
try:
    with open(json_path) as f:
        data = json.load(f)
except Exception:
    data = {}

entries = data if isinstance(data, list) else [data]
for entry in entries:
    entry["avg_power_w"] = avg_power_w
    if avg_power_w is not None and n_tokens > 0:
        entry["energy_j_per_token_approx_note"] = (
            "Use: avg_power_w / tok_per_sec to get J/token in post-processing"
        )

with open(json_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"     [power] avg={avg_power_w} W  "
      f"(from {len(readings)} samples, {n_tokens} tokens)")
PYEOF
}

# ==============================================================================
# PHASE 1: llama-bench (single-sequence, no concurrency)
# ==============================================================================
echo "=========================================================="
echo " PHASE 1: llama-bench — Single-Sequence Profiling"
echo " Platform: Jetson Orin"
echo "=========================================================="

POWER_LOG="${RESULTS_DIR}/tmp_power.log"

for i in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$i]}"
    MODEL_NAME="${MODEL_NAMES[$i]}"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "  [SKIP] Model not found: $MODEL_PATH"
        continue
    fi

    echo ""
    echo ">>> Model: ${MODEL_NAME} <<<"

    for j in "${!SCENARIOS[@]}"; do
        IFS=':' read -r PROMPT GEN <<< "${SCENARIOS[$j]}"
        SCENARIO_NAME="${SCENARIO_NAMES[$j]}"
        N_TOKENS=$(( PROMPT + GEN ))

        # --- Run WITHOUT Flash Attention ---
        OUT_JSON="${RESULTS_DIR}/${MODEL_NAME}_${SCENARIO_NAME}_fa0.json"
        OUT_MD="${RESULTS_DIR}/${MODEL_NAME}_${SCENARIO_NAME}_fa0.md"

        echo "  -> ${SCENARIO_NAME} (pp=${PROMPT}, tg=${GEN}) [FA=off]"

        POWER_PID=$(start_power_log "$POWER_LOG")

        "${BIN_DIR}/llama-bench" \
            -m "$MODEL_PATH" \
            -ngl 99 \
            -fa 0 \
            -p "$PROMPT" \
            -n "$GEN" \
            -r "$REPEATS" \
            -o json \
            -oe md \
            > "$OUT_JSON" 2> "$OUT_MD"

        stop_power_log "$POWER_PID" "$POWER_LOG" "$OUT_JSON" "$N_TOKENS"
        echo "     -> Saved: $OUT_JSON"

        # --- Run WITH Flash Attention ---
        OUT_JSON="${RESULTS_DIR}/${MODEL_NAME}_${SCENARIO_NAME}_fa1.json"
        OUT_MD="${RESULTS_DIR}/${MODEL_NAME}_${SCENARIO_NAME}_fa1.md"

        echo "  -> ${SCENARIO_NAME} (pp=${PROMPT}, tg=${GEN}) [FA=on]"

        POWER_PID=$(start_power_log "$POWER_LOG")

        "${BIN_DIR}/llama-bench" \
            -m "$MODEL_PATH" \
            -ngl 99 \
            -fa 1 \
            -p "$PROMPT" \
            -n "$GEN" \
            -r "$REPEATS" \
            -o json \
            -oe md \
            > "$OUT_JSON" 2> "$OUT_MD"

        stop_power_log "$POWER_PID" "$POWER_LOG" "$OUT_JSON" "$N_TOKENS"
        echo "     -> Saved: $OUT_JSON"
    done
done

echo ""
echo "=========================================================="
echo " PHASE 1 COMPLETE"
echo " Results in: ${RESULTS_DIR}/"
echo "=========================================================="


# ==============================================================================
# PHASE 2: llama-server + concurrent load test
# [JETSON] Concurrency levels reduced: unified memory limits safe parallelism
# ==============================================================================
echo ""
echo "=========================================================="
echo " PHASE 2: llama-server — Concurrency Sweep"
echo " [JETSON] Max concurrency capped at 32 to avoid OOM"
echo "=========================================================="

# [JETSON] Reduced from (1 8 32 64 128)
CONCURRENCY_LEVELS=(1 8 32 64 128)
SERVER_PORT=8080

CONC_MODEL="$MODEL_Q4"
CONC_MODEL_NAME="Q4_K_M"
MAX_CTX=65536   # [JETSON] Reduced from 131072 (64GB unified memory is shared with CPU)

BENCH_SCRIPT="$(dirname "$0")/bench_concurrent.py"

if [ ! -f "$CONC_MODEL" ]; then
    echo "  [SKIP] Concurrency model not found: $CONC_MODEL"
    exit 0
fi

if [ ! -f "$BENCH_SCRIPT" ]; then
    echo "  [ERROR] bench_concurrent.py not found at: $BENCH_SCRIPT"
    exit 1
fi

CONC_SCENARIOS=("128:128" "128:1024")
CONC_SCENARIO_NAMES=("S1_ShortQA" "S4_LongGen")

for s in "${!CONC_SCENARIOS[@]}"; do
    IFS=':' read -r PROMPT GEN <<< "${CONC_SCENARIOS[$s]}"
    SNAME="${CONC_SCENARIO_NAMES[$s]}"

    for CONC in "${CONCURRENCY_LEVELS[@]}"; do
        echo ""
        echo "  -> ${CONC_MODEL_NAME} | ${SNAME} | concurrency=${CONC}"

        NEEDED_CTX=$(( CONC * (PROMPT + GEN + 64) ))
        CTX=$(( NEEDED_CTX > MAX_CTX ? MAX_CTX : NEEDED_CTX ))

        "${BIN_DIR}/llama-server" \
            -m "$CONC_MODEL" \
            -ngl 99 \
            -fa 1 \
            -c "$CTX" \
            --parallel "$CONC" \
            --port "$SERVER_PORT" \
            --log-disable \
            > /dev/null 2>&1 &
        SERVER_PID=$!

        echo "     Waiting for server (PID: $SERVER_PID)..."
        SERVER_READY=false
        for attempt in $(seq 1 30); do
            if curl -s "http://localhost:${SERVER_PORT}/health" 2>/dev/null | grep -q "ok"; then
                SERVER_READY=true
                break
            fi
            sleep 2
        done

        if [ "$SERVER_READY" = false ]; then
            echo "     [ERROR] Server failed to start. Possibly OOM. Skipping."
            kill $SERVER_PID 2>/dev/null
            wait $SERVER_PID 2>/dev/null
            continue
        fi

        OUT_FILE="${RESULTS_DIR}/conc_${CONC_MODEL_NAME}_${SNAME}_npl${CONC}.json"

        POWER_PID=$(start_power_log "$POWER_LOG")

        python3 "$BENCH_SCRIPT" \
            --url "http://localhost:${SERVER_PORT}" \
            --prompt-tokens "$PROMPT" \
            --gen-tokens "$GEN" \
            --concurrency "$CONC" \
            --repeats "$REPEATS" \
            --output "$OUT_FILE" \
            --power-backend tegrastats   # [JETSON]

        N_TOKENS=$(( CONC * GEN * REPEATS ))
        stop_power_log "$POWER_PID" "$POWER_LOG" "$OUT_FILE" "$N_TOKENS"

        echo "     -> Saved: $OUT_FILE"

        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
        sleep 2
    done
done

echo ""
echo "=========================================================="
echo " ALL PHASES COMPLETE"
echo " Results in: ${RESULTS_DIR}/"
echo ""
echo " Next steps:"
echo "   1. Parse JSON results for plotting"
echo "   2. Run nsys profile on 2-3 representative configs"
echo "   3. Compare results_baseline_jetson/ vs results_baseline/"
echo "      Key metric: J/token (energy efficiency across platforms)"
echo "=========================================================="
