#!/usr/bin/env python3
"""
bench_client_orin.py — Benchmark client for llama-server on Jetson Orin

Level 1 (client):  TTFT, ITL distribution, throughput
Level 2 (server):  prefill/decode split from HTTP response timings
Level 3 (system):  tegrastats power sampling (scalar avg for J/token)

Called by run_orin_case.sh. Do not run standalone for profiling.

Usage:
  python3 bench_client_orin.py \
      --url http://localhost:8080 \
      --prompt-tokens 128 \
      --gen-tokens 128 \
      --concurrency 64 \
      --repeats 3 \
      --output results/orin/CASE_DIR/client_metrics.json

Requirements:
  pip install aiohttp
"""

import argparse
import asyncio
import json
import re
import statistics
import subprocess
import threading
import time

import aiohttp


# ══════════════════════════════════════════════════════════════════════════════
# Level 3 — tegrastats power sampler
# ══════════════════════════════════════════════════════════════════════════════

def _tegrastats_sampler(interval_ms: int,
                         readings: list,
                         stop_event: threading.Event):
    """
    Launches tegrastats as a subprocess and reads VDD_GPU_SOC power (mW→W).
    Appends (timestamp, watts) tuples to readings.

    tegrastats line example:
      ... VDD_GPU_SOC 4500mW/15000mW VDD_CPU_CV 1200mW/10000mW ...
    """
    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", str(interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        return  # tegrastats not available

    while not stop_event.is_set():
        line = proc.stdout.readline()
        if not line:
            break
        m = re.search(r'VDD_GPU_SOC (\d+)mW', line)
        if m:
            readings.append((time.time(), float(m.group(1)) / 1000.0))

    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()


# ══════════════════════════════════════════════════════════════════════════════
# Prompt builder
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(n_tokens: int) -> str:
    words = "The quick brown fox jumps over the lazy dog and then runs around the park "
    target_words = int(n_tokens * 1.3)
    repeated = words * (target_words // len(words.split()) + 1)
    return " ".join(repeated.split()[:target_words])


# ══════════════════════════════════════════════════════════════════════════════
# Single request (fixed ITL + server timings from HTTP response)
# ══════════════════════════════════════════════════════════════════════════════

async def single_request(session: aiohttp.ClientSession,
                          url: str, prompt: str,
                          gen_tokens: int, request_id: int) -> dict:
    """
    ITL fix: last_token_time tracks absolute perf_counter of previous token.
    server_timings: captured from stop chunk timings field (per-request,
    always reliable regardless of log verbosity).
    """
    payload = {
        "prompt":       prompt,
        "n_predict":    gen_tokens,
        "stream":       True,
        "temperature":  0.0,
        "cache_prompt": False,
    }

    tokens_received  = 0
    ttft             = None
    last_token_time  = None
    itl_list         = []
    server_timings   = None
    t_start          = time.perf_counter()

    try:
        async with session.post(f"{url}/completion", json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                return {"request_id": request_id,
                        "error": f"HTTP {resp.status}: {text[:200]}"}

            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data: "):
                    continue
                data_str = decoded[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                t_now = time.perf_counter()

                if data.get("stop", False):
                    if "timings" in data:
                        t = data["timings"]
                        server_timings = {
                            "prefill_tokens":    t.get("prompt_n"),
                            "prefill_ms":        t.get("prompt_ms"),
                            "prefill_tok_per_s": t.get("prompt_per_second"),
                            "decode_tokens":     t.get("predicted_n"),
                            "decode_ms":         t.get("predicted_ms"),
                            "decode_tok_per_s":  t.get("predicted_per_second"),
                        }
                    break

                if data.get("content"):
                    tokens_received += 1
                    if ttft is None:
                        ttft            = t_now - t_start
                        last_token_time = t_now
                    else:
                        itl_list.append(round(t_now - last_token_time, 6))
                        last_token_time = t_now

    except Exception as e:
        return {"request_id": request_id, "error": str(e)}

    t_end      = time.perf_counter()
    total_time = t_end - t_start

    result = {
        "request_id":       request_id,
        "tokens_generated": tokens_received,
        "total_time_s":     round(total_time, 4),
        "ttft_s":           round(ttft, 4) if ttft is not None else None,
        "tok_per_sec":      round(tokens_received / total_time, 2) if total_time > 0 else 0,
        "server_timings":   server_timings,
    }

    if itl_list:
        sorted_itl = sorted(itl_list)
        result["itl_mean_s"]         = round(statistics.mean(itl_list), 6)
        result["itl_p50_s"]          = round(statistics.median(itl_list), 6)
        result["itl_p95_s"]          = round(sorted_itl[int(len(sorted_itl) * 0.95)], 6)
        result["itl_distribution_s"] = itl_list
    else:
        result["itl_mean_s"] = result["itl_p50_s"] = result["itl_p95_s"] = None
        result["itl_distribution_s"] = []

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Batch runner
# ══════════════════════════════════════════════════════════════════════════════

async def run_batch(url: str, prompt: str,
                    gen_tokens: int, concurrency: int) -> list:
    timeout   = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with aiohttp.ClientSession(timeout=timeout,
                                      connector=connector) as session:
        tasks = [single_request(session, url, prompt, gen_tokens, i)
                 for i in range(concurrency)]
        return list(await asyncio.gather(*tasks))


def summarize_batch(results: list) -> dict:
    successful = [r for r in results if "error" not in r]
    failed     = [r for r in results if "error" in r]

    if not successful:
        return {
            "error":    "all requests failed",
            "n_failed": len(failed),
            "failures": [r.get("error", "") for r in failed],
        }

    total_tokens = sum(r["tokens_generated"] for r in successful)
    max_time     = max(r["total_time_s"]     for r in successful)
    ttfts        = [r["ttft_s"] for r in successful if r["ttft_s"] is not None]
    per_req_tps  = [r["tok_per_sec"] for r in successful]

    all_itls = []
    for r in successful:
        all_itls.extend(r.get("itl_distribution_s", []))

    summary = {
        "n_requests":                  len(successful),
        "n_failed":                    len(failed),
        "total_tokens":                total_tokens,
        "wall_time_s":                 round(max_time, 4),
        "aggregate_tok_per_sec":       round(total_tokens / max_time, 2) if max_time > 0 else 0,
        "avg_per_request_tok_per_sec": round(statistics.mean(per_req_tps), 2),
        "ttft_mean_s":   round(statistics.mean(ttfts),   4) if ttfts else None,
        "ttft_median_s": round(statistics.median(ttfts), 4) if ttfts else None,
        "ttft_p95_s": (
            round(sorted(ttfts)[int(len(ttfts) * 0.95)], 4)
            if len(ttfts) >= 20
            else round(max(ttfts), 4) if ttfts else None
        ),
        "ttft_p95_note": ("true p95" if len(ttfts) >= 20
                          else "max (n<20)"),
    }

    if all_itls:
        sorted_itls = sorted(all_itls)
        summary["itl_mean_s"] = round(statistics.mean(all_itls), 6)
        summary["itl_p50_s"]  = round(statistics.median(all_itls), 6)
        summary["itl_p95_s"]  = round(sorted_itls[int(len(sorted_itls) * 0.95)], 6)
    else:
        summary["itl_mean_s"] = summary["itl_p50_s"] = summary["itl_p95_s"] = None

    # Level 2: server prefill/decode (aggregated across requests)
    prefill_tps = [r["server_timings"]["prefill_tok_per_s"]
                   for r in successful
                   if r.get("server_timings") and
                   r["server_timings"].get("prefill_tok_per_s")]
    decode_tps  = [r["server_timings"]["decode_tok_per_s"]
                   for r in successful
                   if r.get("server_timings") and
                   r["server_timings"].get("decode_tok_per_s")]

    def _med(lst):
        return round(statistics.median(lst), 2) if lst else None

    summary["server_prefill_tok_per_s"] = {
        "median": _med(prefill_tps),
        "mean":   round(statistics.mean(prefill_tps), 2) if prefill_tps else None,
        "n":      len(prefill_tps),
    }
    summary["server_decode_tok_per_s"] = {
        "median": _med(decode_tps),
        "mean":   round(statistics.mean(decode_tps), 2) if decode_tps else None,
        "n":      len(decode_tps),
    }

    if failed:
        summary["failure_errors"] = [r.get("error", "") for r in failed]

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(
        description="Orin benchmark client for llama-server")
    parser.add_argument("--url",           default="http://localhost:8080")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--gen-tokens",    type=int, default=128)
    parser.add_argument("--concurrency",   type=int, default=1)
    parser.add_argument("--repeats",       type=int, default=3)
    parser.add_argument("--output",        default=None)
    args = parser.parse_args()

    prompt = build_prompt(args.prompt_tokens)

    print(f"  prompt≈{args.prompt_tokens} tok | gen={args.gen_tokens} tok | "
          f"concurrency={args.concurrency} | repeats={args.repeats}")

    # Start tegrastats power sampler (500ms interval)
    power_readings: list = []
    stop_power = threading.Event()
    power_thread = threading.Thread(
        target=_tegrastats_sampler,
        args=(500, power_readings, stop_power),
        daemon=True,
    )
    power_thread.start()

    # Run repeats
    all_runs = []
    for run_i in range(args.repeats):
        print(f"    Run {run_i + 1}/{args.repeats}...", end=" ", flush=True)
        results = await run_batch(args.url, prompt, args.gen_tokens,
                                  args.concurrency)
        summary = summarize_batch(results)
        tput  = summary.get("aggregate_tok_per_sec", "N/A")
        ttft  = summary.get("ttft_median_s", "N/A")
        nfail = summary.get("n_failed", 0)
        pre   = summary.get("server_prefill_tok_per_s", {}).get("median", "N/A")
        dec   = summary.get("server_decode_tok_per_s",  {}).get("median", "N/A")
        print(f"throughput={tput} tok/s  ttft_p50={ttft} s  "
              f"prefill={pre} tok/s  decode={dec} tok/s  failed={nfail}")
        all_runs.append(summary)

    # Stop power sampler
    stop_power.set()
    power_thread.join()

    # Energy summary
    watt_values      = [w for _, w in power_readings]
    avg_power_w      = statistics.mean(watt_values) if watt_values else None
    valid_runs       = [r for r in all_runs if "error" not in r]
    total_wall_time  = sum(r["wall_time_s"]  for r in valid_runs)
    total_tokens_all = sum(r["total_tokens"] for r in valid_runs)
    energy_j         = avg_power_w * total_wall_time if avg_power_w else None
    j_per_token      = (energy_j / total_tokens_all
                        if energy_j and total_tokens_all else None)

    # Aggregate across repeats
    throughputs = [r["aggregate_tok_per_sec"] for r in valid_runs]
    ttfts_med   = [r["ttft_median_s"] for r in valid_runs
                   if r.get("ttft_median_s")]

    final = {
        "median_throughput_tok_per_sec": (round(statistics.median(throughputs), 2)
                                          if throughputs else None),
        "median_ttft_s":                 (round(statistics.median(ttfts_med), 4)
                                          if ttfts_med else None),
        "energy": {
            "avg_power_w":        round(avg_power_w, 2) if avg_power_w else None,
            "n_power_samples":    len(watt_values),
            "energy_j_per_token": round(j_per_token, 6) if j_per_token else None,
            "source":             "VDD_GPU_SOC via tegrastats",
        },
        "all_runs": all_runs,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(final, f, indent=2)
    else:
        print(json.dumps(final, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
