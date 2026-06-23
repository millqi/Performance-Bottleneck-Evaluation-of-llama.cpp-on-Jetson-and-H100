#!/usr/bin/env python3
"""
bench_concurrent.py — Concurrent load generator for llama-server

Fires N parallel completion requests and measures:
  - TTFT (Time to First Token) per request
  - ITL  (Inter-Token Latency, average) per request
  - Total generation time per request
  - Aggregate throughput (tokens/sec across all concurrent requests)

  [ADDED] Power & cost efficiency metrics:
  - avg_power_w            : mean GPU wattage during the batch (W)
  - energy_j_per_token     : Joules per generated token (J/tok)
  - elec_cost_usd_per_token: electricity cost per token (USD/tok)
  - rental_cost_usd_per_token: H100 cloud-rental cost per token (USD/tok)

  [ADDED] Multi-platform power backend:
  - --power-backend nvidia-smi  : for H100 (default)
  - --power-backend tegrastats   : for Jetson Orin

Usage:
  # H100
  python3 bench_concurrent.py \
      --url http://localhost:8080 \
      --prompt-tokens 512 \
      --gen-tokens 128 \
      --concurrency 16 \
      --repeats 5 \
      --output results.json \
      --gpu-index 0 \
      --power-backend nvidia-smi \
      --elec-price 0.10 \
      --rental-price 2.50

  # Jetson Orin
  python3 bench_concurrent.py \
      --url http://localhost:8080 \
      --prompt-tokens 128 \
      --gen-tokens 128 \
      --concurrency 8 \
      --repeats 5 \
      --output results.json \
      --power-backend tegrastats \
      --elec-price 0.10

Requirements:
  pip install aiohttp
"""

import argparse
import asyncio
import json
import time
import statistics
import threading
import subprocess
import re
import aiohttp


# ==============================================================================
# Background power sampler (supports nvidia-smi and tegrastats)
# ==============================================================================

def sample_power_nvidia_smi(gpu_index: int, interval: float,
                             readings: list, stop_event: threading.Event):
    """H100 path: poll nvidia-smi every `interval` seconds."""
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i", str(gpu_index),
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
            readings.append(float(out))
        except Exception:
            pass
        stop_event.wait(interval)


def sample_power_tegrastats(interval: float,
                             readings: list, stop_event: threading.Event):
    """
    Jetson path: launch tegrastats as a subprocess and read its streaming
    output continuously. Extracts VDD_GPU_SOC (mW) and converts to W.

    tegrastats output example:
      ... VDD_GPU_SOC 15000mW/15000mW VDD_CPU_CV 3000mW/3000mW ...
    """
    interval_ms = str(int(interval * 1000))
    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", interval_ms],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        # tegrastats not available — leave readings empty
        return

    while not stop_event.is_set():
        line = proc.stdout.readline()
        if not line:
            break
        m = re.search(r'VDD_GPU_SOC (\d+)mW', line)
        if m:
            readings.append(float(m.group(1)) / 1000)  # mW → W

    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()


def sample_power(gpu_index: int, interval: float,
                 readings: list, stop_event: threading.Event,
                 backend: str = "nvidia-smi"):
    """Dispatcher: route to the correct backend."""
    if backend == "tegrastats":
        sample_power_tegrastats(interval, readings, stop_event)
    else:
        sample_power_nvidia_smi(gpu_index, interval, readings, stop_event)


# ==============================================================================
# Original helpers (unchanged)
# ==============================================================================

def build_prompt(n_tokens: int) -> str:
    """
    Build a prompt that will tokenize to approximately n_tokens.
    Uses a repeating pattern of common English words.
    Rough heuristic: ~1.3 words per token for LLaMA tokenizer.
    """
    words = "The quick brown fox jumps over the lazy dog and then runs around the park "
    target_words = int(n_tokens * 1.3)
    repeated = (words * (target_words // len(words.split()) + 1))
    return " ".join(repeated.split()[:target_words])


async def single_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    gen_tokens: int,
    request_id: int,
) -> dict:
    """Send one completion request with streaming, measure TTFT and ITL."""

    payload = {
        "prompt": prompt,
        "n_predict": gen_tokens,
        "stream": True,
        "temperature": 0.0,
        "cache_prompt": False,
    }

    tokens_received = 0
    ttft = None
    token_times = []
    t_start = time.perf_counter()

    try:
        async with session.post(f"{url}/completion", json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                return {
                    "request_id": request_id,
                    "error": f"HTTP {resp.status}: {error_text[:200]}",
                }

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
                    break

                if "content" in data and data["content"]:
                    tokens_received += 1

                    if ttft is None:
                        ttft = t_now - t_start

                    if len(token_times) > 0:
                        token_times.append(t_now - token_times[-1])
                    else:
                        token_times.append(t_now)  # placeholder

    except Exception as e:
        return {"request_id": request_id, "error": str(e)}

    t_end = time.perf_counter()
    total_time = t_end - t_start

    if len(token_times) > 1:
        itl_values = token_times[1:]
        avg_itl = statistics.mean(itl_values) if itl_values else 0
    else:
        avg_itl = 0

    return {
        "request_id": request_id,
        "tokens_generated": tokens_received,
        "total_time_s": round(total_time, 4),
        "ttft_s": round(ttft, 4) if ttft else None,
        "avg_itl_s": round(avg_itl, 4),
        "tok_per_sec": round(tokens_received / total_time, 2) if total_time > 0 else 0,
    }


# ==============================================================================
# run_batch — starts/stops power sampler around requests
# ==============================================================================

async def run_batch(
    url: str,
    prompt: str,
    gen_tokens: int,
    concurrency: int,
    gpu_index: int,
    power_backend: str,
) -> tuple[list, list]:
    """Fire `concurrency` requests in parallel and collect results."""

    power_readings: list[float] = []
    stop_event = threading.Event()
    power_thread = threading.Thread(
        target=sample_power,
        args=(gpu_index, 1.0, power_readings, stop_event, power_backend),
        daemon=True,
    )
    power_thread.start()

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            single_request(session, url, prompt, gen_tokens, i)
            for i in range(concurrency)
        ]
        results = await asyncio.gather(*tasks)

    stop_event.set()
    power_thread.join()

    return list(results), power_readings


# ==============================================================================
# summarize_batch — computes power & cost fields
# ==============================================================================

def summarize_batch(
    results: list,
    power_readings: list,
    electricity_usd_per_kwh: float = 0.10,
    h100_rental_usd_per_hr: float = 2.50,
) -> dict:
    """Compute aggregate stats from a batch of request results."""
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    if not successful:
        return {"error": "all requests failed", "failures": len(failed)}

    total_tokens = sum(r["tokens_generated"] for r in successful)
    max_time = max(r["total_time_s"] for r in successful)
    ttfts = [r["ttft_s"] for r in successful if r["ttft_s"] is not None]
    itls = [r["avg_itl_s"] for r in successful if r["avg_itl_s"] > 0]
    per_req_tps = [r["tok_per_sec"] for r in successful]

    summary = {
        "n_requests": len(successful),
        "n_failed": len(failed),
        "total_tokens": total_tokens,
        "wall_time_s": round(max_time, 4),
        "aggregate_tok_per_sec": round(total_tokens / max_time, 2) if max_time > 0 else 0,
        "avg_per_request_tok_per_sec": round(statistics.mean(per_req_tps), 2),
        "ttft_median_s": round(statistics.median(ttfts), 4) if ttfts else None,
        "ttft_p95_s": round(sorted(ttfts)[int(len(ttfts) * 0.95)] if ttfts else 0, 4),
        "itl_median_s": round(statistics.median(itls), 4) if itls else None,
    }

    if power_readings and total_tokens > 0:
        avg_power_w      = statistics.mean(power_readings)
        energy_j         = avg_power_w * max_time
        energy_j_per_tok = energy_j / total_tokens
        energy_kwh       = energy_j / 3_600_000
        elec_cost_per_tok   = (energy_kwh * electricity_usd_per_kwh) / total_tokens
        rental_cost_per_tok = (h100_rental_usd_per_hr / 3600 * max_time) / total_tokens

        summary.update({
            "avg_power_w":               round(avg_power_w, 2),
            "energy_j_per_token":        round(energy_j_per_tok, 6),
            "elec_cost_usd_per_token":   round(elec_cost_per_tok, 9),
            "rental_cost_usd_per_token": round(rental_cost_per_tok, 9),
        })
    else:
        summary.update({
            "avg_power_w":               None,
            "energy_j_per_token":        None,
            "elec_cost_usd_per_token":   None,
            "rental_cost_usd_per_token": None,
        })

    return summary


# ==============================================================================
# main
# ==============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Concurrent llama-server benchmark")
    parser.add_argument("--url",           default="http://localhost:8080")
    parser.add_argument("--prompt-tokens", type=int,   default=512)
    parser.add_argument("--gen-tokens",    type=int,   default=128)
    parser.add_argument("--concurrency",   type=int,   default=1)
    parser.add_argument("--repeats",       type=int,   default=5)
    parser.add_argument("--output",        default=None)
    parser.add_argument("--gpu-index",     type=int,   default=0,
                        help="GPU index (nvidia-smi only, ignored for tegrastats)")
    parser.add_argument("--power-backend", default="nvidia-smi",
                        choices=["nvidia-smi", "tegrastats"],
                        help="Power monitoring backend: nvidia-smi (H100) or tegrastats (Jetson)")
    parser.add_argument("--elec-price",    type=float, default=0.10,
                        help="Electricity price (USD/kWh)")
    parser.add_argument("--rental-price",  type=float, default=2.50,
                        help="H100 cloud rental price (USD/hr)")
    args = parser.parse_args()

    prompt = build_prompt(args.prompt_tokens)

    print(f"Config: prompt≈{args.prompt_tokens}tok, gen={args.gen_tokens}tok, "
          f"concurrency={args.concurrency}, repeats={args.repeats}")
    print(f"Power:  backend={args.power_backend}, gpu-index={args.gpu_index}, "
          f"elec={args.elec_price} USD/kWh, rental={args.rental_price} USD/hr")

    all_summaries = []
    for run in range(args.repeats):
        print(f"  Run {run + 1}/{args.repeats}...", end=" ", flush=True)

        results, power_readings = await run_batch(
            args.url, prompt, args.gen_tokens, args.concurrency,
            args.gpu_index, args.power_backend,
        )

        summary = summarize_batch(
            results, power_readings,
            electricity_usd_per_kwh=args.elec_price,
            h100_rental_usd_per_hr=args.rental_price,
        )

        print(
            f"throughput={summary.get('aggregate_tok_per_sec', 'N/A')} tok/s  "
            f"power={summary.get('avg_power_w', 'N/A')} W  "
            f"J/tok={summary.get('energy_j_per_token', 'N/A')}"
        )
        all_summaries.append(summary)

    throughputs = [s["aggregate_tok_per_sec"] for s in all_summaries if "error" not in s]
    ttfts       = [s["ttft_median_s"]         for s in all_summaries if s.get("ttft_median_s")]
    powers      = [s["avg_power_w"]           for s in all_summaries if s.get("avg_power_w") is not None]
    eff         = [s["energy_j_per_token"]    for s in all_summaries if s.get("energy_j_per_token") is not None]

    final = {
        "config": {
            "prompt_tokens":            args.prompt_tokens,
            "gen_tokens":               args.gen_tokens,
            "concurrency":              args.concurrency,
            "repeats":                  args.repeats,
            "gpu_index":                args.gpu_index,
            "power_backend":            args.power_backend,
            "elec_price_usd_per_kwh":   args.elec_price,
            "rental_price_usd_per_hr":  args.rental_price,
        },
        "median_throughput_tok_per_sec": round(statistics.median(throughputs), 2) if throughputs else None,
        "median_ttft_s":                 round(statistics.median(ttfts), 4)        if ttfts       else None,
        "median_power_w":                round(statistics.median(powers), 2)       if powers      else None,
        "median_energy_j_per_token":     round(statistics.median(eff), 6)          if eff         else None,
        "all_runs": all_summaries,
    }

    print(f"\n  Median throughput:  {final['median_throughput_tok_per_sec']} tok/s")
    print(f"  Median TTFT:        {final['median_ttft_s']} s")
    print(f"  Median power:       {final['median_power_w']} W")
    print(f"  Median J/token:     {final['median_energy_j_per_token']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(final, f, indent=2)
        print(f"  Saved to: {args.output}")
    else:
        print(json.dumps(final, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
