import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request


RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run experiment-agent for all discovered stocks over multiple rounds "
            "with retry/backoff and pacing to reduce API/model rate limits.\n\n"
            "Why this is slow: each stock triggers 1 LLM call plus 3 POST /run-forecast "
            "calls (each trains XGB several times). Defaults add ~1.5s between stocks and "
            "12s between rounds.\n\n"
            "Speed tips: use --rounds 1 for a single pass; lower --inter-request-delay and "
            "--inter-round-delay when your LLM quota allows; start the forecast API with "
            "FORECAST_ENGINES_SKIP_CV=1 to skip time-series CV inside /run-forecast "
            "(faster, slightly less info in logs)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:3000/api/experiment-agent",
        help="Orchestrator endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Forecast horizon to pass to the API (default: %(default)s)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="How many full passes across all stocks (default: %(default)s)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="HTTP timeout per request in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Max retries per stock request (default: %(default)s)",
    )
    parser.add_argument(
        "--base-backoff",
        type=float,
        default=2.0,
        help="Base exponential backoff in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.75,
        help="Random jitter seconds added to delays/backoff (default: %(default)s)",
    )
    parser.add_argument(
        "--inter-request-delay",
        type=float,
        default=1.5,
        help="Delay between stocks in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--inter-round-delay",
        type=float,
        default=12.0,
        help="Delay between rounds in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "raw",
        help="Directory used to discover stock CSVs (default: services/data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "experiments_runs",
        help="Where to save JSONL results and summary JSON (default: services/data/experiments_runs)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable per-round stock order shuffling",
    )
    parser.add_argument(
        "--resume-file",
        type=Path,
        default=None,
        help=(
            "Resume from an existing JSONL run log. Completed (round,target) entries "
            "are skipped and new records are appended to the same file."
        ),
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def discover_stocks(raw_dir: Path) -> list[str]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    stocks = sorted(path.stem.upper() for path in raw_dir.glob("*.csv"))
    if not stocks:
        raise RuntimeError(f"No CSV files found in: {raw_dir}")
    return stocks


def parse_retry_after(header_value: str | None) -> float | None:
    if not header_value:
        return None
    try:
        return max(0.0, float(header_value.strip()))
    except ValueError:
        return None


def post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any], dict[str, str]]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        data = json.loads(raw) if raw else {}
        return resp.status, data, dict(resp.headers.items())


def run_with_retries(
    endpoint: str,
    payload: dict[str, Any],
    timeout: float,
    max_retries: int,
    base_backoff: float,
    jitter: float,
) -> dict[str, Any]:
    attempt = 0

    while True:
        started = time.perf_counter()
        try:
            status_code, data, headers = post_json(endpoint, payload, timeout)
            duration = time.perf_counter() - started
            return {
                "ok": True,
                "status_code": status_code,
                "data": data,
                "attempt": attempt,
                "duration_sec": round(duration, 3),
                "headers": headers,
            }
        except error.HTTPError as exc:
            duration = time.perf_counter() - started
            response_body = exc.read().decode("utf-8", errors="replace")
            retry_after = parse_retry_after(exc.headers.get("Retry-After"))
            retryable = exc.code in RETRYABLE_HTTP_CODES

            if retryable and attempt < max_retries:
                sleep_for = max(
                    retry_after or 0.0,
                    base_backoff * (2 ** attempt) + random.uniform(0, jitter),
                )
                time.sleep(sleep_for)
                attempt += 1
                continue

            return {
                "ok": False,
                "status_code": exc.code,
                "error": response_body,
                "attempt": attempt,
                "duration_sec": round(duration, 3),
            }
        except (error.URLError, TimeoutError, ConnectionError) as exc:
            duration = time.perf_counter() - started
            if attempt < max_retries:
                sleep_for = base_backoff * (2 ** attempt) + random.uniform(0, jitter)
                time.sleep(sleep_for)
                attempt += 1
                continue

            return {
                "ok": False,
                "status_code": None,
                "error": str(exc),
                "attempt": attempt,
                "duration_sec": round(duration, 3),
            }


def load_completed_pairs(path: Path) -> tuple[set[tuple[int, str]], dict[str, int]]:
    completed: set[tuple[int, str]] = set()
    stats = {
        "completed": 0,
        "success": 0,
        "failure": 0,
        "rate_limited": 0,
    }

    if not path.exists():
        return completed, stats

    with path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSONL at line {line_number} in {path}")
                continue

            round_number = record.get("round")
            target = str(record.get("target", "")).upper()
            if not isinstance(round_number, int) or not target:
                continue

            key = (round_number, target)
            if key in completed:
                continue

            completed.add(key)
            stats["completed"] += 1

            if record.get("ok"):
                stats["success"] += 1
            else:
                stats["failure"] += 1
                if record.get("status_code") == 429:
                    stats["rate_limited"] += 1

    return completed, stats


def main() -> None:
    args = parse_args()

    stocks = discover_stocks(args.raw_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_jsonl = args.resume_file if args.resume_file else args.output_dir / f"full_experiment_{run_id}.jsonl"
    summary_json = args.output_dir / f"full_experiment_{run_id}_summary.json"

    total_requests = len(stocks) * args.rounds
    print(f"Discovered {len(stocks)} stocks.")
    print(f"Running {args.rounds} rounds => {total_requests} total requests")
    print(f"Endpoint: {args.endpoint}")
    print(f"Saving per-request logs to: {output_jsonl}")

    completed_pairs, resumed_stats = load_completed_pairs(output_jsonl)
    if args.resume_file:
        print(
            "Resuming from existing log: "
            f"completed={resumed_stats['completed']}, "
            f"success={resumed_stats['success']}, "
            f"failure={resumed_stats['failure']}"
        )

    completed = resumed_stats["completed"]
    success_count = resumed_stats["success"]
    failure_count = resumed_stats["failure"]
    rate_limited_count = resumed_stats["rate_limited"]
    interrupted = False

    with output_jsonl.open("a", encoding="utf-8") as out:
        for round_number in range(1, args.rounds + 1):
            round_stocks = stocks[:]
            if not args.no_shuffle:
                random.shuffle(round_stocks)

            print(f"\n=== Round {round_number}/{args.rounds} ===")

            for stock in round_stocks:
                if (round_number, stock) in completed_pairs:
                    print(
                        f"[{completed}/{total_requests}] Skipping {stock} "
                        f"(already completed for round {round_number})"
                    )
                    continue

                completed += 1
                payload = {"target": stock, "horizon": args.horizon}

                print(f"[{completed}/{total_requests}] Running experiments for {stock} ...")

                try:
                    result = run_with_retries(
                        endpoint=args.endpoint,
                        payload=payload,
                        timeout=args.request_timeout,
                        max_retries=args.max_retries,
                        base_backoff=args.base_backoff,
                        jitter=args.jitter,
                    )
                except KeyboardInterrupt:
                    interrupted = True
                    print("\nInterrupted by user. Finalizing partial summary...")
                    break

                record = {
                    "timestamp": now_iso(),
                    "round": round_number,
                    "target": stock,
                    "horizon": args.horizon,
                    "attempts": result["attempt"] + 1,
                    "status_code": result.get("status_code"),
                    "duration_sec": result.get("duration_sec"),
                    "ok": result["ok"],
                }

                if result["ok"]:
                    success_count += 1
                    data = result.get("data", {})
                    record["best"] = data.get("best")
                    record["response"] = data
                    print(f"  ✅ OK | best={data.get('best')}")
                else:
                    failure_count += 1
                    if result.get("status_code") == 429:
                        rate_limited_count += 1
                    record["error"] = result.get("error")
                    print(
                        f"  ❌ Failed | status={result.get('status_code')} | "
                        f"error={result.get('error')}"
                    )

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                completed_pairs.add((round_number, stock))

                if completed < total_requests and args.inter_request_delay > 0:
                    sleep_for = args.inter_request_delay + random.uniform(0, args.jitter)
                    time.sleep(sleep_for)

            if interrupted:
                break

            if round_number < args.rounds and args.inter_round_delay > 0:
                time.sleep(args.inter_round_delay)

    summary = {
        "run_id": run_id,
        "endpoint": args.endpoint,
        "horizon": args.horizon,
        "rounds": args.rounds,
        "stocks_count": len(stocks),
        "total_requests": total_requests,
        "success_count": success_count,
        "failure_count": failure_count,
        "rate_limited_count": rate_limited_count,
        "success_rate": round(success_count / total_requests, 4) if total_requests else 0,
        "raw_dir": str(args.raw_dir),
        "output_jsonl": str(output_jsonl),
        "interrupted": interrupted,
        "finished_at": now_iso(),
    }

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Finished ===")
    print(json.dumps(summary, indent=2))
    print(f"Summary saved to: {summary_json}")


if __name__ == "__main__":
    main()