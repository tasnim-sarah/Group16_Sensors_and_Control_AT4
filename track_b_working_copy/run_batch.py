#!/usr/bin/env python3
"""Run batches of headless sorting-robot trials, then summarise them.

Run it from the folder that holds test.py:

  python run_batch.py --suite nominal --seeds 1          # smoke test: one run
  python run_batch.py --suite nominal --seeds 1-30       # the nominal batch
  python run_batch.py --suite varied  --seeds 1-30       # noise / order / red-cube sweeps
  python run_batch.py --suite lqr     --seeds 1-10       # LQR weight sweep (nominal included)
  python run_batch.py --suite nominal --seeds 1-30 --jobs 4
  python run_batch.py --suite all --dry-run              # list conditions and the run count
  python run_batch.py --suite nominal --resume results/nominal-20261105-120000

Each batch gets its own folder, results/<suite>-<timestamp>/, holding:
  runs/<label>__seed<N>.csv   one result row per run
  logs/<label>__seed<N>.log   the full console output of that run
  results.csv                 all rows merged (this is what the summary reads)
  index.csv                   every scheduled run and whether it finished
  summary.md                  success rates and failure categories

A new batch never reuses old results, so a re-run after a code change is clean.
--resume DIR continues an interrupted batch in the same folder and skips finished runs.
"""
import argparse
import concurrent.futures as cf
import csv
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOMINAL_ORDER = "r,b,y"


def build_suites():
    """Each condition is (label, extra environment variables, pick order)."""
    nominal = [("nominal", {}, NOMINAL_ORDER)]
    # the nominal point (5e-4 rad, no depth noise, r,b,y, red 0.06 m / 6 s) comes from the nominal suite
    encoder = [(f"enc_{v:g}", {"ENCODER_NOISE_STD": repr(v)}, NOMINAL_ORDER)
               for v in (0.0, 1e-3, 2e-3, 4e-3)]
    depth = [(f"depth_{v:g}", {"DEPTH_NOISE_STD": repr(v)}, NOMINAL_ORDER)
             for v in (0.002, 0.005, 0.01)]
    orders = [(f"order_{''.join(p)}", {}, ",".join(p))
              for p in itertools.permutations("rby") if ",".join(p) != NOMINAL_ORDER]
    red = [
        ("red_amp_0.03", {"RED_AMPLITUDE": "0.03"}, NOMINAL_ORDER),
        ("red_amp_0.09", {"RED_AMPLITUDE": "0.09"}, NOMINAL_ORDER),
        ("red_period_4", {"RED_PERIOD": "4"}, NOMINAL_ORDER),
        ("red_period_9", {"RED_PERIOD": "9"}, NOMINAL_ORDER),
    ]
    # control-design sweep around the nominal weights (position 20, velocity 1, input 0.1)
    lqr = (
        [(f"lqr_pos_{v:g}", {"LQR_POS_W": repr(v)}, NOMINAL_ORDER) for v in (1.0, 5.0, 100.0, 500.0)]
        + [(f"lqr_vel_{v:g}", {"LQR_VEL_W": repr(v)}, NOMINAL_ORDER) for v in (0.1, 10.0)]
        + [(f"lqr_in_{v:g}", {"LQR_IN_W": repr(v)}, NOMINAL_ORDER) for v in (0.01, 1.0)]
    )
    suites = {"nominal": nominal, "encoder": encoder, "depth": depth,
              "orders": orders, "red": red, "lqr": nominal + lqr}
    suites["varied"] = encoder + depth + orders + red
    suites["all"] = nominal + suites["varied"]
    return suites


def parse_seeds(text):
    seeds = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            low, high = part.split("-", 1)
            seeds.extend(range(int(low), int(high) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def tail(path, lines=25):
    try:
        return "".join(Path(path).read_text(errors="replace").splitlines(True)[-lines:])
    except OSError:
        return "(no log)"


def main():
    suites = build_suites()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="nominal", choices=sorted(suites))
    ap.add_argument("--seeds", default="1-30", help="e.g. 1-30 or 1,5,9-12")
    ap.add_argument("--script", default=str(HERE / "test.py"))
    ap.add_argument("--jobs", type=int, default=1, help="parallel runs (default 1)")
    ap.add_argument("--timeout", type=float, default=1800, help="seconds per run")
    ap.add_argument("--out-root", default=str(HERE / "results"))
    ap.add_argument("--resume", default=None, help="existing batch folder to continue")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    conditions = suites[args.suite]
    tasks = [(label, env, order, seed)
             for (label, env, order) in conditions for seed in seeds]

    if args.dry_run:
        for label, env, order in conditions:
            print(f"  {label:16s} order={order:6s} env={env}")
        print(f"{len(conditions)} conditions x {len(seeds)} seeds = {len(tasks)} runs")
        return 0

    script = Path(args.script).resolve()
    if not script.exists():
        sys.exit(f"script not found: {script}")
    if args.resume:
        out_dir = Path(args.resume).resolve()
    else:
        out_dir = Path(args.out_root).resolve() / f"{args.suite}-{time.strftime('%Y%m%d-%H%M%S')}"
    runs_dir, logs_dir = out_dir / "runs", out_dir / "logs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Batch folder: {out_dir}")
    print(f"{len(conditions)} conditions x {len(seeds)} seeds = {len(tasks)} runs, "
          f"{args.jobs} at a time, {args.timeout:.0f}s limit per run")

    def execute(task):
        label, env_extra, order, seed = task
        stem = f"{label}__seed{seed}"
        csv_path = runs_dir / f"{stem}.csv"
        log_path = logs_dir / f"{stem}.log"
        if csv_path.exists():
            return task, "cached", 0.0, 0
        env = os.environ.copy()
        env.setdefault("MUJOCO_GL", "egl")
        env.update(HEADLESS="1", PYTHONUNBUFFERED="1", CUBE_SEED=str(seed),
                   ORDER=order, RUN_LABEL=label, RESULTS_CSV=str(csv_path))
        env.update(env_extra)
        started = time.perf_counter()
        try:
            with log_path.open("w") as log_file:
                proc = subprocess.run(
                    [sys.executable, str(script)], env=env, cwd=script.parent,
                    stdout=log_file, stderr=subprocess.STDOUT, timeout=args.timeout)
            status = "ok" if proc.returncode == 0 and csv_path.exists() else "crash"
            code = proc.returncode
        except subprocess.TimeoutExpired:
            status, code = "timeout", -1
        return task, status, time.perf_counter() - started, code

    results = {}
    consecutive_crashes = 0
    aborted = interrupted = False
    wall_start = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(execute, t) for t in tasks]
        try:
            for done, future in enumerate(cf.as_completed(futures), 1):
                task, status, seconds, code = future.result()
                label, _, _, seed = task
                results[(label, seed)] = (status, seconds, code)
                elapsed = time.perf_counter() - wall_start
                eta = elapsed / done * (len(tasks) - done)
                print(f"[{done:3d}/{len(tasks)}] {label:16s} seed {seed:<3d} "
                      f"{status:7s} {seconds:6.1f}s   elapsed {elapsed/60:5.1f} min, "
                      f"~{eta/60:5.1f} min left", flush=True)
                if status == "crash":
                    consecutive_crashes += 1
                elif status != "cached":
                    consecutive_crashes = 0
                if consecutive_crashes >= 3:
                    aborted = True
                    print("\n3 runs in a row crashed, so stopping. Last log lines of "
                          f"{label} seed {seed}:\n")
                    print(tail(logs_dir / f"{label}__seed{seed}.log"))
                    for f in futures:
                        f.cancel()
                    break
        except KeyboardInterrupt:
            interrupted = True
            for f in futures:
                f.cancel()

    # index of every scheduled run, and the merged results
    index_path = out_dir / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "seed", "order", "status", "seconds", "returncode"])
        for label, _, order, seed in tasks:
            status, seconds, code = results.get((label, seed), ("not run", 0.0, ""))
            writer.writerow([label, seed, order, status, f"{seconds:.1f}", code])

    merged, fieldnames = [], []
    for label, _, _, seed in tasks:
        path = runs_dir / f"{label}__seed{seed}.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    merged.append(row)
                    for key in row:
                        if key not in fieldnames:
                            fieldnames.append(key)
    results_path = out_dir / "results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(merged)
    print(f"\nMerged {len(merged)} result rows into {results_path}")

    summariser = HERE / "summarise_runs.py"
    if merged and summariser.exists():
        subprocess.run([sys.executable, str(summariser), str(results_path),
                        "--index", str(index_path), "--md", str(out_dir / "summary.md")])
    elif merged:
        print("summarise_runs.py not found next to run_batch.py; skipping the summary.")

    if interrupted:
        print(f"\nInterrupted. Continue with:  python run_batch.py --suite {args.suite} "
              f"--seeds {args.seeds} --resume {out_dir}")
        return 130
    return 2 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())