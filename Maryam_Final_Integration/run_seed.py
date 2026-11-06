"""Run main_final_integration.py for 10 seeds in parallel.
Save in the same folder as main_final_integration.py, then:  python run_seeds.py
Output goes to results/timeseries/seed_<n>/ (the folder plot_timeseries.py reads).
"""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [1, 3, 5, 7, 9, 10, 11, 15, 17, 20]
DEPTH_NOISE = 0.003          # metres (3 mm); use the same value for every seed
OUT_ROOT = "results/timeseries"

def run(seed):
    out = f"{OUT_ROOT}/seed_{seed}"
    full = os.path.join(HERE, out)
    if os.path.exists(os.path.join(full, "trajectory.csv")):
        print(f"skip seed {seed} (already done)", flush=True)
        return seed, 0
    print(f"start seed {seed}", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, "main_final_integration.py"),
                        "r,b,y", "--headless", "--seed", str(seed),
                        "--depth-noise", str(DEPTH_NOISE), "--output", out], cwd=HERE)
    print(f"done seed {seed} (exit code {r.returncode})", flush=True)
    return seed, r.returncode

if __name__ == "__main__":
    workers = max(1, min(len(SEEDS), (os.cpu_count() or 2) - 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, SEEDS))
    failed = [s for s, c in results if c != 0]
    print(f"\n{len(SEEDS) - len(failed)} of {len(SEEDS)} seeds succeeded")
    if failed:
        print("FAILED seeds:", failed)