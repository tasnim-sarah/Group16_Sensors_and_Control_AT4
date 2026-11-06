import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

seeds = [1, 3, 5, 7, 9, 10, 11, 15, 17, 20]
noise_levels = [0, 0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015]

def outdir(noise, seed):
    return f"results/robustness/noise_{noise}_seed_{seed}"

def done(noise, seed):
    d = outdir(noise, seed)
    return os.path.isdir(d) and len(os.listdir(d)) > 0

all_jobs = [(n, s) for n in noise_levels for s in seeds]
jobs = [j for j in all_jobs if not done(*j)]
print(f"{len(all_jobs) - len(jobs)} already done, running {len(jobs)} now")

def run(job):
    noise, seed = job
    print(f"Starting seed={seed}, noise={noise}", flush=True)
    r = subprocess.run([
        sys.executable, "main_final_integration.py", "r,b,y",
        "--headless",
        "--seed", str(seed),
        "--depth-noise", str(noise),
        "--output", outdir(noise, seed),
    ])
    print(f"Finished seed={seed}, noise={noise} (exit code {r.returncode})", flush=True)
    return job, r.returncode

workers = max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
with ThreadPoolExecutor(max_workers=workers) as pool:
    results = list(pool.map(run, jobs))

failed = [j for j, code in results if code != 0]
print(f"\n{len(jobs) - len(failed)} of {len(jobs)} runs succeeded")
for noise, seed in failed:
    print(f"FAILED: noise={noise}, seed={seed}")