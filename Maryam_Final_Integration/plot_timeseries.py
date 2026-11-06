"""Time-series plots over many seeds, built from the files main_final_integration.py writes:
   trajectory.csv       joint observer + LQR (estimate_q, truth_q, sigma_q, command, torque, ...)
   object_tracking.csv  cube Kalman filter (filtered vs truth_position, sigma)
   summary.json         run config (LQR weights)
Run after run_seeds.py:  python plot_timeseries.py
"""
import glob, json, os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import chi2

ROOT = "results/timeseries"      # folder containing seed_<n> folders
GRID_DT = 0.1                    # s, common time grid for the joint/LQR data
BIN_S = 0.5                      # s, time bins for the (irregular) cube-tracker data
SMOOTH_S = 0.5                   # s, moving window for RMSE/ANEES (set 0 to disable)
SHOW_SEED = None                 # seed for the truth-vs-estimate panels (None = first)
U_SOURCE = "command"             # LQR input used in the cost: "command" or "torque"
CUBE_LABEL = None                # e.g. "Red Cube" to plot one cube only (None = all)

J = range(1, 7)
def cols(prefix): return [f"{prefix}_{i}" for i in J]

def load():
    runs = {}
    for d in sorted(glob.glob(f"{ROOT}/seed_*")):
        seed = int(d.split("seed_")[1])
        traj, obj, summ = (os.path.join(d, n) for n in ("trajectory.csv", "object_tracking.csv", "summary.json"))
        if not os.path.exists(traj):
            print(f"seed {seed}: no trajectory.csv, skipped"); continue
        r = {"traj": pd.read_csv(traj),
             "obj": pd.read_csv(obj) if os.path.exists(obj) else pd.DataFrame(),
             "summary": json.load(open(summ)) if os.path.exists(summ) else {}}
        runs[seed] = r
    if not runs: raise SystemExit(f"No runs found under {ROOT}")
    return runs

def smooth(a, n):
    """moving average along the last axis"""
    if n <= 1: return a
    k = np.ones(n) / n
    return np.apply_along_axis(lambda v: np.convolve(np.pad(v, (n//2, n-1-n//2), mode="edge"), k, "valid"), -1, a)

COL = {s: plt.cm.tab10(i % 10) for i, s in enumerate(sorted(int(d.split("seed_")[1]) for d in glob.glob(f"{ROOT}/seed_*")))}

def seed_lines(ax, x, arr, seeds, lw=.9, alpha=.6):
    """one thin line per seed, same colour for a seed in every figure"""
    for s, r in zip(seeds, arr):
        ax.plot(x, r, color=COL.get(s, "gray"), lw=lw, alpha=alpha, label=f"seed {s}")

def band(ax, t, arr, label, color):
    m, s = arr.mean(0), arr.std(0)
    ax.plot(t, m, color=color, label=label); ax.fill_between(t, m - s, m + s, color=color, alpha=.25)

runs = load(); seeds = list(runs); N = len(seeds)
SHOW = SHOW_SEED if SHOW_SEED in runs else seeds[0]
cfg = runs[SHOW]["summary"].get("config", {})
QW, VW, RW = cfg.get("lqr_position", 20), cfg.get("lqr_velocity", 1), cfg.get("lqr_input", .1)

dur = {s: runs[s]["traj"]["time_s"].max() for s in seeds}
print("Run durations (s):", {s: round(d, 1) for s, d in dur.items()})
T_END = min(dur.values())
t = np.arange(0, T_END, GRID_DT)
nsm = int(round(SMOOTH_S / GRID_DT))
print(f"{N} seeds; plotting 0..{T_END:.1f} s (shortest run). Task stages are NOT aligned across seeds.")

def on_grid(s, names):
    df = runs[s]["traj"]
    return np.stack([np.interp(t, df["time_s"], df[c]) for c in names], axis=-1)   # (T, k)

# ---------------- joint observer + LQR quantities, per seed ----------------
est, tru, sig, ref = (np.stack([on_grid(s, cols(p)) for s in seeds]) for p in
                      ("estimate_q", "truth_q", "sigma_q", "reference"))     # (N, T, 6)
dq = np.stack([on_grid(s, cols("truth_dq")) for s in seeds])
u  = np.stack([on_grid(s, cols(U_SOURCE)) for s in seeds])
sat = np.stack([on_grid(s, ["saturated"])[:, 0] for s in seeds])               # (N, T)
e_q = est - tru
nees_q = ((e_q / np.maximum(sig, 1e-12)) ** 2).sum(-1)                         # (N, T), dof 6

# ---------------- cube tracker (object_tracking.csv), binned in time ----------------
edges = np.arange(0, T_END + BIN_S, BIN_S); tb = edges[:-1] + BIN_S / 2
nb = len(tb)
obj_se = np.full((N, nb), np.nan); obj_nees = np.full((N, nb), np.nan); obj_cnt = np.zeros((N, nb))
obj_norm = np.full((N, nb), np.nan)
have_obj = all(len(runs[s]["obj"]) for s in seeds)
if have_obj:
    for i, s in enumerate(seeds):
        o = runs[s]["obj"]
        if CUBE_LABEL: o = o[o["label"] == CUBE_LABEL]
        e = o[["filtered_x", "filtered_y", "filtered_z"]].values - o[["truth_position_x", "truth_position_y", "truth_position_z"]].values
        sg = np.maximum(o[["sigma_x", "sigma_y", "sigma_z"]].values, 1e-9)
        b = np.digitize(o["time_s"].values, edges) - 1
        for k in range(nb):
            m = b == k
            if m.any():
                obj_se[i, k] = (e[m] ** 2).sum(1).mean(); obj_norm[i, k] = np.linalg.norm(e[m], axis=1).mean()
                obj_nees[i, k] = ((e[m] / sg[m]) ** 2).sum(1).mean(); obj_cnt[i, k] = m.sum()

# ======================= 1. RMSE over time =======================
fig, axes = plt.subplots(2 if have_obj else 1, 1, figsize=(10, 7 if have_obj else 4), squeeze=False); axes = axes[:, 0]
per_seed = np.sqrt(smooth((e_q ** 2).mean(-1), nsm))
seed_lines(axes[0], t, per_seed * 1e3, seeds)
ens = np.sqrt(smooth((e_q ** 2).mean(-1).mean(0), nsm))
axes[0].plot(t, ens * 1e3, "k", lw=2.5, label=f"Ensemble RMSE ({N} seeds)")
axes[0].set(ylabel="Joint estimation RMSE (mrad)", title="Joint observer: RMSE over time"); axes[0].grid(True)
if have_obj:
    seed_lines(axes[1], tb, np.sqrt(obj_se / 3) * 1e3, seeds)
    axes[1].plot(tb, np.sqrt(np.nanmean(obj_se, 0) / 3) * 1e3, "k", lw=2.5, label="Ensemble RMSE per axis")
    axes[1].set(ylabel="Cube position RMSE (mm per axis)", title="Cube Kalman filter: RMSE over time"); axes[1].grid(True)
for ax in axes: ax.legend(ncol=4, fontsize=7)
axes[-1].set_xlabel("Time (s)"); fig.tight_layout(); fig.savefig("rmse_over_time.png", dpi=150)

# ======================= 2. ANEES over time =======================
fig, axes = plt.subplots(2 if have_obj else 1, 1, figsize=(10, 7 if have_obj else 4), squeeze=False); axes = axes[:, 0]
seed_lines(axes[0], t, smooth(nees_q, nsm) / 6, seeds, lw=.7, alpha=.45)     # per-seed NEES/n
a = smooth(nees_q.mean(0), nsm) / 6
dof = N * 6 * max(nsm, 1)
lo, hi = chi2.ppf(.025, dof) / dof, chi2.ppf(.975, dof) / dof
axes[0].plot(t, a, "k", lw=2.5, label="ANEES / n (all seeds)"); axes[0].axhline(1, color="g", ls="--", label="Ideal = 1")
axes[0].axhspan(lo, hi, color="g", alpha=.12, label="95% chi-square band (ensemble)")
axes[0].set(ylabel="NEES / n", yscale="log", title="Joint observer: NEES per seed and ANEES over time"); axes[0].grid(True, which="both")
if have_obj:
    seed_lines(axes[1], tb, obj_nees / 3, seeds, lw=.7, alpha=.45)
    n = np.nansum(obj_cnt, 0)
    ao = np.nansum(obj_nees * obj_cnt, 0) / np.maximum(n, 1) / 3
    ok = n > 0
    d3 = np.maximum(n * 3, 1)
    axes[1].plot(tb[ok], ao[ok], "k", lw=2.5, label="ANEES / n (all seeds)"); axes[1].axhline(1, color="g", ls="--", label="Ideal = 1")
    axes[1].fill_between(tb[ok], (chi2.ppf(.025, d3) / d3)[ok], (chi2.ppf(.975, d3) / d3)[ok], color="g", alpha=.12, label="95% band (optimistic)")
    axes[1].set(ylabel="NEES / n", yscale="log", title="Cube Kalman filter: NEES per seed and ANEES over time"); axes[1].grid(True, which="both")
for ax in axes: ax.legend(ncol=4, fontsize=7)
axes[-1].set_xlabel("Time (s)"); fig.tight_layout(); fig.savefig("anees_over_time.png", dpi=150)

# ======================= 3. Ground truth vs estimate, and error =======================
i0 = seeds.index(SHOW)
fig, axes = plt.subplots(3 if have_obj else 2, 1, figsize=(10, 10 if have_obj else 7), sharex=True)
for k in range(6):
    axes[0].plot(t, tru[i0, :, k], color=f"C{k}", label=f"q{k+1} truth"); axes[0].plot(t, est[i0, :, k], color=f"C{k}", ls="--")
axes[0].set(ylabel="Joint angle (rad)", title=f"Joint truth (solid) vs estimate (dashed), seed {SHOW} (colours = joints)"); axes[0].legend(ncol=6, fontsize=7); axes[0].grid(True)
seed_lines(axes[1], t, np.linalg.norm(e_q, axis=-1) * 1e3, seeds, lw=.7, alpha=.5)
axes[1].plot(t, np.linalg.norm(e_q, axis=-1).mean(0) * 1e3, "k", lw=2.5, label=f"mean over {N} seeds")
axes[1].set(ylabel="‖q_est − q_true‖ (mrad)", title="Joint estimation error over time"); axes[1].legend(ncol=4, fontsize=7); axes[1].grid(True)
if have_obj:
    seed_lines(axes[2], tb, obj_norm * 1e3, seeds, lw=.7, alpha=.5)
    axes[2].plot(tb, np.nanmean(obj_norm, 0) * 1e3, "k", lw=2.5, label=f"mean over {N} seeds")
    axes[2].set(ylabel="Cube position error (mm)", title="Cube Kalman filter error over time"); axes[2].legend(ncol=4, fontsize=7); axes[2].grid(True)
axes[-1].set_xlabel("Time (s)"); fig.tight_layout(); fig.savefig("truth_vs_estimate.png", dpi=150)

# ======================= 4. LQR over time =======================
track = np.linalg.norm(ref - tru, axis=-1)                       # true tracking error
effort = np.linalg.norm(u, axis=-1)
cost = QW * track ** 2 + VW * (dq ** 2).sum(-1) + RW * (u ** 2).sum(-1)
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
panels = [(track * 1e3, "‖ref − q_true‖ (mrad)", "LQR tracking error (against ground truth)", False),
          (effort, f"‖{U_SOURCE}‖", "Control effort", False),
          (cost, "Stage cost", f"LQR stage cost (Qpos={QW}, Qvel={VW}, R={RW})", True)]
for ax, (arr, yl, ttl, logy) in zip(axes, panels):
    seed_lines(ax, t, arr, seeds, lw=.7, alpha=.5)
    ax.plot(t, arr.mean(0), "k", lw=2.5, label=f"mean over {N} seeds")
    ax.set(ylabel=yl, title=ttl, yscale="log" if logy else "linear")
seed_lines(axes[3], t, smooth(sat, nsm), seeds, lw=.7, alpha=.5)
axes[3].plot(t, smooth(sat.mean(0), nsm), "k", lw=2.5, label=f"mean over {N} seeds")
axes[3].set(ylabel="Fraction saturated", title="Actuator saturation", xlabel="Time (s)")
for ax in axes: ax.grid(True, which="both"); ax.legend(ncol=4, fontsize=7)
fig.tight_layout(); fig.savefig("lqr_over_time.png", dpi=150)

print("Saved: rmse_over_time.png, anees_over_time.png, truth_vs_estimate.png, lqr_over_time.png")
if "agg" not in matplotlib.get_backend().lower(): plt.show()