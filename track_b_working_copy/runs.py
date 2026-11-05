#!/usr/bin/env python3
"""Summarise batch results: success rate per condition and failure categories.

  python summarise_runs.py results/<batch>/results.csv --index results/<batch>/index.csv --md summary.md

Several CSVs can be given (for example the nominal batch and a varied batch); conditions
are grouped by the run_label column and shown in order of first appearance, nominal first.
Pass --index so crashed or timed-out runs count as failures instead of vanishing.
"""
import argparse
import csv
import math
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

# failure_causes is one string: "<Cube>:<reason>;<Cube>:<reason>;...". Some reasons contain a
# ';' themselves, so split only where the next piece starts with a cube name.
CAUSE_SPLIT = re.compile(r";\s*(?=(?:Red|Yellow|Blue) Cube:)")
CAUSE_PARSE = re.compile(r"^(Red|Yellow|Blue) Cube:(.*)$", re.S)
TRAY_NAME = re.compile(r"(Pink|Green|Purple) Tray")

# first matching rule wins, so the order matters
CATEGORIES = [
    ("sensed grasp disagrees with truth", lambda r: "sensed grasp disagrees" in r),
    ("tray localisation failed", lambda r: "localize" in r or "no pre-pick" in r),
    ("cube not re-acquired by camera", lambda r: "reacquire" in r),
    ("grasp not retained during lift", lambda r: "grasp was not retained" in r),
    ("cube lost in carry (carry sensor tripped)",
     lambda r: "[carry sensor tripped]" in r or "slipped" in r or "cube was lost" in r),
    ("cube released outside tray (truth)", lambda r: "released outside" in r),
    ("return/retreat move failed", lambda r: "return-home" in r or "retreat" in r),
    ("arm motion failed (PBVS timeout / IK)", lambda r: "[motion failed]" in r or "PBVS" in r),
]


def parse_causes(text):
    text = (text or "").strip()
    if not text:
        return []
    events = []
    for piece in CAUSE_SPLIT.split(text):
        match = CAUSE_PARSE.match(piece.strip())
        events.append((match.group(1), match.group(2).strip()) if match else ("?", piece.strip()))
    return events


def categorise(reason):
    for name, test in CATEGORIES:
        if test(reason):
            return name
    return "other"


def normalise(reason):
    return TRAY_NAME.sub("<tray>", reason)


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def to_bool(value):
    return str(value).strip().lower() == "true"


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes in n runs."""
    if n == 0:
        return math.nan, math.nan
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / denom, (centre + half) / denom


def pct(x):
    return "n/a" if x is None or math.isnan(x) else f"{100 * x:.0f}%"


def ratio(num, den):
    return num / den if den else math.nan


def mean_of(group, key):
    values = [to_float(r.get(key)) for r in group]
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else math.nan


def fmt(x, scale=1.0, digits=3, suffix=""):
    return "n/a" if x is None or math.isnan(x) else f"{x * scale:.{digits}f}{suffix}"


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def md_table(header, rows):
    rows = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(header)]
    def line(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    out = [line(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out += [line(r) for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="results.csv file(s)")
    ap.add_argument("--index", nargs="*", default=[], help="index.csv file(s) from run_batch.py")
    ap.add_argument("--md", default=None, help="also write the summary to this markdown file")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    index = load_rows(args.index) if args.index else []

    by_label = OrderedDict()
    for row in rows:
        by_label.setdefault(row.get("run_label") or "unlabelled", []).append(row)

    scheduled, bad = Counter(), Counter()
    order_seen = []
    for item in index:
        label = item["label"]
        if label not in order_seen:
            order_seen.append(label)
        scheduled[label] += 1
        if item["status"] not in ("ok", "cached"):
            bad[label] += 1
    for label in by_label:
        if label not in order_seen:
            order_seen.append(label)
        if not index:
            scheduled[label] = len(by_label[label])
    labels = sorted(order_seen, key=lambda l: (l != "nominal", order_seen.index(l)))

    # ---- headline table
    head = []
    for label in labels:
        group = by_label.get(label, [])
        n = scheduled[label]
        k = sum(to_bool(r.get("sequence_success")) for r in group)
        low, high = wilson(k, n)
        head.append([
            label, f"{n} ({bad[label]} crashed)" if bad[label] else str(n),
            f"{k}/{n} = {pct(ratio(k, n))}", f"{pct(low)}-{pct(high)}",
            pct(ratio(sum(to_float(r.get('completed_cubes')) for r in group),
                      sum(to_float(r.get('requested_cubes')) for r in group))),
            f"{ratio(sum(to_float(r.get('retries')) for r in group), len(group)):.2f}" if group else "n/a",
            pct(ratio(sum(to_float(r.get('pick_successes')) for r in group),
                      sum(to_float(r.get('attempts')) for r in group))),
            pct(ratio(sum(to_float(r.get('placement_successes')) for r in group),
                      sum(to_float(r.get('attempts')) for r in group))),
        ])
    sections = ["## Success by condition",
                md_table(["condition", "runs", "full sequence", "95% CI",
                          "cubes sorted", "retries/run", "pick rate", "placement rate"], head)]

    # ---- control design: tracking error against effort (only if the runs logged it)
    if any("torque_rms_nm" in r for r in rows):
        control = []
        for label in labels:
            group = by_label.get(label, [])
            if not group:
                continue
            weights = "/".join(f"{to_float(group[0].get(k)):g}" for k in
                               ("lqr_position_weight", "lqr_velocity_weight", "lqr_input_weight"))
            control.append([
                label, weights,
                fmt(mean_of(group, "joint_tracking_rmse_rad")),
                fmt(mean_of(group, "joint_tracking_rmse_away_rad")),
                fmt(mean_of(group, "joint_settled_rmse_rad")),
                fmt(mean_of(group, "pbvs_position_rmse_m"), 1000, 1),
                fmt(mean_of(group, "torque_rms_nm"), 1, 2),
                fmt(mean_of(group, "peak_torque_fraction"), 100, 0, "%"),
                fmt(mean_of(group, "torque_saturation_fraction"), 100, 2, "%"),
                fmt(mean_of(group, "ctrl_clip_fraction"), 100, 3, "%"),
            ])
        sections += ["", "## Control design (means over finished runs)",
                     "weights = position/velocity/input; errors in rad except TCP (mm)",
                     md_table(["condition", "weights", "tracking RMSE", "away from HOME",
                               "when reached", "TCP settled (mm)", "torque RMS (N m)",
                               "peak torque", "at torque limit", "cmd clipped"], control)]

    # ---- failure categories: number of runs with at least one event in the category
    runs_with = defaultdict(lambda: defaultdict(int))
    cube_events = defaultdict(Counter)
    reasons_all, reasons_by_label = Counter(), defaultdict(Counter)
    for label in labels:
        for row in by_label.get(label, []):
            cats = set()
            for cube, reason in parse_causes(row.get("failure_causes")):
                cat = categorise(reason)
                cats.add(cat)
                cube_events[label][cube] += 1
                reasons_all[(cat, normalise(reason))] += 1
                reasons_by_label[label][(cat, normalise(reason))] += 1
            for cat in cats:
                runs_with[cat][label] += 1
    categories = [c for c, _ in CATEGORIES if c in runs_with]
    if "other" in runs_with:
        categories.append("other")
    table = [[cat] + [str(runs_with[cat][l]) if runs_with[cat][l] else "-" for l in labels]
             for cat in categories]
    if any(bad.values()):
        table.append(["run crashed / timed out"] + [str(bad[l]) if bad[l] else "-" for l in labels])
    sections += ["", "## Failure categories (runs with at least one event)",
                 md_table(["category"] + labels, table) if table else "No failure causes logged."]

    # ---- which cube fails
    cube_table = [[cube] + [str(cube_events[l][cube]) if cube_events[l][cube] else "-" for l in labels]
                  for cube in ("Red", "Yellow", "Blue")]
    sections += ["", "## Failure events by cube", md_table(["cube"] + labels, cube_table)]

    # ---- most common raw reasons
    def reason_list(counter, limit=10):
        return [f"{count:4d} x [{cat}] {reason}" for (cat, reason), count in counter.most_common(limit)]
    if "nominal" in reasons_by_label:
        sections += ["", "## Most common reasons, nominal"] + reason_list(reasons_by_label["nominal"])
    sections += ["", "## Most common reasons, all conditions"] + reason_list(reasons_all)

    # ---- nominal failing seeds, to replay with the log
    if "nominal" in by_label:
        failing = sorted(int(r["seed"]) for r in by_label["nominal"]
                         if not to_bool(r.get("sequence_success")) and str(r.get("seed", "")).isdigit())
        sections += ["", f"Nominal seeds that failed: {failing if failing else 'none'}"]

    text = "\n".join(sections)
    print("\n" + text)
    if args.md:
        Path(args.md).write_text(text + "\n", encoding="utf-8")
        print(f"\nSummary written to {args.md}")


if __name__ == "__main__":
    sys.exit(main())