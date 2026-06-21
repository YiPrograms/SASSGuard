#!/usr/bin/env python3
"""Generate SVG figures for the GPU-Sentry paper from checked-in experiment numbers.

Run: ../.venv/bin/python make_figures.py
Outputs: Figures/throttle.svg, Figures/api_overhead.svg
"""
import json
import os
import glob
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams.update({
    "svg.fonttype": "path",   # render text as vector paths: identical on any build host
    "font.size": 9,
    "font.family": "serif",
    "axes.linewidth": 0.6,
})

GPU_SENTRY = "#1f4e79"   # blue
BASELINE = "#c0392b"     # red
NATIVE = "#7f8c8d"       # gray

# ---------------------------------------------------------------------------
# Figure 1: mining recall vs kernel-launch throttling rate
# ---------------------------------------------------------------------------
def throttle_fig():
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    # GPU-Sentry is invariant to throttling: recall 1.0 at every evaluated rate,
    # including the 5% extreme.
    gs_x = [5, 10, 50, 75, 100]
    gs_y = [1.0, 1.0, 1.0, 1.0, 1.0]
    # Behavioral baseline at 5/10/25/50/75/100.
    bl_x = [5, 10, 25, 50, 75, 100]
    bl_y = [1.0, 0.875, 0.0, 0.0, 0.0, 1.0]

    ax.plot(gs_x, gs_y, "-o", color=GPU_SENTRY, lw=1.6, ms=4.5,
            label="GPU-Sentry", zorder=3)
    ax.plot(bl_x, bl_y, "--s", color=BASELINE, lw=1.6, ms=4.5,
            label="Behavioral baseline", zorder=2)

    ax.set_xlabel("Kernel-launch rate (%)")
    ax.set_ylabel("Mining recall")
    ax.set_xlim(0, 105)
    ax.set_ylim(-0.06, 1.1)
    ax.set_xticks([5, 10, 25, 50, 75, 100])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, color="0.85", lw=0.5)
    ax.set_axisbelow(True)
    # Legend above the axes so it never floats over the data.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, fontsize=8, handlelength=1.9,
              columnspacing=1.4, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig("Figures/throttle.svg", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: median CUDA Driver-API latency (log scale), native vs cuPCAP vs ours
# ---------------------------------------------------------------------------
def api_overhead_fig():
    native = [2.575, 133.819, 25.302]
    cupcap = [6.545, 402.839, 41.292]
    ours = [2.706, 202.484, 35.081]

    x = range(3)
    w = 0.26
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.bar([i - w for i in x], native, w, color=NATIVE, label="Native")
    ax.bar([i for i in x], cupcap, w, color=BASELINE, label="cuPCAP (sync)")
    ax.bar([i + w for i in x], ours, w, color=GPU_SENTRY, label="GPU-Sentry (async)")

    ax.set_yscale("log")
    ax.set_ylabel("Median latency (µs, log scale)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["cuDeviceGetCount", "cuModuleLoad", "cuLaunchKernel"],
                       fontsize=6.5, rotation=12, fontfamily="monospace")
    ax.set_ylim(1, 800)
    ax.grid(True, axis="y", color="0.85", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=True, framealpha=1.0,
              edgecolor="0.7", fontsize=7.5)
    fig.tight_layout(pad=0.3)
    fig.savefig("Figures/api_overhead.svg", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: decision thresholds (real-world per-workload mean vs max mining prob)
# ---------------------------------------------------------------------------
def decision_threshold_fig():
    rep = os.path.join(REPO, "experiments/results/reports/modernbert_binary_gpu3_realworld/evaluate_test_report.json")
    preds = json.load(open(rep))["predictions"]
    # gminer_cortex is excluded everywhere in the paper: its pool served no work,
    # so the behavioral baseline could not capture it, and we keep both detectors
    # on the same 58-workload set (48 mining, 10 benign).
    preds = [p for p in preds if "cortex" not in (p.get("workload") or "").lower()]
    bm = [(p["mining_probability_mean"], p["mining_probability_max"]) for p in preds if p["label"] == "benign"]
    mm = [(p["mining_probability_mean"], p["mining_probability_max"]) for p in preds if p["label"] == "mining"]

    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    # flagged region: mean >= 0.30 AND max >= 0.50
    ax.axvspan(0.30, 1.02, ymin=(0.50 + 0.04) / 1.08, ymax=1.0, color="#eaf2f8", zorder=0)
    ax.scatter([m for m, _ in mm], [x for _, x in mm], s=20, color=GPU_SENTRY,
               alpha=0.8, label=f"Mining (n={len(mm)})", zorder=3)
    ax.scatter([m for m, _ in bm], [x for _, x in bm], s=26, color=BASELINE,
               marker="s", alpha=0.85, label=f"Benign (n={len(bm)})", zorder=3)
    ax.axvline(0.30, color="0.2", lw=1.1, ls="--")
    ax.axhline(0.50, color="0.2", lw=1.1, ls="--")
    ax.text(0.32, 0.04, r"mean $\geq 0.30$", fontsize=7, color="0.2", rotation=90, va="bottom")
    ax.text(0.04, 0.52, r"max $\geq 0.50$", fontsize=7, color="0.2")
    ax.text(0.985, 0.92, "flagged", fontsize=7.5, color="0.35", ha="right", style="italic")
    ax.annotate("Pyrin matrix-hash\n(resembles AI matmul)", xy=(0.10, 0.10), xytext=(0.45, 0.22),
                fontsize=6.5, color="0.25", ha="left",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=0.7))
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlabel("Mean mining probability (per workload)")
    ax.set_ylabel("Max mining probability")
    ax.grid(True, color="0.9", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=True, framealpha=1.0, edgecolor="0.7", fontsize=7.5)
    fig.tight_layout(pad=0.3)
    fig.savefig("Figures/decision_threshold.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    throttle_fig()
    decision_threshold_fig()
    # api_overhead is presented as a table (tab:api_overhead), not a figure.
    print("wrote throttle, decision_threshold SVGs")
