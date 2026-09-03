#!/usr/bin/env python3
"""Figures of chapter 5 (Implementation and Scaling Analysis), as PDFs.

Every number here is measured by the scripts next to this one:
  bench_frames.py    -> scaling_n, scaling_degree      (2026-08-27)
  canon_scaling.py   -> tab:bench-n canonicalization block
  canon_rgs.py       -> canon_rgs                      (2026-08-30, 5 reps)
  the drift check of subsec:bench-canon -> delta_degree (2026-08-30, 40 reps)

Run from anywhere:
    .env/bin/python3 benchmarks/plots_ch5.py [--out DIR]
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Default: the thesis figure directory next to this checkout
# (<repo>/Eulerian-frame/paper/figures/graphStates), overridable with --out.
OUT = Path(__file__).resolve().parents[2] / "paper" / "figures" / "graphStates"
VCOL = "#37474F"   # thesis Vcol, structure
ACC  = "#B8860B"   # thesis Acc, highlight
THIRD = "#7B1E3A"  # third and fourth series (RGS families)
FOURTH = "#2E6F6B"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "figure.figsize": (7.2, 4.4),
})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


# ---------------------------------------------------------------- scaling_n
def scaling_n():
    n      = [50, 80, 130, 200, 320, 500, 800, 1300, 2000, 3200]
    graphsim = [14.28, 15.77, 16.57, 17.10, 19.01, 20.02, 21.45, 22.25, 23.66, 25.32]
    euler    = [12.43, 13.60, 14.14, 14.37, 16.23, 16.66, 18.21, 19.06, 21.93, 25.42]
    fig, ax = plt.subplots()
    ax.semilogx(n, graphsim, "s--", color=VCOL, ms=5, lw=1.4, label="GraphSim")
    ax.semilogx(n, euler, "o-", color=ACC, ms=5, lw=1.4, label="Eulerian frame")
    ax.set_xlabel("number of qubits $n$")
    ax.set_ylabel(r"time per operation ($\mu$s)")
    ax.set_ylim(0, 30)
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation(base=10.0))
    ax.xaxis.set_minor_locator(
        matplotlib.ticker.LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1), numticks=100))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(loc="upper left")
    save(fig, "scaling_n")


# ----------------------------------------------------------- scaling_degree
def scaling_degree():
    d = [2, 4, 8, 16, 32, 64, 128, 256]
    graphsim = [10.88, 16.20, 32.04, 84.39, 275.37, 1112.15, 4646.23, 15935.09]
    euler    = [8.23, 14.03, 30.82, 84.21, 278.22, 1119.35, 4655.16, 15686.36]
    fig, ax = plt.subplots()
    guide = np.array([4, 330.0])
    ax.loglog(guide, 0.2841 * guide**2, ":", color=VCOL, alpha=0.55, lw=1.2)
    ax.text(150, 0.2841 * 150**2 * 2.1, r"slope $d^{2}$", color=VCOL,
            alpha=0.8, fontsize=9, rotation=38)
    ax.loglog(d, graphsim, "s--", color=VCOL, ms=5, lw=1.4, label="GraphSim")
    ax.loglog(d, euler, "o-", color=ACC, ms=5, lw=1.4, label="Eulerian frame")
    ax.set_xlabel("average degree $d$")
    ax.set_ylabel(r"time per operation ($\mu$s)")
    ax.set_xticks(d)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(1.75, 440)
    ax.legend(loc="lower right")
    save(fig, "scaling_degree")


# ------------------------------------------------------------- delta_degree
def delta_degree():
    n = 800
    meas = [(2, 0.004, 0.0069), (4, 0.021, 0.0139), (8, 0.075, 0.0288),
            (16, 0.348, 0.0705), (32, 1.099, 0.1946), (64, 4.286, 0.5310),
            (128, 14.768, 1.5563), (200, 25.499, 2.5148), (300, 27.507, 3.2408),
            (380, 8.803, 4.1439), (400, -0.148, 4.0141), (420, -11.416, 3.7002),
            (500, -79.467, 3.6935), (600, -225.847, 4.9873),
            (700, -458.488, 6.9575), (780, -722.940, 5.4691)]
    d = np.linspace(0, 800, 2000)
    fig, ax = plt.subplots()
    ax.set_yscale("symlog", linthresh=1e-2, linscale=0.5)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.axvline(400, color="0.6", ls="--", lw=0.9)
    ax.plot(d, d**2 / n * (1 - 2 * d / n), lw=1.8, color=VCOL,
            label=r"$\Delta d=\frac{d^2}{n}\left(1-\frac{2d}{n}\right)$")
    ax.errorbar([m[0] for m in meas], [m[1] for m in meas],
                yerr=[m[2] for m in meas], fmt="o", ms=4.5, color=ACC,
                capsize=3, lw=1, ls="none", label="measured, one re-framing")
    ax.set_xlim(0, 800)
    ax.set_ylim(-1e3, 1e2)
    ax.set_xlabel("average degree $d$")
    ax.set_ylabel(r"degree drift $\Delta d$")
    ax.text(408, 30, "$d=n/2$", fontsize=9, color="0.35")
    ax.legend(loc="lower left")
    save(fig, "delta_degree")


# ----------------------------------------------------------------- canon_rgs
RGS = {                       # m: (n, ms, mean degree after, largest degree after)
    5:  [(80, 1.52, 8.2, 24.6), (320, 5.09, 8.8, 32.0),
         (1280, 23.78, 9.4, 46.0), (5120, 135.74, 9.6, 52.6)],
    10: [(160, 4.77, 7.4, 35.0), (640, 17.68, 7.3, 39.6),
         (2560, 90.28, 8.5, 66.6), (10240, 804.26, 8.3, 65.8)],
    15: [(240, 8.65, 6.2, 43.2), (960, 43.68, 7.8, 61.6),
         (3840, 226.84, 7.2, 68.2), (15360, 2525.63, 7.2, 81.0)],
    20: [(320, 17.30, 5.7, 55.4), (1280, 78.43, 7.3, 69.6),
         (5120, 498.45, 7.0, 80.2), (20480, 5283.60, 6.8, 85.2)],
}
STYLE = {5: (VCOL, "s--"), 10: (ACC, "o-"), 15: (THIRD, "^-."),
         20: (FOURTH, "D-")}


def canon_rgs():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for m, rows in RGS.items():
        c, st = STYLE[m]
        ns = [r[0] for r in rows]
        ax.loglog(ns, [r[1] for r in rows], st, color=c, ms=5, lw=1.4,
                  label=f"$m={m}$")
        bx.semilogx(ns, [r[2] for r in rows], st.replace("--", "-"), color=c,
                    ms=5, lw=1.4, label=f"$m={m}$")
    guide = np.array([80.0, 20480.0])
    ax.loglog(guide, 0.0125 * guide, ":", color="0.35", lw=1.2)
    ax.text(4200, 0.0125 * 4200 * 0.40, "slope $n$", color="0.35", fontsize=9,
            rotation=30)
    ax.set_xlabel("number of qubits $n$")
    ax.set_ylabel("canonicalization time (ms)")
    ax.legend(loc="upper left")

    bx.set_xlabel("number of qubits $n$")
    bx.set_ylabel(r"mean degree after canonicalization")
    bx.set_ylim(0, 16)
    bx.legend(loc="upper right", ncol=2, columnspacing=0.8,
              handlelength=1.6, framealpha=0.95)
    save(fig, "canon_rgs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT,
                    help="directory for the PDFs (default: the thesis figure "
                         "directory next to this checkout)")
    OUT = ap.parse_args().out
    OUT.mkdir(parents=True, exist_ok=True)
    scaling_n()
    scaling_degree()
    delta_degree()
    canon_rgs()
