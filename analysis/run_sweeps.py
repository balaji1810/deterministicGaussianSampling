"""Experiment harness: sweeps L, bMax, seeds, tolerances over lcd_experiment.

Each run invokes the C++ driver (plots/lcd_experiment), loads the sample CSV,
computes the quality metrics from metrics.py, and appends one row to a
results CSV.  Usage:

    python analysis/run_sweeps.py --exe build/plots/lcd_experiment.exe \
        --outdir <results-dir> [--only exp1,exp2,...]

Experiments:
  baseline_L   quality vs L at default tolerances (reproduces the docs figure)
  strict_L     same but ftolRel=0 (optimizer runs to gradient convergence)
  bmax         quality vs bMax at L in {20,40}, default and strict tolerances
  multistart   20 seeds at L=40, strict; post-analysis picks the best
  sigma        scale test: isotropic sigma in {0.1,1,10} x bMax sweep, strict
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics  # noqa: E402

FIELDS = [
    "experiment", "L", "N", "bMax", "seed", "sigma", "ftolRel", "gtol",
    "init", "ok", "iterations", "elapsedMicro", "dist", "distEval",
    "evalBMax", "lastFtolAbs", "lastFtolRel", "lastGtol",
    "l2_star_disc_u", "mean_norm", "cov_err_fro", "cov_trace_ratio",
    "nn_cv_u", "min_dist_x", "samples_file",
]


def parse_result_line(text):
    for line in text.splitlines():
        if line.startswith("RESULT "):
            return dict(kv.split("=", 1) for kv in line[7:].split())
    raise RuntimeError(f"no RESULT line in output:\n{text}")


def run_one(exe, outdir, tag, L, sigma, bMax, seed, ftolRel=1e-10,
            gtol=1e-6, init_file=None, eval_bmax=0, experiment=""):
    os.makedirs(os.path.join(outdir, "samples"), exist_ok=True)
    sample_path = os.path.join(outdir, "samples", tag + ".csv")
    n_dim = len(sigma)
    cmd = [exe, "--N", str(n_dim), "--L", str(L),
           "--sigma", ",".join("%.17g" % s for s in sigma),
           "--bMax", str(bMax), "--seed", str(seed),
           "--ftolRel", "%.17g" % ftolRel, "--gtol", "%.17g" % gtol,
           "--out", sample_path]
    if init_file:
        cmd += ["--init", init_file]
    if eval_bmax:
        cmd += ["--evalBMax", str(eval_bmax)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  !! driver failed (rc={proc.returncode}): {tag}")
        row = {f: "" for f in FIELDS}
        row.update({"experiment": experiment, "L": L, "N": n_dim,
                    "bMax": bMax, "seed": seed,
                    "sigma": ";".join("%g" % s for s in sigma),
                    "ftolRel": ftolRel, "gtol": gtol,
                    "init": os.path.basename(init_file) if init_file
                    else "random",
                    "ok": 0, "samples_file": ""})
        return row
    res = parse_result_line(proc.stdout)

    x = np.loadtxt(sample_path, delimiter=",").reshape(L, n_dim)
    m = metrics.all_metrics(x, np.asarray(sigma))

    row = {
        "experiment": experiment, "L": L, "N": n_dim, "bMax": bMax,
        "seed": seed, "sigma": ";".join("%g" % s for s in sigma),
        "ftolRel": ftolRel, "gtol": gtol,
        "init": os.path.basename(init_file) if init_file else "random",
        "ok": res["ok"], "iterations": res["iterations"],
        "elapsedMicro": res["elapsedMicro"], "dist": res["dist"],
        "distEval": res["distEval"], "evalBMax": eval_bmax,
        "lastFtolAbs": res["lastFtolAbs"], "lastFtolRel": res["lastFtolRel"],
        "lastGtol": res["lastGtol"],
        "l2_star_disc_u": m["l2_star_disc_u"],
        "mean_norm": m["mean_norm"], "cov_err_fro": m["cov_err_fro"],
        "cov_trace_ratio": m["cov_trace_ratio"], "nn_cv_u": m["nn_cv_u"],
        "min_dist_x": m["min_dist_x"], "samples_file": sample_path,
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.path.join("build", "plots",
                                                  "lcd_experiment.exe"))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    os.makedirs(args.outdir, exist_ok=True)
    results_path = os.path.join(args.outdir, "results.csv")
    exists = os.path.exists(results_path)
    fh = open(results_path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if not exists:
        writer.writeheader()

    def emit(row):
        writer.writerow(row)
        fh.flush()
        print(f"[{row['experiment']}] L={row['L']} bMax={row['bMax']} "
              f"seed={row['seed']} init={row['init']} "
              f"iters={row['iterations']} dist={row['dist']}")

    L_grid = [10, 15, 20, 25, 30, 40, 50, 70, 100, 150, 200]
    seeds = [1, 2, 3, 42]

    if only is None or "baseline_L" in only:
        for L in L_grid:
            for seed in seeds:
                emit(run_one(args.exe, args.outdir,
                             f"base_L{L}_s{seed}", L, [1.0, 1.0], 100, seed,
                             eval_bmax=100, experiment="baseline_L"))

    if only is None or "strict_L" in only:
        for L in L_grid:
            for seed in seeds:
                emit(run_one(args.exe, args.outdir,
                             f"strict_L{L}_s{seed}", L, [1.0, 1.0], 100, seed,
                             ftolRel=0.0, eval_bmax=100,
                             experiment="strict_L"))

    if only is None or "bmax" in only:
        for L in (20, 40):
            for bmax in (1, 2, 3, 5, 10, 20, 50, 100, 200, 500):
                for seed in seeds:
                    for ftol, name in ((1e-10, "default"), (0.0, "strict")):
                        emit(run_one(
                            args.exe, args.outdir,
                            f"bmax_L{L}_b{bmax}_s{seed}_{name}",
                            L, [1.0, 1.0], bmax, seed, ftolRel=ftol,
                            eval_bmax=100,
                            experiment=f"bmax_{name}"))

    if only is None or "multistart" in only:
        for seed in range(1, 21):
            emit(run_one(args.exe, args.outdir,
                         f"ms_L40_s{seed}", 40, [1.0, 1.0], 100, seed,
                         ftolRel=0.0, eval_bmax=100,
                         experiment="multistart"))

    if only is None or "sigma" in only:
        for sig in (0.1, 1.0, 10.0):
            for bmax in (1, 2, 3, 10, 30, 100, 300, 1000):
                for seed in (1, 2, 42):
                    emit(run_one(
                        args.exe, args.outdir,
                        f"sig{sig:g}_b{bmax}_s{seed}", 20, [sig, sig],
                        bmax, seed, ftolRel=0.0,
                        eval_bmax=max(1, round(100 * sig)),
                        experiment="sigma"))

    fh.close()
    print("results written to", results_path)


if __name__ == "__main__":
    main()
