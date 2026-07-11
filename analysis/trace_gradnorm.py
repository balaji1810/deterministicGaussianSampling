"""Reconstruct the BFGS gradient-norm-vs-iteration trajectory.

The optimizer trajectory is fully deterministic given the seed and objective;
the stopping tolerance only decides *when* the loop halts, never *where* it
steps.  So the gradient-norm curve can be reconstructed exactly, without
instrumenting the shipped library, by re-running the same seed with an
increasing --maxIter cap and recording the final gradient norm (lastGtol,
already reported by lcd_experiment) at each cap.  This keeps the public
ApproximateOptions / release path untouched (the preferred fallback noted in
the task) while giving an exact per-iteration trace.

Writes <outdir>/trace_gradnorm.csv with columns: iter, gradNorm.

Usage:
    python analysis/trace_gradnorm.py --exe build/plots/lcd_experiment.exe \
        --outdir <results-dir> [--L 40] [--seed 42] [--maxIter 400]
"""

import argparse
import csv
import os
import re
import subprocess
import tempfile


def run_capped(exe, L, seed, cap, tmp):
    out = os.path.join(tmp, "s.csv")
    cmd = [exe, "--L", str(L), "--seed", str(seed), "--ftolRel", "0",
           "--maxIter", str(cap), "--out", out]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT"))
    iters = int(re.search(r"iterations=(\d+)", line).group(1))
    gnorm = float(re.search(r"lastGtol=([-0-9.eE+]+)", line).group(1))
    return iters, gnorm


def caps(max_iter):
    """Dense early, coarser late — enough to draw a smooth log-y curve."""
    pts = list(range(1, 51)) + list(range(52, 101, 2)) \
        + list(range(105, max_iter + 1, 5))
    return sorted(set(pts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.path.join("build", "plots",
                                                  "lcd_experiment.exe"))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--L", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--maxIter", type=int, default=400)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, "trace_gradnorm.csv")
    seen_iters = set()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for cap in caps(args.maxIter):
            iters, gnorm = run_capped(args.exe, args.L, args.seed, cap, tmp)
            # once the natural stop is reached, capping higher repeats it
            if iters in seen_iters:
                continue
            seen_iters.add(iters)
            rows.append((iters, gnorm))
            print(f"cap={cap:4d} -> iter={iters:4d} |grad|={gnorm:.3e}")
            if iters < cap:            # natural termination reached
                break
    rows.sort()
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iter", "gradNorm"])
        w.writerows(rows)
    print("wrote", path, f"({len(rows)} points)")


if __name__ == "__main__":
    main()
