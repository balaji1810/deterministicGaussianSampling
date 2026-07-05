import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# plotting constants
AXIS_LIMIT = 4.0
GRID_RES = 400
SEED = "42"

# Figure A: a correlation x sample-count grid. Rows vary the correlation rho of
# a unit-variance covariance Sigma = [[1, rho], [rho, 1]]; columns vary the
# deterministic sample count L.
CORRELATED_RHOS = [0.3, 0.6, 0.9]  # rows: increasing correlation
CORRELATED_LS = [10, 20, 40, 80]   # columns: increasing sample count

# Figure B: standard normal (Sigma = I)
IDENTITY_SIGMA = [[1.0, 0.0], [0.0, 1.0]]
STANDARD_NORMAL_L_GRID = [[10, 20, 30], [40, 50, 60]]


def correlation_sigma(rho):
    return [[1.0, rho], [rho, 1.0]]


def resolve_generator(path):
    for candidate in (path, path + ".exe"):
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    sys.exit(f"error: generator not found at '{path}' (or '{path}.exe'). ")


def run_generator(generator, sigma_diag, n_samples):
    """Invoke the C++ sampler and return its samples as an (L, N) array."""
    n_dim = len(sigma_diag)
    env = dict(os.environ, GSL_RNG_SEED=SEED)
    with tempfile.TemporaryDirectory() as tmp:
        out_csv = os.path.join(tmp, "samples.csv")
        cmd = [generator, str(n_dim), str(n_samples), out_csv]
        cmd += ["%.17g" % s for s in sigma_diag]
        subprocess.run(cmd, env=env, check=True)
        samples = np.loadtxt(out_csv, delimiter=",")
    return samples.reshape(n_samples, n_dim)


def sample_points(generator, sigma, n_samples):
    sigma = np.asarray(sigma, dtype=float)
    lam, q = np.linalg.eigh(sigma)
    lam = np.clip(lam, 0.0, None)  # guard tiny negative round-off eigenvalues
    z = run_generator(generator, np.sqrt(lam), n_samples)
    return z @ q.T


def gaussian_density(sigma):
    sigma = np.asarray(sigma, dtype=float)
    axis = np.linspace(-AXIS_LIMIT, AXIS_LIMIT, GRID_RES)
    xx, yy = np.meshgrid(axis, axis)
    pts = np.stack([xx, yy], axis=-1)
    inv = np.linalg.inv(sigma)
    quad = np.einsum("...i,ij,...j->...", pts, inv, pts)
    norm = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(sigma)))
    return norm * np.exp(-0.5 * quad)


def draw_panel(ax, generator, case):
    sigma = case["sigma"]
    n_samples = case["L"]

    density = gaussian_density(sigma)
    ax.imshow(
        density,
        origin="lower",
        extent=[-AXIS_LIMIT, AXIS_LIMIT, -AXIS_LIMIT, AXIS_LIMIT],
        vmin=0.0,
        aspect="equal",
    )

    points = sample_points(generator, sigma, n_samples)
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=30,
        c="#e8000b",
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )

    ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_xticks(range(-4, 5, 2))
    ax.set_yticks(range(-4, 5, 2))


def build_figure(generator, grid, suptitle, out_path,
                 col_headers=None, row_labels=None, panel_titles=False):

    n_rows = len(grid)
    n_cols = len(grid[0])
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 4.2 * n_rows + 0.6),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")
    for i, row in enumerate(grid):
        for j, case in enumerate(row):
            ax = axes[i][j]
            draw_panel(ax, generator, case)
            if panel_titles:
                ax.set_title("L = %d" % case["L"], fontsize=12)
            if col_headers is not None and i == 0:
                ax.set_title(col_headers[j], fontsize=13)
            if row_labels is not None and j == 0:
                ax.set_ylabel(row_labels[i], fontsize=13, labelpad=8)
    fig.suptitle(suptitle, fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generator",
        default=os.path.join("build", "plots", "generate_samples"),
        help="path to the compiled generate_samples binary",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("doxygen", "images"),
        help="directory to write the PNG figures into",
    )
    args = parser.parse_args()

    generator = resolve_generator(args.generator)
    os.makedirs(args.out_dir, exist_ok=True)

    # Figure A: correlation (rows) x sample count (columns).
    correlated_grid = [
        [{"sigma": correlation_sigma(rho), "L": L} for L in CORRELATED_LS]
        for rho in CORRELATED_RHOS
    ]
    build_figure(
        generator,
        correlated_grid,
        "Gaussian-to-Dirac (LCD) approximation: correlation (rows) "
        "x sample count (columns)",
        os.path.join(args.out_dir, "samples_correlated.png"),
        col_headers=["L = %d" % L for L in CORRELATED_LS],
        row_labels=["ρ = %.1f" % rho for rho in CORRELATED_RHOS],
    )

    # Figure B: standard normal (Sigma = I) with increasing L.
    standard_grid = [
        [{"sigma": IDENTITY_SIGMA, "L": L} for L in row]
        for row in STANDARD_NORMAL_L_GRID
    ]
    build_figure(
        generator,
        standard_grid,
        "Standard normal (Σ = I) approximated with increasing sample counts",
        os.path.join(args.out_dir, "samples_standard_normal.png"),
        panel_titles=True,
    )


if __name__ == "__main__":
    main()
