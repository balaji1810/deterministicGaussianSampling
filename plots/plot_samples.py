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

# Covariance matrices for the figures
# Figure A: three correlated covariance structures with increasing L.
CORRELATED_SIGMAS = [
    {"sigma": [[2.0, 0.5], [0.5, 2.0]], "L": 10},
    {"sigma": [[1.5, -0.6], [-0.6, 1.5]], "L": 20},
    {"sigma": [[2.0, 1.0], [1.0, 2.0]], "L": 30},
]

# Figure B: standard normal (Sigma = I) approximated with more and more samples.
STANDARD_NORMAL_SIGMAS = [
    {"sigma": [[1.0, 0.0], [0.0, 1.0]], "L": 10},
    {"sigma": [[1.0, 0.0], [0.0, 1.0]], "L": 20},
    {"sigma": [[1.0, 0.0], [0.0, 1.0]], "L": 40},
]


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


def matrix_label(sigma):
    (a, b), (c, d) = sigma
    return "Σ = [[%g, %g], [%g, %g]]" % (a, b, c, d)


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
    ax.set_title("L = %d\n%s" % (n_samples, matrix_label(sigma)), fontsize=11)


def build_figure(generator, cases, suptitle, out_path):
    fig, axes = plt.subplots(1, len(cases), figsize=(4.2 * len(cases), 4.8))
    fig.patch.set_facecolor("white")
    for ax, case in zip(np.atleast_1d(axes), cases):
        draw_panel(ax, generator, case)
    fig.suptitle(suptitle, fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
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

    build_figure(
        generator,
        CORRELATED_SIGMAS,
        "Gaussian-to-Dirac (LCD) approximation for three covariance structures",
        os.path.join(args.out_dir, "samples_correlated.png"),
    )
    build_figure(
        generator,
        STANDARD_NORMAL_SIGMAS,
        "Standard normal approximated with increasing sample counts",
        os.path.join(args.out_dir, "samples_standard_normal.png"),
    )


if __name__ == "__main__":
    main()
