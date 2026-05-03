#!/usr/bin/env python3
"""
Generate presentation figures for cosine-similarity detection:
  1) Baseline master vector (aggregate CMS-style nonnegative vector).
  2) Benign epoch vs master (small shape change, high cosine similarity).
  3) Attack epoch vs master (strong distortion, lower cosine similarity).
  4) 2D / 3D coordinate views: same high-dimensional vectors projected for comparison.

Output PNGs default to: LG/figures/CosineSimilarity/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def make_master_vector(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Nonnegative, heavy-tailed sketch-like energy (lognormal-ish)."""
    v = rng.lognormal(mean=0.0, sigma=0.45, size=dim)
    v /= v.max()
    return v.astype(np.float64)


def make_benign_vector(master: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Small multiplicative noise around master (typical clean-epoch drift)."""
    noise = 1.0 + 0.04 * rng.standard_normal(master.shape)
    v = master * noise
    return np.clip(v, 0.0, None)


def make_attack_vector(master: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Pollution-style distortion: keep partial overlap with master but add a large
    component orthogonal in direction (low cosine), plus spikes.
    """
    u = master.astype(np.float64)
    noise = rng.lognormal(0.15, 0.5, master.shape)
    # Gram–Schmidt step: component of noise orthogonal to u
    nu = np.dot(noise, u) / (np.dot(u, u) + 1e-12)
    orth = noise - nu * u
    orth /= np.linalg.norm(orth) + 1e-12
    v = 0.55 * u + 1.15 * np.linalg.norm(u) * orth
    v = np.clip(v, 0.0, None)
    k = max(12, master.size // 8)
    idx = rng.choice(master.size, size=k, replace=False)
    v[idx] *= rng.uniform(3.0, 8.0, size=k)
    return v


def plot_master_strip(out_path: Path, master: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 2.2))
    ax.imshow(master[np.newaxis, :], aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_yticks([])
    ax.set_xlabel("Flattened CMS index (conceptual)")
    ax.set_title("Baseline master vector (aggregated over clean baseline epochs)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(
    out_path: Path,
    master: np.ndarray,
    other: np.ndarray,
    title: str,
    subtitle: str,
    show_slice: int = 80,
    baseline_profile: np.ndarray | None = None,
) -> None:
    sl = slice(0, min(show_slice, master.size))
    x = np.arange(sl.stop)
    m = master[sl].astype(np.float64)
    o = other[sl].astype(np.float64)
    if baseline_profile is not None:
        b = baseline_profile[sl].astype(np.float64)
        if b.shape == m.shape:
            top = float(max(m.max(), o.max(), b.max(), 1e-12))
            m, o, b = m / top, o / top, b / top
            fig, ax = plt.subplots(figsize=(9.0, 4.2))
            ax.plot(x, m, label="Master $\\mathbf{m}$", color="#1f77b4", lw=2.0, alpha=0.9)
            ax.plot(x, o, label="Epoch sketch $\\mathbf{v}$", color="#ff7f0e", lw=1.8, alpha=0.85)
            ax.plot(
                x,
                b,
                label="Clean baseline (typical epoch)",
                color="#2ca02c",
                lw=1.5,
                ls="--",
                alpha=0.9,
            )
        else:
            fig, ax = plt.subplots(figsize=(9.0, 4.2))
            ax.plot(x, master[sl], label="Master $\\mathbf{m}$", color="#1f77b4", lw=2.0, alpha=0.9)
            ax.plot(x, other[sl], label="Epoch sketch $\\mathbf{v}$", color="#ff7f0e", lw=1.8, alpha=0.85)
    else:
        fig, ax = plt.subplots(figsize=(9.0, 4.2))
        ax.plot(x, master[sl], label="Master $\\mathbf{m}$", color="#1f77b4", lw=2.0, alpha=0.9)
        ax.plot(x, other[sl], label="Epoch sketch $\\mathbf{v}$", color="#ff7f0e", lw=1.8, alpha=0.85)
    ax.set_xlabel("Index (first segment of flattened sketch)")
    ax.set_ylabel("Counter magnitude (normalised)")
    ax.set_title(title)
    ax.text(
        0.02,
        0.95,
        subtitle,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def project_pca_rows(
    master: np.ndarray, benign: np.ndarray, attack: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stack three sketch vectors as rows (3 x d) and project onto top right singular
    directions (same basis for all three — comparable coordinates).
    Returns (proj (3, k), explained_ratio per axis).
    """
    X = np.vstack([master, benign, attack]).astype(np.float64)
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(n_components, Vt.shape[0], X.shape[1])
    basis = Vt[:k].T  # (d, k)
    proj = X @ basis  # (3, k)
    total = float((s**2).sum()) + 1e-12
    ratio = (s[:k] ** 2) / total
    return proj, ratio


def project_master_plane_2d(
    master: np.ndarray, benign: np.ndarray, attack: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpretable 2D plane: x = projection on master direction; y = component of
    (attack - master) orthogonal to master, then benign/attack/master expressed there.
    """
    m = master.astype(np.float64)
    e1 = m / (np.linalg.norm(m) + 1e-12)
    ref = attack.astype(np.float64) - m
    ref_orth = ref - np.dot(ref, e1) * e1
    n2 = np.linalg.norm(ref_orth)
    if n2 < 1e-9:
        g = np.zeros_like(m)
        g[0] = 1.0
        ref_orth = g - np.dot(g, e1) * e1
        n2 = np.linalg.norm(ref_orth) + 1e-12
    e2 = ref_orth / n2

    def xy(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.float64)
        return np.array([np.dot(v, e1), np.dot(v, e2)])

    return xy(m), xy(benign), xy(attack)


def plot_2d_vectors(
    out_path: Path,
    master: np.ndarray,
    benign: np.ndarray,
    attack: np.ndarray,
    sim_b: float,
    sim_a: float,
) -> None:
    p_m, p_b, p_a = project_master_plane_2d(master, benign, attack)
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    pts = np.vstack([np.zeros(2), p_m, p_b, p_a])
    colors = ["#444444", "#1f77b4", "#2ca02c", "#d62728"]
    labels = ["Origin", "Master m", "Benign v", "Attack v'"]
    for i in range(1, 4):
        ax.annotate(
            "",
            xy=pts[i],
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=colors[i], lw=2.2, shrinkA=0, shrinkB=5),
        )
    ax.scatter(pts[1:, 0], pts[1:, 1], c=colors[1:], s=120, zorder=5, edgecolors="k", linewidths=0.6)
    for i, lab in enumerate(labels[1:], start=1):
        ax.annotate(lab, xy=(pts[i, 0], pts[i, 1]), xytext=(8, 8), textcoords="offset points", fontsize=10)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Along baseline master direction")
    ax.set_ylabel("Out-of-master-plane spread (attack-driven axis)")
    ax.set_title("2D view: benign stays near master, attack swings away")
    ax.text(
        0.02,
        0.98,
        f"$\\cos(\\mathbf{{m}},\\mathbf{{v}}_{{\\mathrm{{ben}}}}) \\approx {sim_b:.3f}$\n"
        f"$\\cos(\\mathbf{{m}},\\mathbf{{v}}'_{{\\mathrm{{atk}}}}) \\approx {sim_a:.3f}$",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_2d_pca(
    out_path: Path,
    master: np.ndarray,
    benign: np.ndarray,
    attack: np.ndarray,
    sim_b: float,
    sim_a: float,
) -> None:
    proj, ratio = project_pca_rows(master, benign, attack, 2)
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    names = ["Master m", "Benign v", "Attack v'"]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    for i, (name, c) in enumerate(zip(names, colors)):
        ax.annotate(
            "",
            xy=proj[i],
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=c, lw=2.2, shrinkA=0, shrinkB=5),
        )
    ax.scatter(proj[:, 0], proj[:, 1], c=colors, s=120, zorder=5, edgecolors="k", linewidths=0.6)
    for i, name in enumerate(names):
        ax.annotate(
            name,
            xy=(proj[i, 0], proj[i, 1]),
            xytext=(8, 6 + i * 4),
            textcoords="offset points",
            fontsize=10,
        )
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel(f"PC1 (energy {100 * ratio[0]:.1f}%)")
    ax.set_ylabel(f"PC2 (energy {100 * ratio[1]:.1f}%)")
    ax.set_title("2D PCA projection of three sketch vectors (same basis)")
    ax.text(
        0.02,
        0.98,
        f"$\\cos(\\mathbf{{m}},\\mathbf{{v}}) \\approx {sim_b:.3f}$ (benign), "
        f"${sim_a:.3f}$ (attack)",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4),
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_3d_pca(
    out_path: Path,
    master: np.ndarray,
    benign: np.ndarray,
    attack: np.ndarray,
    sim_b: float,
    sim_a: float,
) -> None:
    proj, ratio = project_pca_rows(master, benign, attack, 3)
    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    names = ["Master m", "Benign v", "Attack v'"]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    for i, c in enumerate(colors):
        ax.quiver(
            0,
            0,
            0,
            proj[i, 0],
            proj[i, 1],
            proj[i, 2],
            color=c,
            arrow_length_ratio=0.12,
            linewidth=2,
            length=1.0,
        )
    ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=colors, s=100, depthshade=True)
    for i, name in enumerate(names):
        ax.text(proj[i, 0], proj[i, 1], proj[i, 2], f"  {name}", fontsize=9)
    ax.set_xlabel(f"PC1 ({100 * ratio[0]:.0f}%)")
    ax.set_ylabel(f"PC2 ({100 * ratio[1]:.0f}%)")
    ax.set_zlabel(f"PC3 ({100 * ratio[2]:.0f}%)")
    ax.set_title("3D PCA projection: master vs benign vs attack")
    fig.text(
        0.5,
        0.02,
        f"Cosine(m, benign) ≈ {sim_b:.3f}   |   Cosine(m, attack) ≈ {sim_a:.3f}",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_combined_panel(
    out_path: Path,
    master: np.ndarray,
    benign: np.ndarray,
    attack: np.ndarray,
    sim_b: float,
    sim_a: float,
    show_slice: int = 80,
) -> None:
    sl = slice(0, min(show_slice, master.size))
    x = np.arange(sl.stop)
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.5), sharex=True)
    for ax, other, lab, sim in zip(
        axes,
        (benign, attack),
        ("Benign epoch", "Attack / polluted epoch"),
        (sim_b, sim_a),
    ):
        ax.plot(x, master[sl], label="Master $\\mathbf{m}$", color="#1f77b4", lw=2.0)
        ax.plot(x, other[sl], label=f"{lab} $\\mathbf{{v}}$", color="#ff7f0e", lw=1.8)
        ax.set_ylabel("Magnitude")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.25)
        ax.text(
            0.02,
            0.92,
            f"$\\cos(\\mathbf{{m}},\\mathbf{{v}}) \\approx {sim:.3f}$",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
        )
    axes[0].set_title("Cosine similarity vs baseline master (illustrative)")
    axes[1].set_xlabel("Index (first segment of flattened sketch)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for PNGs (default: LG/figures/CosineSimilarity)",
    )
    parser.add_argument("--dim", type=int, default=256, help="Synthetic flattened sketch length")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else repo / "figures" / "CosineSimilarity"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    master = make_master_vector(args.dim, rng)
    benign = make_benign_vector(master, rng)
    attack = make_attack_vector(master, rng)
    sim_b = cosine_similarity(master, benign)
    sim_a = cosine_similarity(master, attack)

    plot_master_strip(out_dir / "baseline_master_vector.png", master)
    plot_overlay(
        out_dir / "benign_epoch_vs_master.png",
        master,
        benign,
        title="Benign traffic: epoch sketch stays close to the master",
        subtitle=f"Cosine similarity $\\approx {sim_b:.3f}$ (high — typical clean drift)",
    )
    plot_overlay(
        out_dir / "attack_epoch_vs_master.png",
        master,
        attack,
        title="Pollution / attack: epoch sketch diverges from the master",
        subtitle=f"Cosine similarity $\\approx {sim_a:.3f}$ (lower — shape breaks baseline)",
        baseline_profile=benign,
    )
    plot_combined_panel(
        out_dir / "benign_vs_attack_vs_master.png",
        master,
        benign,
        attack,
        sim_b,
        sim_a,
    )
    plot_2d_vectors(out_dir / "vectors_2d_master_plane.png", master, benign, attack, sim_b, sim_a)
    plot_2d_pca(out_dir / "vectors_2d_pca.png", master, benign, attack, sim_b, sim_a)
    plot_3d_pca(out_dir / "vectors_3d_pca.png", master, benign, attack, sim_b, sim_a)

    print(f"Wrote figures to: {out_dir}")
    print(f"  cosine(benign)  = {sim_b:.4f}")
    print(f"  cosine(attack)  = {sim_a:.4f}")


if __name__ == "__main__":
    main()
