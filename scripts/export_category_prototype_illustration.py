#!/usr/bin/env python3
"""Export clean 3D illustrations for semantic category enrichment."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "latex" / "rapport_final" / "figures"
COVERAGE_OUT = FIG_DIR / "category_prototype_manifold_coverage.pdf"
INTERLACED_OUT = FIG_DIR / "category_interlaced_manifolds_enrichment.pdf"


BLUE = "#2563eb"
TEAL = "#0f766e"
ORANGE = "#f59e0b"
RED = "#dc2626"
SLATE = "#334155"
GRAY = "#94a3b8"
PURPLE = "#7c3aed"


def configure_matplotlib():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_3d(ax, title):
    ax.set_title(title, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("dim. 1", labelpad=-8, color="#64748b")
    ax.set_ylabel("dim. 2", labelpad=-8, color="#64748b")
    ax.set_zlabel("dim. 3", labelpad=-8, color="#64748b")
    ax.grid(False)
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.25, 1, 0.72))
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    ax.xaxis.line.set_color("#cbd5e1")
    ax.yaxis.line.set_color("#cbd5e1")
    ax.zaxis.line.set_color("#cbd5e1")


def semantic_curve(t, phase=0.0, offset=(0.0, 0.0, 0.0)):
    return np.vstack(
        [
            0.92 * t + offset[0],
            0.55 * np.sin(1.1 * t + phase) + offset[1],
            0.40 * np.cos(0.85 * t + phase / 2) + offset[2],
        ]
    ).T


def ribbon_from_curve(points, width=0.22, twist=0.0):
    n = len(points)
    u = np.linspace(-1.0, 1.0, 18)
    offsets = []
    for i in range(n):
        angle = twist + 1.5 * i / max(n - 1, 1)
        offsets.append(np.array([0.05 * np.sin(angle), width * np.cos(angle), width * np.sin(angle)]))
    offsets = np.array(offsets)
    surface = points[:, None, :] + u[None, :, None] * offsets[:, None, :]
    return surface[:, :, 0], surface[:, :, 1], surface[:, :, 2]


def plot_manifold(ax, points, color, alpha=0.16, width=0.22, twist=0.0, line_label=None):
    x, y, z = ribbon_from_curve(points, width=width, twist=twist)
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, antialiased=True, shade=False)
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=2.8, label=line_label)


def line_between(ax, a, b, color, linewidth=2.4, linestyle="-", label=None, alpha=1.0):
    ax.plot(
        [a[0], b[0]],
        [a[1], b[1]],
        [a[2], b[2]],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
        alpha=alpha,
    )


def nearest_by_cosine(product, prototypes):
    product_norm = product / np.linalg.norm(product)
    protos_norm = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)
    return int(np.argmax(protos_norm @ product_norm))


def export_coverage_figure():
    t = np.linspace(-2.2, 2.2, 140)
    manifold = semantic_curve(t)

    prototype_t = np.array([-1.85, -1.05, -0.25, 0.55, 1.15, 1.82])
    prototypes = semantic_curve(prototype_t)
    prototypes += np.array(
        [
            [0.00, 0.02, 0.00],
            [-0.02, -0.05, 0.04],
            [0.03, 0.04, -0.03],
            [-0.04, 0.02, 0.02],
            [0.01, -0.04, -0.02],
            [0.02, 0.03, 0.03],
        ]
    )
    product = semantic_curve(np.array([0.95]))[0] + np.array([0.10, -0.03, 0.08])
    canonical = prototypes[0]
    nearest = prototypes[nearest_by_cosine(product, prototypes)]

    fig = plt.figure(figsize=(13.6, 5.6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    style_3d(ax1, "Sans enrichissement : un prototype unique")
    plot_manifold(ax1, manifold, TEAL, alpha=0.12, width=0.25, line_label="manifold semantique")
    ax1.scatter(*canonical, s=130, color=BLUE, edgecolor="white", linewidth=1.2, label="nom canonique")
    ax1.scatter(*product, s=130, color=RED, edgecolor="white", linewidth=1.2, label="produit")
    line_between(ax1, product, canonical, GRAY, linestyle="--", label="distance au seul point")
    ax1.text2D(
        0.03,
        0.02,
        "Un seul prototype echantillonne mal\nla region semantique de la categorie.",
        transform=ax1.transAxes,
        color=SLATE,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.96),
    )
    ax1.legend(loc="upper left", frameon=True, framealpha=0.95)

    style_3d(ax2, "Avec enrichissement : couverture du manifold")
    plot_manifold(ax2, manifold, TEAL, alpha=0.14, width=0.25, line_label="manifold semantique")
    enriched_prototypes = prototypes[1:]
    ax2.scatter(
        enriched_prototypes[:, 0],
        enriched_prototypes[:, 1],
        enriched_prototypes[:, 2],
        s=86,
        color=TEAL,
        edgecolor="white",
        linewidth=1.0,
        label="prototypes ajoutes",
    )
    ax2.scatter(
        *canonical,
        s=130,
        color=BLUE,
        edgecolor="white",
        linewidth=1.2,
        label="nom canonique",
    )
    ax2.scatter(*nearest, s=165, color=ORANGE, edgecolor="white", linewidth=1.2, label="prototype retenu")
    ax2.scatter(*product, s=130, color=RED, edgecolor="white", linewidth=1.2, label="produit")
    for proto in prototypes:
        line_between(ax2, product, proto, "#cbd5e1", linewidth=0.9, linestyle=":", alpha=0.75)
    line_between(ax2, product, nearest, ORANGE, linewidth=3.0, label="max des similarites")
    ax2.text2D(
        0.03,
        0.02,
        "Les prototypes enrichis recouvrent mieux\nle manifold : le max choisit le point le plus proche.",
        transform=ax2.transAxes,
        color=SLATE,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.96),
    )
    ax2.legend(loc="upper left", frameon=True, framealpha=0.95)

    fig.suptitle(
        "Recouvrement du manifold semantique par les prototypes de categorie",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.01,
        "Illustration en dimension d=3 ; dans le systeme reel, les embeddings sont de dimension beaucoup plus elevee.",
        ha="center",
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(COVERAGE_OUT, bbox_inches="tight")
    plt.close(fig)


def helix_curve(t, phase=0.0, radius=1.0, z_scale=0.22):
    return np.vstack(
        [
            radius * np.cos(t + phase),
            radius * np.sin(t + phase),
            z_scale * (t - t.mean()),
        ]
    ).T


def export_interlaced_figure():
    t = np.linspace(-2.75, 2.75, 190)
    manifold_a = helix_curve(t, phase=0.0, radius=1.0)
    manifold_b = helix_curve(t, phase=0.82, radius=1.0) + np.array([0.10, -0.08, 0.02])

    idx_sparse = np.array([30, 88, 145])
    idx_dense = np.array([18, 42, 67, 92, 118, 143, 168])
    proto_a = manifold_a[idx_dense] + np.array([0.00, 0.00, 0.03])
    proto_b = manifold_b[idx_dense] + np.array([0.00, 0.00, -0.03])
    sparse_a = manifold_a[idx_sparse]
    sparse_b = manifold_b[idx_sparse]
    product = manifold_a[115] + np.array([0.02, -0.05, 0.08])

    fig = plt.figure(figsize=(13.6, 5.8))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    style_3d(ax1, "Deux categories proches avec peu de points")
    plot_manifold(ax1, manifold_a, TEAL, alpha=0.13, width=0.12, twist=0.3, line_label="categorie A")
    plot_manifold(ax1, manifold_b, PURPLE, alpha=0.13, width=0.12, twist=0.9, line_label="categorie B")
    ax1.scatter(sparse_a[:, 0], sparse_a[:, 1], sparse_a[:, 2], s=82, color=TEAL, edgecolor="white", linewidth=1.0)
    ax1.scatter(sparse_b[:, 0], sparse_b[:, 1], sparse_b[:, 2], s=82, color=PURPLE, edgecolor="white", linewidth=1.0)
    ax1.scatter(*product, s=140, color=RED, edgecolor="white", linewidth=1.2, label="produit ambigu")
    for proto in np.vstack([sparse_a, sparse_b]):
        line_between(ax1, product, proto, "#cbd5e1", linewidth=0.8, linestyle=":", alpha=0.55)
    ax1.text2D(
        0.03,
        0.02,
        "Avec peu de prototypes, deux categories\nentrelacees peuvent etre confondues.",
        transform=ax1.transAxes,
        color=SLATE,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.96),
    )
    ax1.legend(loc="upper left", frameon=True, framealpha=0.95)

    style_3d(ax2, "Prototypes enrichis : meilleure separation locale")
    plot_manifold(ax2, manifold_a, TEAL, alpha=0.13, width=0.12, twist=0.3, line_label="categorie A")
    plot_manifold(ax2, manifold_b, PURPLE, alpha=0.13, width=0.12, twist=0.9, line_label="categorie B")
    ax2.scatter(proto_a[:, 0], proto_a[:, 1], proto_a[:, 2], s=78, color=TEAL, edgecolor="white", linewidth=1.0)
    ax2.scatter(proto_b[:, 0], proto_b[:, 1], proto_b[:, 2], s=78, color=PURPLE, edgecolor="white", linewidth=1.0)
    ax2.scatter(*product, s=140, color=RED, edgecolor="white", linewidth=1.2, label="produit ambigu")
    nearest_a = proto_a[nearest_by_cosine(product, proto_a)]
    nearest_b = proto_b[nearest_by_cosine(product, proto_b)]
    line_between(ax2, product, nearest_a, TEAL, linewidth=3.0, label="meilleur proto A")
    line_between(ax2, product, nearest_b, PURPLE, linewidth=2.2, linestyle="--", label="meilleur proto B")
    ax2.text2D(
        0.03,
        0.02,
        "Ajouter des points augmente les chances\nde trouver le bon voisin semantique local.",
        transform=ax2.transAxes,
        color=SLATE,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.96),
    )
    ax2.legend(loc="upper left", frameon=True, framealpha=0.95)

    fig.suptitle(
        "Manifolds semantiques entrelaces : pourquoi multiplier les prototypes aide",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.01,
        "Illustration qualitative : les vrais manifolds sont inconnus, mais les prototypes enrichis les echantillonnent plus finement.",
        ha="center",
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(INTERLACED_OUT, bbox_inches="tight")
    plt.close(fig)


def main():
    configure_matplotlib()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    export_coverage_figure()
    export_interlaced_figure()
    print(f"Saved {COVERAGE_OUT}")
    print(f"Saved {INTERLACED_OUT}")


if __name__ == "__main__":
    main()
