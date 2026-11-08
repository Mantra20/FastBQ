"""
Inspect a dataset .npz file (TSP/ATSP/...): keys, shapes, and details of one instance.

Usage:
  python data_tools/inspect_npz.py data/atsp100_test.npz
  python data_tools/inspect_npz.py data/atsp100_test.npz --idx 3 --n 8 --plot
"""

import argparse

import numpy as np


def describe_keys(data):
    print("== Keys ==")
    for k in data.files:
        a = data[k]
        extra = f" value={a.item()}" if a.ndim == 0 else ""
        print(f"{k:22s} shape={str(a.shape):18s} dtype={a.dtype}{extra}")


def describe_matrix(m, n):
    print(f"\n== dist_matrices[idx] : {m.shape} ==")
    print(f"min={m.min():.4f} max={m.max():.4f} mean={m.mean():.4f}")
    print(f"diagonal all zero : {np.allclose(np.diag(m), 0)}")
    print(f"symmetric (TSP)   : {np.allclose(m, m.T)}  "
          f"(mean |D-D^T| / mean D = {np.abs(m - m.T).mean() / m.mean():.3f})")
    print(f"row 0 == row last : {np.allclose(m[0], m[-1])}  (last node = copy of origin)")
    print(f"col 0 == col last : {np.allclose(m[:, 0], m[:, -1])}")
    print(f"top-left {n}x{n} corner:")
    with np.printoptions(precision=3, suppress=True, linewidth=160):
        print(m[:n, :n])
    print("\nWhat the model gets as edge_features: stack([D, D^T]) -> shape",
          (*m.shape, 2), "then D / D.max()")


def describe_coords(c, n):
    print(f"\n== node_coords[idx] : {c.shape} ==")
    print(f"min={c.min(axis=0)} max={c.max(axis=0)}")
    with np.printoptions(precision=3, suppress=True):
        print(f"first {n} nodes:\n{c[:n]}")


def plot(m, coords, title):
    import matplotlib.pyplot as plt

    ncols = 2 if coords is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)
    im = axes[0, 0].imshow(m, cmap="viridis")
    axes[0, 0].set_title("distance matrix")
    axes[0, 0].set_xlabel("j")
    axes[0, 0].set_ylabel("i")
    fig.colorbar(im, ax=axes[0, 0])
    if coords is not None:
        axes[0, 1].scatter(coords[:, 0], coords[:, 1], s=12)
        axes[0, 1].scatter(*coords[0], c="red", s=60, label="origin")
        axes[0, 1].set_title("node coordinates")
        axes[0, 1].legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--idx", type=int, default=0, help="instance index")
    parser.add_argument("--n", type=int, default=6, help="size of the printed matrix corner")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    data = np.load(args.path)
    describe_keys(data)

    num_instances = len(data[data.files[0]]) if data[data.files[0]].ndim > 0 else 0
    print(f"\ninstances: {num_instances}, inspecting idx={args.idx}")

    coords = data["node_coords"][args.idx] if "node_coords" in data.files else None
    if "dist_matrices" in data.files:
        m = data["dist_matrices"][args.idx]
    elif coords is not None:
        from scipy.spatial.distance import pdist, squareform
        m = squareform(pdist(coords))
        print("no dist_matrices in file: computed from node_coords (euclidean)")
    else:
        m = None

    if coords is not None:
        describe_coords(coords, args.n)
    if m is not None:
        describe_matrix(m, args.n)

    for k in ("tour_lens", "objective_values"):
        if k in data.files:
            print(f"\n{k}[idx] = {data[k][args.idx]:.4f}  (mean over dataset = {data[k].mean():.4f})")

    if args.plot and m is not None:
        plot(m, coords, f"{args.path} - instance {args.idx}")


if __name__ == "__main__":
    main()
