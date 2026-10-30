"""
Genere un petit jeu TSP euclidien au format GOAL, pour un entrainement rapide.
A lancer depuis la racine du repo :  python gen_tsp_data.py
Cree data/tsp20_train.npz (noeuds ranges dans l'ordre du tour) et data/tsp20_val.npz.
Les tours de reference viennent d'un 2-opt avec redemarrages : proches de l'optimum,
pas garantis optimaux (un gap legerement negatif est donc possible).
"""
import time
import numpy as np

N_SITES = 20
N_TRAIN = 10000
N_VAL = 256
RESTARTS = 3


def tour_len(D, tour):
    return D[tour, np.roll(tour, -1)].sum()


def two_opt(D, tour):
    n = len(tour)
    i_idx, j_idx = np.triu_indices(n, k=2)
    keep = ~((i_idx == 0) & (j_idx == n - 1))
    i_idx, j_idx = i_idx[keep], j_idx[keep]
    while True:
        a, b = tour[i_idx], tour[i_idx + 1]
        c, d = tour[j_idx], tour[(j_idx + 1) % n]
        delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
        k = np.argmin(delta)
        if delta[k] > -1e-10:
            return tour
        i, j = i_idx[k], j_idx[k]
        tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]


def solve(coords, rng):
    D = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    best, best_len = None, np.inf
    for _ in range(RESTARTS):
        tour = two_opt(D, rng.permutation(len(coords)))
        length = tour_len(D, tour)
        if length < best_len:
            best, best_len = tour.copy(), length
    return best, best_len


def make(n_instances, ordered, rng):
    coords_out = np.zeros((n_instances, N_SITES + 1, 2))
    lens = np.zeros(n_instances)
    for k in range(n_instances):
        coords = rng.random((N_SITES, 2))
        tour, lens[k] = solve(coords, rng)
        order = tour if ordered else rng.permutation(N_SITES)   # entrainement : ordre du tour
        coords_out[k, :-1] = coords[order]
        coords_out[k, -1] = coords[order[0]]                    # dernier noeud = retour au depart
    return coords_out, lens


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = time.time()
    train_coords, train_lens = make(N_TRAIN, True, rng)
    np.savez("data/tsp20_train.npz", node_coords=train_coords, tour_lens=train_lens, is_training_dataset=True)
    val_coords, val_lens = make(N_VAL, False, rng)
    np.savez("data/tsp20_val.npz", node_coords=val_coords, tour_lens=val_lens, is_training_dataset=False)

    c = train_coords[0]
    assert abs(np.linalg.norm(c[1:] - c[:-1], axis=-1).sum() - train_lens[0]) < 1e-9
    print(f"OK en {time.time() - t:.0f}s : {N_TRAIN} instances d'entrainement, {N_VAL} de validation, "
          f"longueur moyenne {train_lens.mean():.3f}")