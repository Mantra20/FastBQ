"""
Test de la formulation TSP (I, c, a, b) dans GOAL.
A lancer depuis la racine du repo :  python test_tsp_bq.py
Doit finir par "TOUT OK".
"""
import sys
import numpy as np
import torch

sys.path.insert(0, ".")
from model.goal import GOAL
from utils import data_manipulation as dm
from utils.misc import compute_tour_lens
from learning.tsp.decoding import decode
from learning.tsp.dataset import DataSet, collate_func_with_sample

torch.manual_seed(0)
np.random.seed(0)

# vraies instances ATSP du repo : 100 sites ey retour au depart = 101 noeuds
data = np.load("data/atsp100_test.npz")
dist = data["dist_matrices"][:8]
ds = DataSet(dist_matrices=dist, objective_values=data["tour_lens"][:8])
bs = 8

# petit reseau pour aller vite, meme structure que GOAL
net = GOAL(dim_node_idx=1, dim_emb=32, num_layers=2, dim_ff=64, activation_ff="relu",
           node_feature_low_dim=8, edge_feature_low_dim=4, activation_edge_adapter="relu", num_heads=4)

# 1. un pas d'entrainement sur un sous-probleme echantillonne
batch = collate_func_with_sample([ds[i] for i in range(bs)])
batch_of_data = dm.prepare_batch(batch, "tsp", "cpu", sample=True)
nf, ef, pdata = dm.prepare_data(batch_of_data, "tsp")
n = batch_of_data[0].shape[1] - 2                     # sites de I = tous sauf origine et destination
assert nf.shape == (bs, n, 2), f"features (a, b) attendues, recu {tuple(nf.shape)}"
assert ef.shape == (bs, n, n, 2), f"aretes c entre sites de I attendues, recu {tuple(ef.shape)}"
scores = net(nf, ef, pdata)
assert scores.shape == (bs, n), f"un score par site de I attendu, recu {tuple(scores.shape)}"
gt = dm.create_ground_truth(bs, pdata, "cpu")
assert (gt == 0).all(), "la cible doit etre l'index 0"
loss = torch.nn.CrossEntropyLoss()(scores, gt)
loss.backward()
assert torch.isfinite(loss)
print(f"entrainement OK : |I| = {n}, scores {tuple(scores.shape)}, loss = {loss.item():.3f}")

# 2. decodage : glouton, glouton avec kNN, beam search
d = torch.stack([ds[i].dist_matrices for i in range(bs)])
for kwargs in [dict(), dict(knns=20), dict(beam_size=4)]:
    with torch.no_grad():
        vals, tours = decode("tsp", [d], net, **kwargs)
    for t in tours:
        assert sorted(t[:-1].tolist()) == list(range(100)) and t[-1] == 100, "tour invalide"
    direct = compute_tour_lens(torch.tensor(tours), d[..., 0])
    assert torch.allclose(vals, direct), "cout du tour incoherent"
    print(f"decodage {kwargs or 'glouton'} OK, cout moyen {vals.mean().item():.3f}")

# 3. le raisonnement BQ : cout direct du tour == D + q apres toutes les transitions
c = torch.tensor(dist[0], dtype=torch.float64)
N = c.shape[0] - 1                       # noeud 0 = depot, noeud N = retour au depot
a, b = c[0].clone(), c[:, N].clone()     # a_i = depot -> i, b_i = i -> depot
q, D = c[0, N], torch.tensor(0., dtype=torch.float64)
tour = list(np.random.permutation(range(1, N)))
direct = a[tour[0]] + sum(c[tour[k], tour[k + 1]] for k in range(len(tour) - 1)) + b[tour[-1]]
for o in tour:                           # theta' = (I\{o}, c, c_o., b, b_o, D + a_o)
    D, q, a = D + a[o], b[o], c[o].clone()
print(f"cout direct {direct.item():.6f} | D + q apres transitions {(D + q).item():.6f}")
assert abs(direct - (D + q)) < 1e-9

print("TOUT OK")
