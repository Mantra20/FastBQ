"""
Gap d'optimalite par instance sur ATSP, en deux versions :
- affiche : meme formule que test.py (la fermeture du tour utilise la colonne de destination du fichier, voir B25)
- vrai    : la fermeture utilise le vrai retour d(derniere ville -> 0)
"""
import argparse
import numpy as np
import torch

from args import add_common_args
from learning.tsp.dataset import load_dataset
from learning.tsp.decoding import decode
from utils.data_manipulation import prepare_batch
from utils.exp import setup_test_environment
from utils.misc import set_seed

parser = argparse.ArgumentParser()
add_common_args(parser)
args = parser.parse_args()
set_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
net = setup_test_environment(args)
ckpt = torch.load(args.pretrained_model, map_location=device)
net.load_state_dict(ckpt["net"], strict=True)  # echoue si une cle manque ou est en trop
net.eval()

path = args.test_datasets[0]
loader = load_dataset(path, args.test_batch_size, args.test_datasets_size, what="test")
raw = np.load(path)
D_raw = raw["dist_matrices"][: args.test_datasets_size]  # matrices brutes (non normalisees)
ref_raw = raw["tour_lens"][: args.test_datasets_size]

gaps_shown, gaps_true, start = [], [], 0
with torch.no_grad():
    for data in loader:
        pdata = prepare_batch(data, "tsp", device, sample=False)
        pred, tours = decode("tsp", pdata, net, beam_size=args.beam_size, knns=args.knns)
        ref = pdata[-1]
        gaps_shown += (100 * (pred - ref) / ref).cpu().tolist()

        for b in range(tours.shape[0]):
            t = tours[b]  # 0..99 = villes, 100 = copie du depart
            D = D_raw[start + b]
            true_len = D[t[:-2], t[1:-1]].sum() + D[t[-2], 0]
            gaps_true.append(100 * (true_len - ref_raw[start + b]) / ref_raw[start + b])
        start += tours.shape[0]

for name, g in [("affiche", gaps_shown), ("vrai", gaps_true)]:
    g = np.array(g)
    print(f"gap {name:8s}: moyenne {g.mean():.3f} %  ecart-type {g.std(ddof=1):.3f}  "
          f"erreur std {g.std(ddof=1) / np.sqrt(len(g)):.3f}  min {g.min():.2f}  max {g.max():.2f}  "
          f"(n={len(g)}, negatifs={(g < 0).sum()})")
