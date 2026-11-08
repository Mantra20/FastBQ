"""Corrige la colonne de destination (copie du depart) d'un fichier ATSP : d(i -> copie) := d(i -> depart) (B25)."""
import os
import sys

import numpy as np

src, dst = sys.argv[1], sys.argv[2]
d = np.load(src)
D = d["dist_matrices"].copy()
D[:, :, -1] = D[:, :, 0]
os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
np.savez(dst, dist_matrices=D, tour_lens=d["tour_lens"], is_training_dataset=d["is_training_dataset"])
print("ecrit", dst, "| colonne destination == colonne depart :", np.allclose(D[:, :, -1], D[:, :, 0]))