'''phase de test uniquement avec numpy pour tout
comprendre (objectif mini bq)

Le TSP : on a 6 morceaux (I, c, a, b, q, D)
    I : la liste des villes qu'il reste à visiter (le dépôt 0 n'en fait jamais partie car on ne peut pas y revenir avant la fin)
    c : la matrice des distances, c[i][j] = coût pour aller de i à j (ça nne change jamais)
    a : coût de départ, a[i] = coût pour aller de ma position actuelle jusqu'à la ville i
    b : coût de fin,    b[i] = coût pour rentrer au dépôt depuis la ville i (ça ne change jamais)
    q : coût si je rentre au dépôt tout de suite, sans visiter d'autre ville
    D : les kilomètres déjà roulés (le compteur)

Quand je visite la ville o:
    I  ->  I sans o
    a  ->  la ligne o de c  (je suis maintenant en o)
    q  ->  b[o]             (rentrer depuis o)
    D  ->  D + a[o]         (je viens de rouler jusqu'à o)
    c et b ne changent pas
    récompense = q - a[o] - b[o]  (c'est de combien la tournée s'allonge, toujours négatif)

A la fin (I est vide normalement sinon y'a un soucis) : D + q = longueur totale de la tournée
'''

import numpy as np

def etat_initial(c):
    # on part du depot donc :ville 0
    I = list(range(1, len(c)))
    a = c[0]      # ligne 0
    b = c[:, 0]   # colonne 0
    q = 0.0
    D = 0.0
    return I, c, a, b, q, D


def transition(etat, o):
    I, c, a, b, q, D = etat
    recompense = q - a[o] - b[o]

    nouveau_I = [i for i in I if i != o]
    nouveau_a = c[o]   # on est en o maintenant
    nouveau_q = b[o]
    nouveau_D = D + a[o]

    return (nouveau_I, c, nouveau_a, b, nouveau_q, nouveau_D), recompense


def afficher(etat):
    I, c, a, b, q, D = etat
    print("I =", I)
    print("a =", {i: float(a[i]) for i in I})
    print("b =", {i: float(b[i]) for i in I})
    print("q =", float(q), "  D =", float(D), "  D + q =", float(D + q))


# exemple de test : 4 villes (0 = dépôt, 1, 2, 3)
c = np.array([[0, 2, 3, 3],
              [2, 0, 2, 4],
              [3, 2, 0, 2],
              [3, 4, 2, 0]], dtype=float)

for ordre in ([1, 2, 3], [2, 1, 3]):
    print("\nordre", ordre,)
    etat = etat_initial(c)
    somme = 0.0
    afficher(etat)

    for o in ordre:
        etat, r = transition(etat, o)
        somme += r
        print(f"\nville {o}, recompense = {float(r)}")
        afficher(etat)

    I, _, a, b, q, D = etat
    print(f"\nfin : D + q = {float(D + q)}, somme recompenses = {float(somme)}")  # doit donner 9 puis 12 au vu de l'exemple
