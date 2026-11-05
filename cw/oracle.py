"""Oracle d'identifiabilité (n <= 5) : classe d'équivalence interventionnelle, par énumération.

Convention : adj[a, b] = 1  <=>  a -> b.  Interventions dures, une variable à la fois.
Régimes = {observationnel} U {do(t) : t in cibles}.  Deux DAG sont équivalents ssi, pour chaque régime,
leurs graphes mutilés (arêtes entrantes des cibles retirées) ont même squelette et mêmes v-structures
(Hauser & Bühlmann, 2012).  Oracle volontairement *conservateur* : il ignore ce que des hypothèses
fonctionnelles (bruit additif...) pourraient identifier en plus.
"""
from __future__ import annotations

import functools
import itertools

import numpy as np

MAX_N = 5


def _w(n):
    return (1 << np.arange(n * n, dtype=np.int64)).reshape(n, n)


@functools.lru_cache(maxsize=None)
def all_dags(n):
    """Tous les DAG étiquetés à n noeuds : (N, n, n) uint8 ; N = 3, 25, 543, 29281 pour n = 2..5."""
    assert 1 <= n <= MAX_N, f"énumération exhaustive limitée à n <= {MAX_N}"
    pos = [(a, b) for a in range(n) for b in range(a + 1, n)]
    idx = np.arange(1 << len(pos))
    U = np.zeros((len(idx), n, n), np.uint8)
    for k, (a, b) in enumerate(pos):
        U[:, a, b] = (idx >> k) & 1
    codes = []
    for perm in itertools.permutations(range(n)):
        p = np.array(perm)
        A = np.zeros_like(U)
        A[:, p[:, None], p[None, :]] = U
        codes.append((A.astype(np.int64) * _w(n)).sum(axis=(1, 2)))
    codes = np.unique(np.concatenate(codes))
    out = ((codes[:, None] >> np.arange(n * n, dtype=np.int64)) & 1).reshape(-1, n, n).astype(np.uint8)
    out.setflags(write=False)
    return out


@functools.lru_cache(maxsize=None)
def _codes(n):
    return (all_dags(n).astype(np.int64) * _w(n)).sum(axis=(1, 2))      # trié croissant


def dag_index(adj):
    codes = _codes(adj.shape[0])
    c = int((adj.astype(np.int64) * _w(adj.shape[0])).sum())
    k = int(np.searchsorted(codes, c))
    assert k < len(codes) and codes[k] == c, "adj n'est pas un DAG"
    return k


def _signature(A, targets):
    """Signature (squelette + v-structures) du graphe mutilé, pour chaque DAG de A."""
    M = A.copy()
    for t in targets:
        M[:, :, t] = 0                                    # do(t) : on retire les arêtes entrantes
    n = A.shape[1]
    Mb = M.astype(bool)
    S = Mb | Mb.transpose(0, 2, 1)
    sig, bit = np.zeros(len(A), np.int64), 0
    for a in range(n):
        for b in range(a + 1, n):
            sig |= S[:, a, b].astype(np.int64) << bit
            bit += 1
    for c in range(n):
        for a in range(n):
            for b in range(a + 1, n):
                if c in (a, b):
                    continue
                v = Mb[:, a, c] & Mb[:, b, c] & ~S[:, a, b]
                sig |= v.astype(np.int64) << bit
                bit += 1
    return sig


@functools.lru_cache(maxsize=None)
def _sigs(n):
    A = all_dags(n)
    return np.stack([_signature(A, ())] + [_signature(A, (t,)) for t in range(n)])   # ligne 0 = obs, t+1 = do(t)


def equivalence_class(adj, targets=()):
    """Indices (dans all_dags(n)) des DAG indiscernables de adj avec obs + do(t) pour t in targets."""
    sg, d = _sigs(adj.shape[0]), dag_index(adj)
    ok = sg[0] == sg[0][d]
    for t in targets:
        ok &= sg[t + 1] == sg[t + 1][d]
    return np.flatnonzero(ok)


def analyze(adj, targets=()):
    """Ce que les données (obs + do(cibles)) permettent d'établir sur adj.

    edge_ident[a, b] : 1 / 0 si l'arête a->b est présente / absente dans toute la classe, -1 si ambiguë, -2 diagonale.
    pa_agree[i]      : tous les membres partagent les parents de i => effet de do(X_i) identifiable (ajustement).
    noeffect[i, j]   : j n'est descendant de i dans aucun membre   => loi de X_j inchangée par do(X_i).
    """
    n = adj.shape[0]
    mem = all_dags(n)[equivalence_class(adj, targets)].astype(np.int16)              # (m, n, n)
    frac = mem.mean(0)
    edge = np.where(frac == 1.0, 1, np.where(frac == 0.0, 0, -1)).astype(np.int8)
    np.fill_diagonal(edge, -2)
    R = (mem + np.eye(n, dtype=np.int16)) > 0
    for _ in range(3):                                                                # fermeture transitive (n <= 5)
        R = np.matmul(R.astype(np.int16), R.astype(np.int16)) > 0
    return dict(edge_ident=edge, pa_agree=(mem == mem[:1]).all(axis=(0, 1)),
                noeffect=~R.any(axis=0), n_class=len(mem))