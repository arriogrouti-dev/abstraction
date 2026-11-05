"""Générateur de mondes causaux synthétiques (M0).

Un monde = SCM à n <= 5 variables-octets, bruit additif, interventions dures.  Un enfant lit ses parents
décodés z = (octet - 128) / 40, applique un mécanisme, ajoute du bruit, requantifie en octet.
do(X_i = v) fixe l'octet exact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .oracle import analyze

N_MAX = 5
K_BINS = 8
BIN_W = 256 // K_BINS                  # l'octet x tombe dans le bin x // 32
CENTER, SCALE = 128.0, 40.0


@dataclass(frozen=True)
class WorldConfig:
    graphs: tuple = ("er", "tree")
    mechs: tuple = ("linear", "mlp", "threshold")
    noises: tuple = ("gaussian", "uniform")
    n_range: tuple = (3, 5)
    n_rec: int = 160                   # enregistrements/monde : moitié observationnels, moitié interventionnels
    n_queries: int = 8
    mc: int = 2000                     # tirages Monte-Carlo par requête (vérité terrain)
    shuffle: bool = True


TRAIN = WorldConfig()
OOD = WorldConfig(graphs=("chain", "collider", "dense"), mechs=("xor", "periodic", "table"), noises=("laplace",))


@dataclass
class World:
    """CONTRAT D'INTERFACE GELÉ.  Entrées du modèle : n, data, rec_target, queries.
    Le reste est vérité terrain (perte externe + évaluation)."""
    n: int
    data: np.ndarray        # (n_rec, N_MAX) uint8, colonnes >= n à 0
    rec_target: np.ndarray  # (n_rec,) int8 : variable forcée dans l'enregistrement, -1 = observationnel
    queries: np.ndarray     # (Q, 3) int16 : (i, v, j) = « loi de X_j sous do(X_i = v) ? »
    targets: tuple          # variables intervenues dans les données
    adj: np.ndarray         # (N_MAX, N_MAX) uint8 : adj[a, b] = 1 <=> a -> b
    edge_ident: np.ndarray  # (N_MAX, N_MAX) int8 : 1/0 présente/absente dans toute la classe, -1 ambiguë, -2 sans objet
    q_probs: np.ndarray     # (Q, K_BINS) float32 : loi de (X_j // 32) sous l'intervention
    q_ident: np.ndarray     # (Q,) bool : requête identifiable (conservateur, cf. oracle.analyze)
    meta: dict


@dataclass
class SCM:
    n: int
    adj: np.ndarray
    order: list
    parents: list
    mechs: list
    mech_names: list
    sigma: np.ndarray
    noise: str


def topo_order(adj):
    indeg = adj.sum(0).astype(int)
    ready = [i for i in range(len(adj)) if indeg[i] == 0]
    out = []
    while ready:
        i = ready.pop()
        out.append(i)
        for j in np.flatnonzero(adj[i]):
            indeg[j] -= 1
            if indeg[j] == 0:
                ready.append(int(j))
    return out


def sample_dag(rng, n, kind):
    order = rng.permutation(n)
    A = np.zeros((n, n), np.uint8)

    def er(lo, hi):
        p = rng.uniform(lo, hi)
        for a in range(n):
            for b in range(a + 1, n):
                if rng.random() < p:
                    A[order[a], order[b]] = 1

    if kind == "er":
        er(0.25, 0.6)
    elif kind == "dense":
        er(0.75, 0.95)
    elif kind == "tree":
        for b in range(1, n):
            A[order[rng.integers(0, b)], order[b]] = 1
    elif kind == "chain":
        for b in range(1, n):
            A[order[b - 1], order[b]] = 1
    elif kind == "collider":                       # un puits à n-1 parents + quelques arêtes entre parents
        for a in range(n - 1):
            A[order[a], order[-1]] = 1
            for b in range(a + 1, n - 1):
                if rng.random() < 0.2:
                    A[order[a], order[b]] = 1
    else:
        raise ValueError(kind)
    return A


def make_mech(rng, name, k):
    """Mécanisme Z (m, k) -> (m,) ; paramètres tirés une fois pour toutes à la création."""
    w = rng.normal(size=k)
    w /= np.linalg.norm(w) + 1e-9
    if name == "linear":
        return lambda Z: Z @ w
    if name == "mlp":
        H = 6
        W1, b1 = rng.normal(size=(H, k)), rng.normal(0, 0.5, size=H)
        w2 = rng.normal(size=H) * 1.5 / np.sqrt(H)
        return lambda Z: np.tanh(Z @ W1.T + b1) @ w2
    if name == "threshold":
        t = rng.normal(0, 0.3)
        return lambda Z: 1.2 * np.sign(Z @ w - t)
    if name == "xor":                              # parité des signes (OOD)
        c = rng.normal(0, 0.3, size=k)
        return lambda Z: 1.2 * np.prod(np.sign(Z - c), axis=1)
    if name == "periodic":                         # (OOD)
        om, ph = rng.uniform(2.0, 4.0), rng.uniform(0, 2 * np.pi)
        return lambda Z: 1.2 * np.sin(om * (Z @ w) + ph)
    if name == "table":                            # fonction aléatoire sur parents discrétisés en 4 niveaux (OOD)
        tab, base = rng.normal(size=4 ** k), 4 ** np.arange(k)
        return lambda Z: tab[(np.digitize(Z, [-0.7, 0.0, 0.7]) * base).sum(axis=1)]
    raise ValueError(name)


def draw_noise(rng, kind, size):                   # variance unité
    if kind == "gaussian":
        return rng.normal(size=size)
    if kind == "uniform":
        return rng.uniform(-np.sqrt(3), np.sqrt(3), size=size)
    if kind == "laplace":
        return rng.laplace(scale=1 / np.sqrt(2), size=size)
    raise ValueError(kind)


def sample_scm(rng, cfg):
    assert cfg.n_range[1] <= N_MAX
    n = int(rng.integers(cfg.n_range[0], cfg.n_range[1] + 1))
    graph = str(rng.choice(cfg.graphs))
    adj = sample_dag(rng, n, graph)
    while adj.sum() == 0:
        adj = sample_dag(rng, n, graph)
    parents = [np.flatnonzero(adj[:, j]) for j in range(n)]
    names, mechs = [], []
    for j in range(n):
        if len(parents[j]) == 0:
            names.append("root")
            mechs.append(None)
        else:
            nm = str(rng.choice(cfg.mechs))
            names.append(nm)
            mechs.append(make_mech(rng, nm, len(parents[j])))
    sigma = rng.uniform(0.15, 0.6, size=n)
    noise = str(rng.choice(cfg.noises))
    return SCM(n, adj, topo_order(adj), parents, mechs, names, sigma, noise), graph


def simulate(scm, rng, rec_target, rec_val):
    """M enregistrements ; rec_target[m] = variable forcée (-1 = aucune) à l'octet rec_val[m]."""
    M = len(rec_target)
    X = np.zeros((M, scm.n), np.int16)
    for j in scm.order:
        eps = draw_noise(rng, scm.noise, M)
        pa = scm.parents[j]
        if len(pa) == 0:
            lat = eps
        else:
            Z = (X[:, pa] - CENTER) / SCALE
            s = scm.sigma[j]
            lat = np.sqrt(1 - s * s) * scm.mechs[j](Z) + s * eps
        x = np.clip(np.rint(CENTER + SCALE * lat), 0, 255).astype(np.int16)
        X[:, j] = np.where(rec_target == j, rec_val, x)
    return X


def make_queries(rng, n, q):
    i = rng.integers(0, n, size=q)
    j = (i + rng.integers(1, n, size=q)) % n               # j != i
    v = rng.integers(0, 256, size=q)
    return np.stack([i, v, j], 1).astype(np.int16)


def query_truth(scm, rng, queries, mc):
    """Loi (en K_BINS bins) de X_j sous do(X_i = v), par Monte-Carlo."""
    Q = len(queries)
    X = simulate(scm, rng, np.repeat(queries[:, 0], mc), np.repeat(queries[:, 1], mc)).reshape(Q, mc, scm.n)
    out = np.zeros((Q, K_BINS), np.float32)
    for q in range(Q):
        out[q] = np.bincount(X[q, :, queries[q, 2]] // BIN_W, minlength=K_BINS) / mc
    return out


def sample_world(rng, cfg=TRAIN):
    scm, graph = sample_scm(rng, cfg)
    n = scm.n
    k = int(rng.integers(1, n + 1))                                    # nombre de variables intervenues
    targets = tuple(sorted(int(t) for t in rng.choice(n, size=k, replace=False)))
    n_obs = cfg.n_rec // 2
    sizes = [n_obs] + [len(s) for s in np.array_split(np.arange(cfg.n_rec - n_obs), k)]
    rec_t = np.concatenate([np.full(s, -1 if r == 0 else targets[r - 1]) for r, s in enumerate(sizes)])
    X = simulate(scm, rng, rec_t, rng.integers(0, 256, size=cfg.n_rec))
    if cfg.shuffle:
        p = rng.permutation(cfg.n_rec)
        X, rec_t = X[p], rec_t[p]
    data = np.zeros((cfg.n_rec, N_MAX), np.uint8)
    data[:, :n] = X.astype(np.uint8)
    queries = make_queries(rng, n, cfg.n_queries)
    q_probs = query_truth(scm, rng, queries, cfg.mc)
    info = analyze(scm.adj, targets)
    qi, qj = queries[:, 0], queries[:, 2]
    q_ident = info["pa_agree"][qi] | info["noeffect"][qi, qj]
    edge = np.full((N_MAX, N_MAX), -2, np.int8)
    edge[:n, :n] = info["edge_ident"]
    adj = np.zeros((N_MAX, N_MAX), np.uint8)
    adj[:n, :n] = scm.adj
    return World(n=n, data=data, rec_target=rec_t.astype(np.int8), queries=queries, targets=targets, adj=adj,
                 edge_ident=edge, q_probs=q_probs, q_ident=q_ident,
                 meta=dict(graph=graph, mechs=scm.mech_names, noise=scm.noise, n_class=info["n_class"]))


def world_set(seed, cfg, n):
    """n mondes reproductibles : le monde i ne dépend que de (seed, i)."""
    return [sample_world(np.random.default_rng(s), cfg) for s in np.random.SeedSequence(seed).spawn(n)]


def stack(worlds):
    """Liste de World -> dict de tableaux (B, ...) prêts pour un modèle."""
    keys = ("data", "rec_target", "queries", "adj", "edge_ident", "q_probs", "q_ident")
    out = {k: np.stack([getattr(w, k) for w in worlds]) for k in keys}
    out["n"] = np.array([w.n for w in worlds])
    return out


def batches(seed, cfg, batch_size):
    """Flux infini de lots (dict stack) ; chaque monde a sa graine propre."""
    ss = np.random.SeedSequence(seed)
    while True:
        yield stack([sample_world(np.random.default_rng(s), cfg) for s in ss.spawn(batch_size)])