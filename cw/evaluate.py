"""Baselines (plafond / planchers) et métriques de M0 : justesse, couverture, calibration.

Un prédicteur est une fonction  predict(world) -> {"edge": (N_MAX, N_MAX) P(a->b),
                                                   "q":    (Q, K_BINS) loi prédite,
                                                   "conf": (Q,) dans [0, 1] « je peux déterminer cette réponse »}.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .worlds import BIN_W, K_BINS, N_MAX, OOD, TRAIN, world_set


# ----------------------------------------------------------------------------- baselines
def _hist(x, alpha=0.5):
    h = np.bincount(np.asarray(x) // BIN_W, minlength=K_BINS).astype(np.float64) + alpha
    return h / h.sum()


def _ks(a, b):
    ca = np.cumsum(np.bincount(a, minlength=256)) / len(a)
    cb = np.cumsum(np.bincount(b, minlength=256)) / len(b)
    return float(np.abs(ca - cb).max())


def _edges_classic(w):
    """Heuristique classique : adjacence par corrélation observationnelle, sens par test KS entre do(t) et obs."""
    n = w.n
    X, obs = w.data[:, :n], w.rec_target == -1
    Xo = X[obs]
    with np.errstate(all="ignore"):
        C = np.nan_to_num(np.corrcoef(Xo.astype(float).T))
    adj_p = np.clip(2 * np.abs(C), 0.05, 0.95)                      # P(a et b adjacents)
    E = np.full((N_MAX, N_MAX), 0.5, np.float32)
    E[:n, :n] = 0.5 * adj_p                                          # sens inconnu : masse partagée
    for t in w.targets:
        Xt = X[w.rec_target == t]
        crit = 1.63 * np.sqrt((len(Xt) + len(Xo)) / (len(Xt) * len(Xo)))      # seuil KS ~ 1 %
        for b in range(n):
            if b == t:
                continue
            if _ks(Xt[:, b], Xo[:, b]) > crit:                       # b réagit à do(t) : descendant de t
                E[t, b], E[b, t] = 0.9 * adj_p[t, b], 0.02
            else:                                                    # pas de chemin t -> b
                E[t, b], E[b, t] = 0.02, 0.95 * adj_p[t, b]
    return E


def predict_truth(w):                                                # plafond : vérifie les métriques
    return dict(edge=w.adj.astype(np.float32), q=w.q_probs, conf=w.q_ident.astype(np.float32))


def predict_uniform(w):
    Q = len(w.queries)
    return dict(edge=np.full((N_MAX, N_MAX), 0.5, np.float32),
                q=np.full((Q, K_BINS), 1 / K_BINS, np.float32), conf=np.full(Q, 0.5, np.float32))


def predict_marginal(w):                                             # ignore l'intervention
    obs = w.rec_target == -1
    q = np.stack([_hist(w.data[obs, j]) for _, _, j in w.queries]).astype(np.float32)
    return dict(edge=np.full((N_MAX, N_MAX), 0.5, np.float32), q=q, conf=np.full(len(q), 0.5, np.float32))


def predict_kernel(w, bw=40):                                        # estimateur à noyau sur le régime do(i)
    X, obs = w.data, w.rec_target == -1
    q = np.zeros((len(w.queries), K_BINS), np.float32)
    conf = np.zeros(len(w.queries), np.float32)
    for k, (i, v, j) in enumerate(w.queries):
        if i in w.targets:
            sel = (w.rec_target == i) & (np.abs(X[:, i].astype(int) - v) <= bw)
            m = int(sel.sum())
            if m >= 3:
                q[k], conf[k] = _hist(X[sel, j]), min(1.0, m / 10)
                continue
        q[k], conf[k] = _hist(X[obs, j]), 0.3                        # i jamais intervenu : on suppose « pas d'effet »
    return dict(edge=_edges_classic(w), q=q, conf=conf)


# ----------------------------------------------------------------------------- métriques
def kl(t, p, eps=1e-9):
    """KL(vérité || prédiction) en nats, sur le dernier axe."""
    p = np.clip(np.asarray(p, np.float64), eps, None)
    p = p / p.sum(-1, keepdims=True)
    t = np.asarray(t, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(t > 0, t * (np.log(t) - np.log(p)), 0.0)
    return terms.sum(-1)


def ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if p.size == 0:
        return float("nan")
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def risk_coverage(conf, risk):
    """Risque moyen cumulé quand on répond d'abord aux plus confiants ; AURC = .mean()."""
    o = np.argsort(-np.asarray(conf), kind="stable")
    return np.cumsum(np.asarray(risk, float)[o]) / np.arange(1, len(o) + 1)


def _mean(a):
    a = np.asarray(a, float)
    return float(a.mean()) if a.size else float("nan")


def evaluate(predict, worlds, delta=0.1):
    """
    JUSTESSE   edge_acc_ident : sens correct (p > 0.5) sur les paires dont l'état est identifié par la classe
               do_kl_ident / do_kl_nonident : KL moyen sur requêtes identifiables / non identifiables
    COUVERTURE edge_cov_ident : part de ces paires correctes ET confiantes (p >= 0.75 ou <= 0.25)
               do_cov_ident   : part des requêtes identifiables avec KL <= delta et conf >= 0.5
    CALIBRATION edge_ece ; conf_gap = conf moyenne (identifiables) - (non identifiables), > 0 = « sait ce qu'il sait » ;
               do_aurc = aire sous la courbe risque (KL) - couverture, toutes requêtes
    DIAGNOSTIC edge_acc_ambig : sur les paires non identifiables ; nettement > 0.5 => exploite plus que
               l'équivalence de Markov (ex. asymétries du bruit additif).
    """
    P, Y, S, KL, ID, CF = [], [], [], [], [], []
    for w in worlds:
        out = predict(w)
        m = w.edge_ident != -2
        P.append(out["edge"][m]); Y.append(w.adj[m]); S.append(w.edge_ident[m])
        KL.append(kl(w.q_probs, out["q"])); ID.append(w.q_ident); CF.append(out["conf"])
    p, y, s = (np.concatenate(v) for v in (P, Y, S))
    k, ident, conf = (np.concatenate(v) for v in (KL, ID, CF))
    y = y.astype(float)
    det, amb = s >= 0, s == -1
    hit = (p > 0.5) == (y == 1)
    sure = hit & (np.abs(p - 0.5) >= 0.25)
    return {
        "edge_acc_ident": _mean(hit[det]), "edge_cov_ident": _mean(sure[det]), "edge_acc_ambig": _mean(hit[amb]),
        "edge_ece": ece(p, y),
        "do_kl_ident": _mean(k[ident]), "do_kl_nonident": _mean(k[~ident]),
        "do_cov_ident": _mean(((k <= delta) & (conf >= 0.5))[ident]),
        "conf_gap": _mean(conf[ident]) - _mean(conf[~ident]),
        "do_aurc": float(risk_coverage(conf, k).mean()),
    }


COLS = ["edge_acc_ident", "edge_cov_ident", "edge_acc_ambig", "edge_ece",
        "do_kl_ident", "do_kl_nonident", "do_cov_ident", "conf_gap", "do_aurc"]
PREDICTORS = dict(truth=predict_truth, kernel=predict_kernel, marginal=predict_marginal, uniform=predict_uniform)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for split, cfg in (("train", TRAIN), ("ood", OOD)):
        t0 = time.perf_counter()
        ws = world_set(a.seed, cfg, a.n)
        dt = time.perf_counter() - t0
        fid = float(np.mean(np.concatenate([w.q_ident for w in ws])))
        print(f"\n[{split}] {len(ws)} mondes, {len(ws) / dt:.0f} mondes/s | requêtes identifiables : {fid:.0%}")
        print(f"{'':10s}" + "".join(f"{c:>16s}" for c in COLS))
        for name, f in PREDICTORS.items():
            r = evaluate(f, ws)
            print(f"{name:10s}" + "".join(f"{r[c]:16.3f}" for c in COLS))


if __name__ == "__main__":
    main()