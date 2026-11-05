import numpy as np
import pytest

from cw.evaluate import (ece, evaluate, kl, predict_kernel, predict_marginal, predict_truth,
                         predict_uniform, risk_coverage)
from cw.oracle import all_dags, analyze, equivalence_class
from cw.worlds import BIN_W, K_BINS, OOD, SCM, TRAIN, simulate, topo_order, world_set


def adj_from(edges, n):
    A = np.zeros((n, n), np.uint8)
    for a, b in edges:
        A[a, b] = 1
    return A


@pytest.mark.parametrize("n, expected", [(2, 3), (3, 25), (4, 543), (5, 29281)])
def test_dag_enumeration(n, expected):
    A = all_dags(n).astype(np.int64)
    assert len(A) == expected
    P = A.copy()
    for _ in range(n - 1):
        P = P @ A
    assert not P.any()                                   # A^n = 0  <=>  acyclique


def test_chain_equivalence_classes():
    chain = adj_from([(0, 1), (1, 2)], 3)
    assert len(equivalence_class(chain, ())) == 3        # chaîne, chaîne inversée, fourche
    assert len(equivalence_class(chain, (0,))) == 1
    assert len(equivalence_class(chain, (1,))) == 1
    assert len(equivalence_class(chain, (2,))) == 2      # do(2) oriente 1->2 mais pas l'arête 0-1
    r = analyze(chain, (2,))
    e = r["edge_ident"]
    assert e[1, 2] == 1 and e[2, 1] == 0 and e[0, 1] == -1 and e[1, 0] == -1
    assert r["pa_agree"].tolist() == [False, False, True]
    assert r["noeffect"][2, 0] and not r["noeffect"][0, 2]


def test_collider_identified_from_observations():
    assert len(equivalence_class(adj_from([(0, 2), (1, 2)], 3), ())) == 1


def test_do_semantics():
    n = 3
    adj = adj_from([(0, 1)], n)                          # 0 -> 1 ; variable 2 indépendante
    parents = [np.flatnonzero(adj[:, j]) for j in range(n)]
    mechs = [None if len(p) == 0 else (lambda Z: 1.0 * Z[:, 0]) for p in parents]
    scm = SCM(n, adj, topo_order(adj), parents, mechs, ["root", "linear", "root"], np.full(n, 0.3), "gaussian")
    rng, M = np.random.default_rng(0), 20000
    obs = simulate(scm, rng, np.full(M, -1), np.zeros(M, int))
    do0 = simulate(scm, rng, np.full(M, 0), np.full(M, 220))
    assert (do0[:, 0] == 220).all()
    h = lambda x: np.bincount(x // BIN_W, minlength=K_BINS) / len(x)
    assert 0.5 * np.abs(h(do0[:, 2]) - h(obs[:, 2])).sum() < 0.03     # non-descendant : inchangé
    assert do0[:, 1].mean() > obs[:, 1].mean() + 40                    # descendant : suit la cause forcée


def test_world_invariants():
    for cfg in (TRAIN, OOD):
        for w in world_set(0, cfg, 15):
            n = w.n
            assert 3 <= n <= 5
            assert w.data.shape == (cfg.n_rec, 5) and w.data.dtype == np.uint8 and (w.data[:, n:] == 0).all()
            seen = set(map(int, np.unique(w.rec_target)))
            assert seen == {-1, *w.targets}
            assert np.allclose(w.q_probs.sum(1), 1.0, atol=1e-5)
            assert (w.queries[:, 0] != w.queries[:, 2]).all() and w.queries[:, 1].max() <= 255
            assert not ((w.adj == 1) & (w.edge_ident == 0)).any()      # la vérité appartient à sa propre classe
            assert not ((w.adj == 0) & (w.edge_ident == 1)).any()
            if len(w.targets) == n:                                    # tout intervenu => DAG entièrement identifié
                off = ~np.eye(n, dtype=bool)
                assert (w.edge_ident[:n, :n][off] >= 0).all()


def test_determinism():
    for x, y in zip(world_set(7, TRAIN, 3), world_set(7, TRAIN, 3)):
        assert (x.data == y.data).all() and (x.q_probs == y.q_probs).all() and x.targets == y.targets


def test_ood_families_are_disjoint_and_used():
    for k in ("graphs", "mechs", "noises"):
        assert not set(getattr(TRAIN, k)) & set(getattr(OOD, k))
    for w in world_set(3, OOD, 10):
        assert w.meta["graph"] in OOD.graphs and w.meta["noise"] in OOD.noises
        assert set(w.meta["mechs"]) <= set(OOD.mechs) | {"root"}


def test_metric_helpers():
    u, pt = np.full(K_BINS, 1 / K_BINS), np.zeros(K_BINS)
    pt[0] = 1
    assert abs(kl(pt, u) - np.log(K_BINS)) < 1e-9 and abs(kl(u, u)) < 1e-9
    assert ece(np.full(4, 0.5), np.array([1, 0, 1, 0])) == pytest.approx(0)
    assert ece(np.ones(4), np.zeros(4)) == pytest.approx(1)
    good = risk_coverage(np.array([.9, .8, .2, .1]), np.array([0, 0, 1, 1])).mean()
    bad = risk_coverage(np.array([.1, .2, .8, .9]), np.array([0, 0, 1, 1])).mean()
    assert good < bad


def test_baselines_run_and_truth_is_best():
    ws = world_set(1, TRAIN, 30)
    r = {f.__name__: evaluate(f, ws) for f in (predict_truth, predict_kernel, predict_marginal, predict_uniform)}
    t = r["predict_truth"]
    assert t["edge_acc_ident"] == 1.0 and t["do_kl_ident"] < 1e-6 and t["edge_ece"] < 1e-9
    for name in ("predict_kernel", "predict_marginal", "predict_uniform"):
        assert np.isfinite(r[name]["edge_acc_ident"])
        assert r[name]["do_kl_ident"] > t["do_kl_ident"] and r[name]["do_aurc"] > t["do_aurc"]