"""Statistics helpers: win-rate matrix, Bradley-Terry Elo with cluster
bootstrap CIs, Wilson intervals, binomial and McNemar tests."""

import math

import numpy as np
from scipy import optimize, stats

ELO_SCALE = 400.0 / math.log(10.0)


def wilson(k, n, z=1.96):
    """Wilson score interval for k successes out of n (k may be fractional,
    e.g. draws counted as 0.5)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def winrate_matrix(games, players):
    """games: iterable of (a, b, score_a).  Returns (score matrix S[i,j] =
    mean score of i vs j, count matrix N)."""
    idx = {p: i for i, p in enumerate(players)}
    k = len(players)
    s = np.zeros((k, k))
    n = np.zeros((k, k))
    for a, b, sa in games:
        i, j = idx[a], idx[b]
        s[i, j] += sa
        s[j, i] += 1 - sa
        n[i, j] += 1
        n[j, i] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return m, n


def bradley_terry(s, n, anchor=0, anchor_value=1000.0, ridge=1e-6):
    """Maximum-likelihood Bradley-Terry on the Elo scale.

    s[i,j] = total score of i against j (draw = 0.5), n[i,j] = games.
    P(i beats j) = 1 / (1 + 10^((R_j - R_i)/400)).  Ratings are anchored so
    R[anchor] = anchor_value.  A tiny ridge keeps the optimum finite if a
    player is undefeated (it has no visible effect otherwise)."""
    k = s.shape[0]
    free = [i for i in range(k) if i != anchor]

    def unpack(x):
        th = np.zeros(k)
        th[free] = x
        return th

    def nll(x):
        th = unpack(x)
        d = th[:, None] - th[None, :]
        # log sigmoid(d) computed stably
        ll = s * -np.logaddexp(0, -d)
        val = -(ll.sum()) + ridge * (x ** 2).sum()
        p = 1.0 / (1.0 + np.exp(-d))
        g = (s - n * p).sum(axis=1)   # d/dtheta_i of log-likelihood
        grad = -g[free] + 2 * ridge * x
        return val, grad

    res = optimize.minimize(nll, np.zeros(k - 1), jac=True, method="L-BFGS-B")
    th = unpack(res.x)
    return anchor_value + ELO_SCALE * th


def elo_with_bootstrap(games, players, anchor_player, anchor_value=1000.0,
                       reps=1000, seed=0):
    """games: list of dicts with keys a, b, score_a, cluster (games sharing a
    cluster id, e.g. the two colour-swapped games of one opening in one
    pairing, are resampled together).  Returns dict player -> (elo, lo, hi)
    and the bootstrap matrix."""
    idx = {p: i for i, p in enumerate(players)}
    k = len(players)
    anchor = idx[anchor_player]
    # group by pairing -> list of clusters (each = (s_ij contribution, n))
    pairings = {}
    for g in games:
        i, j = idx[g["a"]], idx[g["b"]]
        key = (min(i, j), max(i, j))
        sa = g["score_a"] if i < j else 1 - g["score_a"]
        pairings.setdefault(key, {}).setdefault(g["cluster"], []).append(sa)

    def mats(choice):
        s = np.zeros((k, k))
        n = np.zeros((k, k))
        for (i, j), clusters in pairings.items():
            vals = list(clusters.values())
            pick = choice(len(vals))
            tot = sum(sum(vals[c]) for c in pick)
            cnt = sum(len(vals[c]) for c in pick)
            s[i, j] += tot
            s[j, i] += cnt - tot
            n[i, j] += cnt
            n[j, i] += cnt
        return s, n

    s, n = mats(lambda m: range(m))
    point = bradley_terry(s, n, anchor, anchor_value)
    rng = np.random.default_rng(seed)
    boot = np.zeros((reps, k))
    for r in range(reps):
        s_b, n_b = mats(lambda m: rng.integers(0, m, size=m))
        boot[r] = bradley_terry(s_b, n_b, anchor, anchor_value)
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    return {p: (float(point[i]), float(lo[i]), float(hi[i])) for p, i in idx.items()}, boot


def binom_test(wins, losses):
    """Two-sided exact binomial (sign) test on decisive games."""
    n = wins + losses
    if n == 0:
        return 1.0
    return float(stats.binomtest(wins, n, 0.5).pvalue)


def mcnemar(b, c):
    """Exact McNemar test: b = cases only A correct, c = only B correct."""
    n = b + c
    if n == 0:
        return 1.0
    return float(stats.binomtest(b, n, 0.5).pvalue)
