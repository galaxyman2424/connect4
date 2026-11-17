"""Core helpers for research/interp.py (mechanistic interpretability of C4Net).

* TorchNet  - forward pass of an exported .npz network (BatchNorm already folded,
              exactly the arithmetic of nn/numpy_net.py) written in torch, with hooks to
              capture activations, edit a layer's output, and skip / replace residual
              branches.  tests/test_research_interp.py checks it against NumpyNet.
* concepts  - vectorised numpy labels computed from (cur, mask) bitboards.
* probes    - multi-head linear probes (independent heads trained jointly) in torch,
              with per-head early stopping on a validation split.
* cka       - linear CKA.

Row convention everywhere: row 0 = TOP row of the board (nn/encode.py); "height" and
"row from bottom" are spelled out where used.
"""

import numpy as np
import torch
import torch.nn.functional as F

from engine.board import BOARD_MASK
from nn.encode import bits_to_grid
from nn.numpy_net import NumpyNet

ROWS, COLS = 6, 7
ILLEGAL = -128

# exact fp32 (TF32 convs/matmuls on Ampere drift ~1e-2 from the numpy reference)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


# ----------------------------------------------------------------------------- network
class TorchNet:
    """Torch re-implementation of NumpyNet with activation access.

    Layer names: 'stem', 'block1'..'blockB' (residual stream after the block's final
    ReLU), 'pol_conv' (policy 1x1 conv after BN+ReLU, 2x6x7), 'val_conv' (value 1x1 conv
    after BN+ReLU, 4x6x7), 'val_hidden' (value linear(64) after ReLU)."""

    def __init__(self, path, device="cpu"):
        self.ref = NumpyNet(path)
        self.path = path
        self.device = device
        self.blocks = int(self.ref.config["blocks"])
        self.channels = int(self.ref.w["stem.w"].shape[0])
        self.params = int(sum(v.size for v in self.ref.w.values()))
        self.w = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in self.ref.w.items()}
        self.w["pol.w4"] = self.w["pol.w"][:, :, None, None]
        self.w["val.w4"] = self.w["val.w"][:, :, None, None]

    @property
    def stream_layers(self):
        return ["stem"] + ["block%d" % (i + 1) for i in range(self.blocks)]

    @property
    def layers(self):
        return self.stream_layers + ["pol_conv", "val_conv", "val_hidden"]

    def run(self, x, capture=(), edit=None, block_mode=None, capture_branch=False):
        """x (B,4,6,7) float tensor.  Returns (policy logits, wdl logits, acts dict).
        edit: {layer: fn(h) -> h} applied to that layer's output before it is used.
        block_mode: {k (1-based): 'identity' | 'zero_branch' | ('mean_branch', y_mean (C,6,7))}.
        capture_branch: also store each block's residual branch output as 'blockK_branch'."""
        w = self.w
        acts = {}
        edit = edit or {}
        block_mode = block_mode or {}

        def post(name, h):
            if name in edit:
                h = edit[name](h)
            if name in capture:
                acts[name] = h
            return h

        h = F.relu(F.conv2d(x, w["stem.w"], w["stem.b"], padding=1))
        h = post("stem", h)
        for i in range(self.blocks):
            k = i + 1
            mode = block_mode.get(k)
            if mode != "identity":
                y = F.relu(F.conv2d(h, w["b%d.c1.w" % i], w["b%d.c1.b" % i], padding=1))
                y = F.conv2d(y, w["b%d.c2.w" % i], w["b%d.c2.b" % i], padding=1)
                if capture_branch:
                    acts["block%d_branch" % k] = y
                if mode == "zero_branch":
                    y = torch.zeros_like(y)
                elif isinstance(mode, tuple) and mode[0] == "mean_branch":
                    y = mode[1].unsqueeze(0).expand_as(y)
                h = F.relu(h + y)
            h = post("block%d" % k, h)
        p = F.relu(F.conv2d(h, w["pol.w4"], w["pol.b"]))
        p = post("pol_conv", p)
        logits = p.flatten(1) @ w["pol_fc.w"].T + w["pol_fc.b"]
        v = F.relu(F.conv2d(h, w["val.w4"], w["val.b"]))
        v = post("val_conv", v)
        vh = F.relu(v.flatten(1) @ w["val_fc1.w"].T + w["val_fc1.b"])
        vh = post("val_hidden", vh)
        wdl = vh @ w["val_fc2.w"].T + w["val_fc2.b"]
        return logits, wdl, acts


@torch.no_grad()
def run_batched(net, X, bs=4096, **kw):
    """X: float32 numpy (N,4,6,7) or tensor.  Returns (logits, wdl) as torch tensors on device."""
    outs_l, outs_v = [], []
    for s in range(0, len(X), bs):
        xb = torch.as_tensor(X[s:s + bs], device=net.device, dtype=torch.float32)
        l, v, _ = net.run(xb, **kw)
        outs_l.append(l)
        outs_v.append(v)
    return torch.cat(outs_l), torch.cat(outs_v)


@torch.no_grad()
def layer_acts(net, X, layer, bs=4096, pooled=False, dtype=torch.float16):
    """Activations of one layer for all of X, flattened (N, D) (or spatially averaged)."""
    out = []
    for s in range(0, len(X), bs):
        xb = torch.as_tensor(X[s:s + bs], device=net.device, dtype=torch.float32)
        if layer == "input":
            a = xb
        else:
            a = net.run(xb, capture=(layer,))[2][layer]
        if pooled and a.dim() == 4:
            a = a.mean((2, 3))
        out.append(a.flatten(1).to(dtype))
    return torch.cat(out)


# ----------------------------------------------------------------------------- concepts
_U = np.uint64


def win_cells_bits(pos, mask):
    """Vectorised engine.board.winning_cells_mask: bitboard of EMPTY cells completing four
    for `pos` (playable or not).  pos, mask: uint64 arrays."""
    pos = np.asarray(pos, dtype=np.uint64)
    mask = np.asarray(mask, dtype=np.uint64)
    H = 7

    def sl(a, k):
        return a << _U(k)

    def sr(a, k):
        return a >> _U(k)

    r = sl(pos, 1) & sl(pos, 2) & sl(pos, 3)
    for p in (H, H - 1, H + 1):
        p2 = sl(pos, p) & sl(pos, 2 * p)
        r |= p2 & sl(pos, 3 * p)
        r |= p2 & sr(pos, p)
        p2 = sr(pos, p) & sr(pos, 2 * p)
        r |= p2 & sl(pos, p)
        r |= p2 & sr(pos, 3 * p)
    return r & (_U(BOARD_MASK) ^ mask)


def board_features(cur, mask):
    """Grids and per-position quantities.  All grids (N,6,7) bool, row 0 = top."""
    cur = np.asarray(cur, dtype=np.uint64)
    mask = np.asarray(mask, dtype=np.uint64)
    me = bits_to_grid(cur).astype(bool)
    them = bits_to_grid(cur ^ mask).astype(bool)
    occ = me | them
    heights = occ.sum(1).astype(np.int64)                         # (N,7) stones per column
    rows = np.arange(ROWS)[None, :, None]
    playable = rows == (ROWS - 1 - heights)[:, None, :]           # landing cell of each column
    own_thr = bits_to_grid(win_cells_bits(cur, mask)).astype(bool)
    opp_thr = bits_to_grid(win_cells_bits(cur ^ mask, mask)).astype(bool)
    own_imm = own_thr & playable
    opp_imm = opp_thr & playable
    n_me, n_them = me.sum((1, 2)), them.sum((1, 2))
    return {"me": me, "them": them, "empty": ~occ, "playable": playable,
            "own_thr": own_thr, "opp_thr": opp_thr, "own_imm": own_imm, "opp_imm": opp_imm,
            "heights": heights, "ply": (n_me + n_them).astype(np.int64),
            "first": n_me == n_them, "legal": heights < ROWS}


def _double(thr, imm):
    """>=2 playable immediate wins, or a playable threat with another threat directly above it."""
    two = imm.reshape(len(imm), -1).sum(1) >= 2
    above = np.zeros_like(thr)
    above[:, 1:, :] = thr[:, :-1, :]          # cell (r,c) gets thr of (r-1,c), the cell above it
    stacked = (imm & above).reshape(len(imm), -1).any(1)
    return two | stacked


def compute_concepts(cur, mask, scores, ply=None):
    """Model-independent concept labels.  scores (N,7) int (Pons, side to move, -128 = full)."""
    f = board_features(cur, mask)
    n = len(f["ply"])
    scores = np.asarray(scores).astype(np.int64)
    legal = scores != ILLEGAL
    assert (legal == f["legal"]).all(), "illegal-column sentinel disagrees with the board"
    best = np.where(legal, scores, -10_000).max(1)
    opt = legal & (scores == best[:, None])
    outcome = np.where(best > 0, 0, np.where(best == 0, 1, 2))     # WDL order of the value head
    plyv = f["ply"]
    if ply is not None:
        assert (np.asarray(ply) == plyv).all()
    rows_from_bottom = (ROWS - 1 - np.arange(ROWS))[None, :, None]  # 0 = bottom
    odd = (rows_from_bottom % 2) == 0          # 1-indexed rows 1,3,5 from the bottom = "odd"
    landing = np.where(f["legal"], ROWS - 1 - f["heights"], ROWS)    # row from top, 6 = full
    c = {
        "heights": f["heights"].astype(np.float32),
        "landing_row": landing,
        "ply": plyv.astype(np.float32),
        "first_to_move": f["first"].astype(np.int64),
        "win_now": f["own_imm"].any(1),
        "opp_win_now": f["opp_imm"].any(1),
        "own_threats": f["own_thr"].sum((1, 2)).astype(np.float32),
        "opp_threats": f["opp_thr"].sum((1, 2)).astype(np.float32),
        "own_double": _double(f["own_thr"], f["own_imm"]).astype(np.int64),
        "opp_double": _double(f["opp_thr"], f["opp_imm"]).astype(np.int64),
        "own_odd_thr": (f["own_thr"] & odd).sum((1, 2)).astype(np.float32),
        "own_even_thr": (f["own_thr"] & ~odd).sum((1, 2)).astype(np.float32),
        "opp_odd_thr": (f["opp_thr"] & odd).sum((1, 2)).astype(np.float32),
        "opp_even_thr": (f["opp_thr"] & ~odd).sum((1, 2)).astype(np.float32),
        "opt_set": opt,
        "outcome": outcome,
        "score": best.astype(np.float32),
        "phase": np.where(plyv < 12, 0, np.where(plyv >= 24, 2, 1)),
    }
    extra = {"features": f, "legal": legal, "opt": opt, "outcome": outcome, "best": best, "n": n}
    return c, extra


def flip_planes(x):
    """Same stones, opponent to move: swap 'me'/'them' planes and toggle the colour plane.
    (Off-distribution: piece counts no longer match the colour plane's usual meaning.)"""
    y = x.copy()
    y[:, 0], y[:, 1] = x[:, 1], x[:, 0]
    y[:, 3] = 1.0 - x[:, 3]
    return y


# ----------------------------------------------------------------------------- probes
def _auc(score, y):
    """Mann-Whitney AUC (torch, 1-D).  NaN if one class is missing."""
    y = y.bool()
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = torch.empty(len(score), dtype=torch.float64, device=score.device)
    r[score.argsort()] = torch.arange(1, len(score) + 1, dtype=torch.float64, device=score.device)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def head_metric(kind, out, y, K=None):
    """out: probe outputs for this head on an eval set; y: label tensor."""
    if kind == "multibin":
        v = [_auc(out[:, j], y[:, j]) for j in range(y.shape[1])]
        v = [a for a in v if a == a]
        return float(np.mean(v)) if v else float("nan")
    if kind == "cat":
        return float((out.argmax(1) == y).float().mean())
    if kind == "multicat":
        pred = out.view(len(out), -1, K).argmax(2)
        return float((pred == y).float().mean())
    if kind == "reg":
        sse = ((out - y) ** 2).sum(0)
        sst = ((y - y.mean(0)) ** 2).sum(0)
        ok = sst > 1e-9
        return float((1 - sse[ok] / sst[ok]).mean()) if ok.any() else float("nan")
    raise ValueError(kind)


def head_size(kind, y, K):
    if kind == "multibin":
        return y.shape[1]
    if kind == "cat":
        return K
    if kind == "multicat":
        return y.shape[1] * K
    if kind == "reg":
        return 1 if y.ndim == 1 else y.shape[1]
    raise ValueError(kind)


def chance_score(kind, y_tr, y_te, K=None):
    """Score of the best constant predictor (majority / mean) fitted on train, on test."""
    if kind == "multibin":
        return 0.5
    if kind == "cat":
        maj = np.bincount(y_tr, minlength=K).argmax()
        return float((y_te == maj).mean())
    if kind == "multicat":
        accs = [(y_te[:, j] == np.bincount(y_tr[:, j], minlength=K).argmax()).mean() for j in range(y_tr.shape[1])]
        return float(np.mean(accs))
    if kind == "reg":
        yt = y_te.reshape(len(y_te), -1)
        ytr = y_tr.reshape(len(y_tr), -1)
        m = ytr.mean(0)
        sse = ((yt - m) ** 2).sum(0)
        sst = ((yt - yt.mean(0)) ** 2).sum(0)
        ok = sst > 1e-9
        return float((1 - sse[ok] / sst[ok]).mean())
    raise ValueError(kind)


def train_probes(X, heads, tr, va, te, rounds=15, iters=10, l2=1e-4, seed=0,
                 shuffle=False, return_weights=()):
    """Independent linear probes sharing one input.

    X: (N,D) tensor (any float dtype) on the compute device.  heads: list of dicts
    {id, kind ('multibin'|'cat'|'multicat'|'reg'), y (N,...) numpy, K}.
    tr/va/te: numpy index arrays.  Features are standardised on train; regression
    targets too.  Each head keeps the test score of the epoch with its best validation
    score.  Optimiser: full-batch L-BFGS (strong Wolfe), `rounds` x `iters` iterations,
    validation checked after each round.  Weights start at zero.
    shuffle=True permutes train and validation labels (test labels stay real).
    Returns ({id: test_score}, {id: (W (D,k) in raw-feature units, b)} for return_weights)."""
    dev = X.device
    rng = np.random.default_rng(seed)
    D = X.shape[1]
    tr_t = torch.as_tensor(tr, device=dev)
    # standardisation (chunked to bound memory)
    s1 = torch.zeros(D, device=dev, dtype=torch.float64)
    s2 = torch.zeros(D, device=dev, dtype=torch.float64)
    for s in range(0, len(tr), 8192):
        xb = X[tr_t[s:s + 8192]].double()
        s1 += xb.sum(0)
        s2 += (xb ** 2).sum(0)
    mu = (s1 / len(tr)).float()
    sd = ((s2 / len(tr) - (s1 / len(tr)) ** 2).clamp_min(0).sqrt() + 1e-3).float()

    lab_tr, lab_va = tr, va
    if shuffle:
        lab_tr = tr[rng.permutation(len(tr))]
        lab_va = va[rng.permutation(len(va))]

    specs, off = [], 0
    for h in heads:
        y = h["y"]
        kind, K = h["kind"], h.get("K")
        k = head_size(kind, y, K)
        if kind == "reg":
            y2 = y.reshape(len(y), -1).astype(np.float64)
            m, s = y2[lab_tr].mean(0), y2[lab_tr].std(0) + 1e-9
            yt = torch.tensor((y2 - m) / s, dtype=torch.float32, device=dev)
        elif kind == "multibin":
            yt = torch.tensor(y.astype(np.float32), device=dev)
        else:
            yt = torch.tensor(y.astype(np.int64), device=dev)
        specs.append({"id": h["id"], "kind": kind, "K": K, "sl": slice(off, off + k), "y": yt})
        off += k
    W = torch.zeros(D, off, device=dev, requires_grad=True)
    b = torch.zeros(off, device=dev, requires_grad=True)
    Xtr = torch.empty(len(tr), D, device=dev)                     # standardised train matrix
    for s in range(0, len(tr), 8192):
        Xtr[s:s + 8192] = (X[tr_t[s:s + 8192]].float() - mu) / sd

    def outputs(idx):
        res = []
        it = torch.as_tensor(idx, device=dev)
        for s in range(0, len(idx), 8192):
            xb = (X[it[s:s + 8192]].float() - mu) / sd
            res.append(xb @ W + b)
        return torch.cat(res)

    def loss_of(o, sp, y):
        oo = o[:, sp["sl"]]
        if sp["kind"] == "multibin":
            return F.binary_cross_entropy_with_logits(oo, y)
        if sp["kind"] == "cat":
            return F.cross_entropy(oo, y)
        if sp["kind"] == "multicat":
            return F.cross_entropy(oo.reshape(-1, sp["K"]), y.reshape(-1))
        return F.mse_loss(oo, y.view(len(y), -1))

    best_va = {sp["id"]: -1e9 for sp in specs}
    best_te = {sp["id"]: float("nan") for sp in specs}
    best_w = {}
    lab_tr_t = torch.as_tensor(lab_tr, device=dev)
    lab_va_t = torch.as_tensor(lab_va, device=dev)
    te_t = torch.as_tensor(te, device=dev)
    ytr = {sp["id"]: sp["y"][lab_tr_t] for sp in specs}
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        o = Xtr @ W + b
        loss = sum(loss_of(o, sp, ytr[sp["id"]]) for sp in specs) + l2 * (W ** 2).sum()
        loss.backward()
        return loss

    for _ in range(rounds):
        opt.step(closure)
        with torch.no_grad():
            ov, ot = outputs(va), outputs(te)
            for sp in specs:
                yv = sp["y"][lab_va_t]
                mv = head_metric(sp["kind"], ov[:, sp["sl"]], yv if sp["kind"] != "reg" else yv.view(len(yv), -1), sp["K"])
                if mv == mv and mv > best_va[sp["id"]]:
                    best_va[sp["id"]] = mv
                    yt = sp["y"][te_t]
                    best_te[sp["id"]] = head_metric(sp["kind"], ot[:, sp["sl"]],
                                                    yt if sp["kind"] != "reg" else yt.view(len(yt), -1), sp["K"])
                    if sp["id"] in return_weights:
                        Wr = (W[:, sp["sl"]] / sd[:, None]).detach().clone()
                        br = (b[sp["sl"]] - (mu[:, None] * W[:, sp["sl"]] / sd[:, None]).sum(0)).detach().clone()
                        best_w[sp["id"]] = (Wr, br)
    del Xtr
    return best_te, best_w


# ----------------------------------------------------------------------------- CKA
@torch.no_grad()
def linear_cka(X, Y):
    """Linear CKA between (N,Dx) and (N,Dy) tensors."""
    X = X.double()
    Y = Y.double()
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    xy = (Y.T @ X).pow(2).sum()
    xx = (X.T @ X).pow(2).sum().sqrt()
    yy = (Y.T @ Y).pow(2).sum().sqrt()
    return float(xy / (xx * yy))
