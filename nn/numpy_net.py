"""Inference of an exported network with numpy only (no PyTorch).

The website (Windows laptop), the arena and the evaluation scripts load
``model.npz`` files through this module, so they run anywhere numpy runs.
nn/export.py writes the file: BatchNorm layers are folded into the
preceding convolution (w' = w * gamma / sqrt(var + eps), b' = beta - mean *
gamma / sqrt(var + eps)), so inference is just convolutions, ReLUs and two
small dense heads.  tests/test_nn.py checks this matches PyTorch to 1e-4.
"""

import json

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .encode import encode, legal_mask


def _conv3x3(x, w, b):
    """x (B,Ci,6,7), w (Co,Ci,3,3), b (Co,) -> (B,Co,6,7)  (padding 1)."""
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    patches = sliding_window_view(xp, (3, 3), axis=(2, 3))         # (B,Ci,6,7,3,3)
    out = np.tensordot(patches, w, axes=([1, 4, 5], [1, 2, 3]))     # (B,6,7,Co)
    out += b
    return out.transpose(0, 3, 1, 2)


def _conv1x1(x, w, b):
    """x (B,Ci,6,7), w (Co,Ci) -> (B,Co,6,7)."""
    out = np.tensordot(x, w, axes=([1], [1]))                       # (B,6,7,Co)
    out += b
    return out.transpose(0, 3, 1, 2)


def _relu(x):
    return np.maximum(x, 0, out=x)


def _softmax(z, axis=1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


SCORE_SCALE = 10.0     # same as nn/model.py


def score_to_value(best):
    """Predicted Pons score of the best column -> value in (-1, 1) for search
    (C4_VALUE_SOURCE=score).  Unlike P(win) - P(loss) it ranks a fast win above a slow one:
    score 1 (slowest win) -> 0.58, score 21 (win now) -> 1.0, draw 0 -> 0."""
    a = np.minimum(np.abs(best), 21.0)
    return np.tanh(best) * (0.75 + 0.25 * a / 21.0)


def value_source():
    import os
    v = os.environ.get("C4_VALUE_SOURCE", "wdl").strip().lower()
    return v if v in ("wdl", "score") else "wdl"


class NumpyNet:
    """Loads model.npz.  forward(planes) -> (policy_logits (B,7), wdl_logits (B,3)).

    Nets trained by nn/train_score.py also have a score head (predicted Pons score per
    column).  Their meta says which head should choose moves ("play_head"):
      policy  softmax of the policy head (all older nets)
      score   softmax(score / play_temp)
      combo   softmax(score / play_temp + play_policy_weight * log policy)
    evaluate() returns those priors, so the website, arena, PolicyAgent and MCTS use the
    better head without any change on their side."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.w = {k: z[k].astype(np.float32) for k in z.files if not k.endswith("_json")}
            self.config = json.loads(str(z["config_json"]))
            self.meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
        self.path = path
        self.blocks = int(self.config["blocks"])
        self.has_score = "sc.w" in self.w
        self.play_head = self.meta.get("play_head", "policy") if self.has_score else "policy"
        self.play_temp = float(self.meta.get("play_temp", 0.5))
        self.play_policy_weight = float(self.meta.get("play_policy_weight", 0.0))
        self.value_source = value_source()      # 'wdl' (default) or 'score' (score-head nets only)

    def forward(self, x):
        p, v, _ = self.forward_all(x)
        return p, v

    def forward_all(self, x):
        """(policy logits, wdl logits, scores in Pons units or None)."""
        w = self.w
        x = np.ascontiguousarray(x, dtype=np.float32)
        if "inproj.U" in w:         # PCA input projection (nn/model.py, --pca K in nn/train_score.py)
            b = x.shape[0]
            z = x[:, :2].reshape(b, -1)
            z = (z - w["inproj.mean"]) @ w["inproj.U"] @ w["inproj.U"].T + w["inproj.mean"]
            x = np.concatenate([z.reshape(b, 2, x.shape[2], x.shape[3]).astype(np.float32), x[:, 2:]], 1)
        h = _relu(_conv3x3(x, w["stem.w"], w["stem.b"]))
        for i in range(self.blocks):
            y = _relu(_conv3x3(h, w["b%d.c1.w" % i], w["b%d.c1.b" % i]))
            y = _conv3x3(y, w["b%d.c2.w" % i], w["b%d.c2.b" % i])
            h = _relu(h + y)
        bsz = h.shape[0]
        p = _relu(_conv1x1(h, w["pol.w"], w["pol.b"])).reshape(bsz, -1)
        p = p @ w["pol_fc.w"].T + w["pol_fc.b"]
        v = _relu(_conv1x1(h, w["val.w"], w["val.b"])).reshape(bsz, -1)
        v = _relu(v @ w["val_fc1.w"].T + w["val_fc1.b"])
        v = v @ w["val_fc2.w"].T + w["val_fc2.b"]
        s = None
        if self.has_score:
            s = _relu(_conv1x1(h, w["sc.w"], w["sc.b"])).reshape(bsz, -1)
            s = _relu(s @ w["sc_fc1.w"].T + w["sc_fc1.b"])
            s = (s @ w["sc_fc2.w"].T + w["sc_fc2.b"]) * SCORE_SCALE
        return p, v, s

    def evaluate(self, curs, masks):
        """Batched evaluation for MCTS / agents.
        Returns (priors (B,7) summing to 1 over legal columns,
                 values (B,) = P(win) - P(loss) for the side to move,
                 wdl (B,3) probabilities)."""
        x = encode(curs, masks)
        logits, wdl_logits, scores = self.forward_all(x)
        if self.meta.get("play_mirror"):      # average with the mirror image (+1-1.5 points measured)
            l2, w2, s2 = self.forward_all(x[..., ::-1])
            logits, wdl_logits = (logits + l2[:, ::-1]) / 2, (wdl_logits + w2) / 2
            if scores is not None:
                scores = (scores + s2[:, ::-1]) / 2
        legal = legal_mask(masks)
        if self.play_head in ("score", "combo") and scores is not None:
            z = scores / self.play_temp
            if self.play_head == "combo" and self.play_policy_weight:
                lp = np.where(legal, logits, -np.inf)
                lp = lp - lp.max(1, keepdims=True)
                lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
                z = z + self.play_policy_weight * np.where(legal, lp, 0.0)
            logits = z
        raw_scores = scores
        logits = np.where(legal, logits, -np.inf)
        pri = _softmax(logits)
        wdl = _softmax(wdl_logits)
        val = wdl[:, 0] - wdl[:, 2]
        if raw_scores is not None and self.value_source == "score":
            val = score_to_value(np.where(legal, raw_scores, -np.inf).max(1)).astype(np.float64)
        return pri, val, wdl
