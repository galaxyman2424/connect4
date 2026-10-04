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


class NumpyNet:
    """Loads model.npz.  forward(planes) -> (policy_logits (B,7), wdl_logits (B,3))."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.w = {k: z[k].astype(np.float32) for k in z.files if not k.endswith("_json")}
            self.config = json.loads(str(z["config_json"]))
            self.meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
        self.path = path
        self.blocks = int(self.config["blocks"])

    def forward(self, x):
        w = self.w
        x = np.ascontiguousarray(x, dtype=np.float32)
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
        return p, v

    def evaluate(self, curs, masks):
        """Batched evaluation for MCTS / agents.
        Returns (priors (B,7) summing to 1 over legal columns,
                 values (B,) = P(win) - P(loss) for the side to move,
                 wdl (B,3) probabilities)."""
        x = encode(curs, masks)
        logits, wdl_logits = self.forward(x)
        legal = legal_mask(masks)
        logits = np.where(legal, logits, -np.inf)
        pri = _softmax(logits)
        wdl = _softmax(wdl_logits)
        return pri, wdl[:, 0] - wdl[:, 2], wdl
