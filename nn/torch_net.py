"""Exported model.npz on PyTorch (GPU or CPU), with the SAME move choice as nn/numpy_net.py.

    from nn.torch_net import TorchNpzEvaluator
    ev = TorchNpzEvaluator("runs/score_8x128_s0_r1/best.npz", device="auto")
    pri, val, wdl = ev.evaluate(curs, masks)        # same format as NumpyNet.evaluate

Why this exists: the older torch paths (rl/torchutil.FoldedNet, nn/gpu_server.FoldedNet)
only run the policy + WDL heads.  For score-head nets (nn/train_score.py) the move is chosen
by the SCORE head with mirror averaging (model.npz meta: play_head / play_temp /
play_policy_weight / play_mirror), so on those paths a score net silently plays a different,
weaker move than on the website.  This evaluator reproduces NumpyNet.evaluate() exactly
(tests/test_speedups.py checks it to 1e-4 and the same argmax), including the optional
PCA input projection (inproj.*, nn/model.py).

Speed (measured on the cloud test box, 8x128 score net, 1 CPU thread):
    batch 1:   numpy 5.6 ms, torch 2.8 ms
    batch 256: numpy 790 ms, torch (2 threads) 160 ms; a CUDA GPU is far faster again.
nn.evaluate uses it with --backend torch.
"""

import json

import numpy as np

from .encode import encode, legal_mask

SCORE_SCALE = 10.0


def _torch():
    import torch
    return torch


class TorchNpzNet:
    """Weights of a model.npz as torch tensors + forward_all()."""

    def __init__(self, path, device):
        torch = _torch()
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if not k.endswith("_json")}
            self.config = json.loads(str(z["config_json"]))
            self.meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
        self.device = device
        self.w = {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)).to(device)
                  for k, v in arrays.items()}
        self.blocks = int(self.config["blocks"])
        self.has_score = "sc.w" in self.w
        self.has_proj = "inproj.U" in self.w

    def forward_all(self, x):
        torch = _torch()
        F = torch.nn.functional
        w = self.w
        if self.has_proj:
            b = x.shape[0]
            z = x[:, :2].reshape(b, -1)
            z = (z - w["inproj.mean"]) @ w["inproj.U"] @ w["inproj.U"].T + w["inproj.mean"]
            x = torch.cat([z.reshape(b, 2, x.shape[2], x.shape[3]), x[:, 2:]], 1)
        h = F.relu(F.conv2d(x, w["stem.w"], w["stem.b"], padding=1))
        for i in range(self.blocks):
            y = F.relu(F.conv2d(h, w["b%d.c1.w" % i], w["b%d.c1.b" % i], padding=1))
            y = F.conv2d(y, w["b%d.c2.w" % i], w["b%d.c2.b" % i], padding=1)
            h = F.relu(h + y)
        p = F.relu(F.conv2d(h, w["pol.w"][:, :, None, None], w["pol.b"])).flatten(1)
        p = F.linear(p, w["pol_fc.w"], w["pol_fc.b"])
        v = F.relu(F.conv2d(h, w["val.w"][:, :, None, None], w["val.b"])).flatten(1)
        v = F.linear(F.relu(F.linear(v, w["val_fc1.w"], w["val_fc1.b"])), w["val_fc2.w"], w["val_fc2.b"])
        s = None
        if self.has_score:
            s = F.relu(F.conv2d(h, w["sc.w"][:, :, None, None], w["sc.b"])).flatten(1)
            s = F.linear(F.relu(F.linear(s, w["sc_fc1.w"], w["sc_fc1.b"])), w["sc_fc2.w"], w["sc_fc2.b"])
            s = s * SCORE_SCALE
        return p, v, s


class TorchNpzEvaluator:
    """Drop-in replacement for NumpyNet: .evaluate(curs, masks) -> (priors, values, wdl),
    numpy float64/float64/float32 arrays, same head selection and mirror averaging."""

    def __init__(self, path, device="auto", threads=None):
        torch = _torch()
        from .common import pick_device
        dev = pick_device(device) if isinstance(device, str) else device
        if threads and dev.type == "cpu":
            torch.set_num_threads(int(threads))
        self.net = TorchNpzNet(path, dev)
        self.path = path
        self.device = dev
        self.config = self.net.config
        self.meta = self.net.meta
        self.has_score = self.net.has_score
        self.play_head = self.meta.get("play_head", "policy") if self.has_score else "policy"
        self.play_temp = float(self.meta.get("play_temp", 0.5))
        self.play_policy_weight = float(self.meta.get("play_policy_weight", 0.0))

    def forward_all(self, x):
        """numpy in, numpy out (same as NumpyNet.forward_all)."""
        torch = _torch()
        with torch.inference_mode():
            p, v, s = self.net.forward_all(torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(self.device))
            return (p.cpu().numpy(), v.cpu().numpy(), None if s is None else s.cpu().numpy())

    def evaluate(self, curs, masks):
        torch = _torch()
        x = torch.from_numpy(encode(curs, masks)).to(self.device, non_blocking=True)
        legal = torch.from_numpy(legal_mask(masks)).to(self.device)
        with torch.inference_mode():
            logits, wdl_logits, scores = self.net.forward_all(x)
            if self.meta.get("play_mirror"):
                l2, w2, s2 = self.net.forward_all(x.flip(3))
                logits, wdl_logits = (logits + l2.flip(1)) / 2, (wdl_logits + w2) / 2
                if scores is not None:
                    scores = (scores + s2.flip(1)) / 2
            if self.play_head in ("score", "combo") and scores is not None:
                z = scores / self.play_temp
                if self.play_head == "combo" and self.play_policy_weight:
                    lp = torch.log_softmax(logits.masked_fill(~legal, float("-inf")), 1)
                    z = z + self.play_policy_weight * torch.where(legal, lp, torch.zeros_like(lp))
                logits = z
            logits = logits.masked_fill(~legal, float("-inf"))
            pri = torch.softmax(logits.double(), 1)
            wdl = torch.softmax(wdl_logits.float(), 1)
            val = (wdl[:, 0] - wdl[:, 2]).double()
        return pri.cpu().numpy(), val.cpu().numpy(), wdl.cpu().numpy()
