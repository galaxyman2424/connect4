"""Small PyTorch helpers shared by dqn.py, ppo.py and distill.py."""

import numpy as np
import torch
import torch.nn.functional as F

from nn.encode import encode, legal_mask

NEG = -1e9


def planes(curs, masks, device):
    """uint64 arrays -> network input (B,4,6,7) float tensor on device."""
    x = encode(np.asarray(curs, dtype=np.uint64), np.asarray(masks, dtype=np.uint64))
    return torch.from_numpy(x).to(device, non_blocking=True)


def legal_t(masks, device):
    return torch.from_numpy(legal_mask(np.asarray(masks, dtype=np.uint64))).to(device)


def masked_logits(logits, legal):
    return logits.float().masked_fill(~legal, NEG)


def masked_log_softmax(logits, legal):
    return F.log_softmax(masked_logits(logits, legal), 1)


def mirror_x(x):
    return x.flip(3)


def mirror_cols(t):
    return t.flip(1)


def build_or_init(blocks, channels, init, device):
    """New C4Net, or weights from a checkpoint (.pt) for warm starts (e.g. a supervised run)."""
    from nn.model import build, load_checkpoint
    if init:
        model, _ = load_checkpoint(init, device)
        model.train()
        return model
    return build({"blocks": blocks, "channels": channels}).to(device)


def amp_ctx(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


class FoldedNet(torch.nn.Module):
    """An exported model.npz (BatchNorm already folded) as a PyTorch module, so any
    trained network - whatever produced it - can run on the GPU (e.g. as a teacher)."""

    def __init__(self, npz_path):
        super().__init__()
        import json
        with np.load(npz_path, allow_pickle=False) as z:
            w = {k: torch.from_numpy(z[k].astype(np.float32)) for k in z.files if not k.endswith("_json")}
            self.config = json.loads(str(z["config_json"]))
        self.blocks = int(self.config["blocks"])
        self.p = torch.nn.ParameterDict({k.replace(".", "_"): torch.nn.Parameter(v, requires_grad=False)
                                         for k, v in w.items()})

    def forward(self, x):
        P = self.p
        h = F.relu(F.conv2d(x, P["stem_w"], P["stem_b"], padding=1))
        for i in range(self.blocks):
            y = F.relu(F.conv2d(h, P["b%d_c1_w" % i], P["b%d_c1_b" % i], padding=1))
            y = F.conv2d(y, P["b%d_c2_w" % i], P["b%d_c2_b" % i], padding=1)
            h = F.relu(h + y)
        b = h.shape[0]
        p = F.relu(torch.einsum("bchw,oc->bohw", h, P["pol_w"]) + P["pol_b"][None, :, None, None]).reshape(b, -1)
        p = p @ P["pol_fc_w"].T + P["pol_fc_b"]
        v = F.relu(torch.einsum("bchw,oc->bohw", h, P["val_w"]) + P["val_b"][None, :, None, None]).reshape(b, -1)
        v = F.relu(v @ P["val_fc1_w"].T + P["val_fc1_b"])
        return p, v @ P["val_fc2_w"].T + P["val_fc2_b"]


def load_gpu_evaluator(path, device="auto"):
    """Model file (.npz or .pt) -> batched evaluator on the best device (GPU if present)."""
    from nn.common import TorchEvaluator, pick_device
    dev = pick_device(device)
    if path.endswith(".npz"):
        return TorchEvaluator(FoldedNet(path), dev)
    from nn.model import load_checkpoint
    model, _ = load_checkpoint(path, dev)
    return TorchEvaluator(model, dev)
