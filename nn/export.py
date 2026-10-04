"""Export a PyTorch checkpoint to a numpy .npz (BatchNorm folded into convs).

    python nn/export.py runs/<run>/checkpoints/latest.pt runs/<run>/model.npz

Training scripts call export_model() automatically after every generation /
epoch, so you normally never run this by hand.
"""

import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402


def _fold(conv, bn):
    """Return (w, b) of conv followed by eval-mode BatchNorm."""
    w = conv.weight.detach().cpu().double().numpy()
    b = conv.bias.detach().cpu().double().numpy() if conv.bias is not None else np.zeros(w.shape[0])
    g = bn.weight.detach().cpu().double().numpy()
    beta = bn.bias.detach().cpu().double().numpy()
    mean = bn.running_mean.detach().cpu().double().numpy()
    var = bn.running_var.detach().cpu().double().numpy()
    scale = g / np.sqrt(var + bn.eps)
    w = w * scale[:, None, None, None]
    b = (b - mean) * scale + beta
    return w.astype(np.float32), b.astype(np.float32)


def export_model(model, path, meta=None):
    """Write ``model`` (nn.model.C4Net) to ``path`` (.npz).  Atomic replace."""
    arrays = {}
    arrays["stem.w"], arrays["stem.b"] = _fold(model.stem_conv, model.stem_bn)
    for i, blk in enumerate(model.blocks):
        arrays["b%d.c1.w" % i], arrays["b%d.c1.b" % i] = _fold(blk.conv1, blk.bn1)
        arrays["b%d.c2.w" % i], arrays["b%d.c2.b" % i] = _fold(blk.conv2, blk.bn2)
    w, b = _fold(model.pol_conv, model.pol_bn)
    arrays["pol.w"], arrays["pol.b"] = w[:, :, 0, 0], b
    w, b = _fold(model.val_conv, model.val_bn)
    arrays["val.w"], arrays["val.b"] = w[:, :, 0, 0], b
    for name, lin in (("pol_fc", model.pol_fc), ("val_fc1", model.val_fc1), ("val_fc2", model.val_fc2)):
        arrays[name + ".w"] = lin.weight.detach().cpu().float().numpy()
        arrays[name + ".b"] = lin.bias.detach().cpu().float().numpy()
    arrays["config_json"] = np.array(json.dumps(model.config))
    arrays["meta_json"] = np.array(json.dumps(meta or {}))
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return path


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    a = ap.parse_args()
    from nn.model import load_checkpoint
    model, obj = load_checkpoint(a.checkpoint)
    meta = {k: v for k, v in obj.items() if k not in ("config", "state_dict", "optimizer")
            and isinstance(v, (int, float, str))}
    export_model(model, a.out, meta)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
