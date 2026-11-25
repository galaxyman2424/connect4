"""Training speed settings shared by nn/train_score.py and nn/train_supervised.py (--speed).

About BatchNorm: every conv in nn/model.C4Net is ALREADY followed by BatchNorm (stem, both
convs of every residual block, all three heads), and nn/export.py folds it into the conv
weights, so inference pays nothing for it.  BatchNorm's training benefit is stability (it
tolerates a higher learning rate), not raw step speed.  What actually costs time on the
RTX 3050 is listed below; measure it on your GPU with  python -m nn.train_speedtest.

  --speed off    the old behaviour (bit-for-bit, apart from the logging change below)
  --speed safe   (default) cudnn.benchmark (picks the fastest conv algorithm for the fixed
                 6x7 input once), TF32 for any fp32 matmul/conv left outside bf16 autocast
  --speed max    safe + channels_last memory layout (NHWC: what the tensor cores want for
                 bf16 convs) + torch.compile of the training forward/backward (Linux; the
                 first epoch is ~1 min slower while it compiles)

Every level also logs the losses without a GPU->CPU sync per step: the old loops called
.item() 3-5 times per batch, which stalls the GPU until the batch is finished.  The sums are
now kept on the GPU and read once per epoch (identical numbers).
"""

SPEEDS = ("off", "safe", "max")


def configure(level, device):
    """Global torch flags for ``level``; returns a dict describing what was set."""
    import torch
    info = {"speed": level}
    if level == "off" or device.type != "cuda":
        return info
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    info.update(cudnn_benchmark=True, tf32=True)
    if level == "max":
        info["channels_last"] = True
    return info


def prepare_model(level, model, device, method="forward"):
    """(model to keep / export, callable for the training forward pass = model.<method>).
    With --speed max the model is moved to channels_last and the training call is a
    torch.compile'd version of that method (same parameters, so the optimizer, EMA and
    export keep working on ``model``)."""
    import torch
    fn = getattr(model, method)
    if level != "max" or device.type != "cuda":
        return model, fn
    model = model.to(memory_format=torch.channels_last)
    fn = getattr(model, method)
    try:
        fn = torch.compile(fn, dynamic=False)
    except Exception as e:  # noqa: BLE001  (no compiler on this platform)
        print("torch.compile unavailable (%s) - channels_last only" % e, flush=True)
    return model, fn


def to_input(level, x):
    import torch
    if level == "max" and x.is_cuda:
        return x.contiguous(memory_format=torch.channels_last)
    return x
