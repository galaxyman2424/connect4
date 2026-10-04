"""The network (PyTorch).  Needs torch; only training and .pt loading import this.

C4Net = small AlphaZero-style residual network:

  input (4 x 6 x 7)
    -> stem: 3x3 conv (C channels) + BatchNorm + ReLU
    -> B residual blocks: [3x3 conv, BN, ReLU, 3x3 conv, BN] + skip, ReLU
    -> policy head: 1x1 conv (2) + BN + ReLU -> linear -> 7 logits (one per column)
    -> value head:  1x1 conv (4) + BN + ReLU -> linear(hidden) + ReLU -> 3 logits
                    (Win / Draw / Loss for the side to move)

Why a WDL value head instead of AlphaZero's single tanh output: the perfect-play
labels and self-play results are naturally win/draw/loss, cross-entropy on
three classes trains more smoothly than MSE, and we can still read a scalar
value  v = P(win) - P(loss)  in [-1, 1] for MCTS / minimax.

Default size (6 blocks x 64 channels) has ~0.46 M parameters: plenty for a
6x7 board, and fast enough that self-play is limited by the CPU, not the GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encode import COLS, PLANES, ROWS

DEFAULT_CONFIG = {"blocks": 6, "channels": 64, "policy_channels": 2,
                  "value_channels": 4, "value_hidden": 64}


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class C4Net(nn.Module):
    def __init__(self, blocks=6, channels=64, policy_channels=2, value_channels=4,
                 value_hidden=64):
        super().__init__()
        self.config = {"blocks": blocks, "channels": channels,
                       "policy_channels": policy_channels,
                       "value_channels": value_channels, "value_hidden": value_hidden}
        self.stem_conv = nn.Conv2d(PLANES, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(blocks)])
        n = ROWS * COLS
        self.pol_conv = nn.Conv2d(channels, policy_channels, 1, bias=False)
        self.pol_bn = nn.BatchNorm2d(policy_channels)
        self.pol_fc = nn.Linear(policy_channels * n, COLS)
        self.val_conv = nn.Conv2d(channels, value_channels, 1, bias=False)
        self.val_bn = nn.BatchNorm2d(value_channels)
        self.val_fc1 = nn.Linear(value_channels * n, value_hidden)
        self.val_fc2 = nn.Linear(value_hidden, 3)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        for b in self.blocks:
            x = b(x)
        p = F.relu(self.pol_bn(self.pol_conv(x))).flatten(1)
        p = self.pol_fc(p)
        v = F.relu(self.val_bn(self.val_conv(x))).flatten(1)
        v = self.val_fc2(F.relu(self.val_fc1(v)))
        return p, v          # policy logits (B,7), WDL logits (B,3)


def build(config=None):
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config or {})
    return C4Net(**cfg)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(path, model, extra=None):
    """Checkpoint = {'config', 'state_dict', ...extra}.  Loaded with weights_only=True."""
    obj = {"config": model.config, "state_dict": model.state_dict()}
    if extra:
        obj.update(extra)
    torch.save(obj, path)


def load_checkpoint(path, device="cpu"):
    obj = torch.load(path, map_location=device, weights_only=True)
    model = build(obj["config"])
    model.load_state_dict(obj["state_dict"])
    model.to(device)
    model.eval()
    return model, obj


def masked_policy(logits, legal):
    """Softmax over legal columns only.  legal: bool tensor (B,7)."""
    logits = logits.masked_fill(~legal, float("-inf"))
    return torch.softmax(logits, dim=1)


def wdl_to_value(wdl_logits):
    p = torch.softmax(wdl_logits, dim=1)
    return p[:, 0] - p[:, 2]
