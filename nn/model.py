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

Optional score head (score_head=True, nn/train_score.py):
    -> score head:  1x1 conv (8) + BN + ReLU -> linear(128) + ReLU -> 7 outputs
                    = predicted Pons score of each column for the side to move
                    (scaled by SCORE_SCALE; >0 win, 0 draw, <0 loss, bigger = faster win).
The WDL head cannot tell a fast win from a slow one; this head can, and that is most of
what separates ~93% from 95%+ "optimal" on the benchmark.  forward() still returns
(policy, wdl) so every existing caller keeps working; forward_all() adds the scores.

Optional PCA input projection (pca_k=K, nn/train_score.py --pca K, research/pca_lda.py):
    the 84 stone cells (planes 0-1, flattened) are replaced by their rank-K PCA
    reconstruction  x' = (x - mean) U U^T + mean  (U: 84 x K, fitted on the training set)
    before the stem.  Planes 2-3 are untouched, so the conv trunk is unchanged and
    nn.evaluate / the arena / the website run these nets like any other (the projection is
    exported as inproj.mean / inproj.U and applied by nn/numpy_net.py and nn/torch_net.py).
    K = 84 is the identity; smaller K tests how much of the board the net needs.

Default size (6 blocks x 64 channels) has ~0.46 M parameters: plenty for a
6x7 board, and fast enough that self-play is limited by the CPU, not the GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encode import COLS, PLANES, ROWS

DEFAULT_CONFIG = {"blocks": 6, "channels": 64, "policy_channels": 2,
                  "value_channels": 4, "value_hidden": 64}
SCORE_SCALE = 10.0     # network output * SCORE_SCALE = Pons score


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
                 value_hidden=64, score_head=False, score_channels=8, score_hidden=128, pca_k=0):
        super().__init__()
        self.config = {"blocks": blocks, "channels": channels,
                       "policy_channels": policy_channels,
                       "value_channels": value_channels, "value_hidden": value_hidden}
        if score_head:
            self.config.update(score_head=True, score_channels=score_channels, score_hidden=score_hidden)
        self.pca_k = int(pca_k or 0)
        if self.pca_k:
            self.config["pca_k"] = self.pca_k
            n_in = 2 * ROWS * COLS
            self.register_buffer("inproj_mean", torch.zeros(n_in))
            self.register_buffer("inproj_U", torch.eye(n_in)[:, :self.pca_k].contiguous())
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
        self.has_score = bool(score_head)
        if self.has_score:
            self.sc_conv = nn.Conv2d(channels, score_channels, 1, bias=False)
            self.sc_bn = nn.BatchNorm2d(score_channels)
            self.sc_fc1 = nn.Linear(score_channels * n, score_hidden)
            self.sc_fc2 = nn.Linear(score_hidden, COLS)

    def set_input_projection(self, mean, components):
        """mean (84,), components U (84, K) with orthonormal columns (sklearn PCA.components_.T)."""
        if not self.pca_k:
            raise ValueError("model was built without pca_k")
        mean = torch.as_tensor(mean, dtype=torch.float32)
        comp = torch.as_tensor(components, dtype=torch.float32)
        if comp.shape != self.inproj_U.shape:
            raise ValueError("components must be %s, got %s" % (tuple(self.inproj_U.shape), tuple(comp.shape)))
        self.inproj_mean.copy_(mean.to(self.inproj_mean.device))
        self.inproj_U.copy_(comp.to(self.inproj_U.device))

    def project_input(self, x):
        if not self.pca_k:
            return x
        b = x.shape[0]
        z = x[:, :2].reshape(b, -1).float()
        z = (z - self.inproj_mean) @ self.inproj_U @ self.inproj_U.T + self.inproj_mean
        return torch.cat([z.reshape(b, 2, ROWS, COLS).to(x.dtype), x[:, 2:]], 1)

    def forward_all(self, x):
        """(policy logits (B,7), WDL logits (B,3), scores (B,7) or None)."""
        x = self.project_input(x)
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        for b in self.blocks:
            x = b(x)
        p = F.relu(self.pol_bn(self.pol_conv(x))).flatten(1)
        p = self.pol_fc(p)
        v = F.relu(self.val_bn(self.val_conv(x))).flatten(1)
        v = self.val_fc2(F.relu(self.val_fc1(v)))
        s = None
        if self.has_score:
            s = F.relu(self.sc_bn(self.sc_conv(x))).flatten(1)
            s = self.sc_fc2(F.relu(self.sc_fc1(s)))
        return p, v, s

    def forward(self, x):
        p, v, _ = self.forward_all(x)
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
