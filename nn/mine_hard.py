"""Hard-example mining: find positions the current net gets wrong, for the next training round.

    python -m nn.mine_hard --run score_8x128_s0 --positions 600000 --out nn_data/hard_r1.npz

1. The net plays itself (GPU, thousands of games at once) from random 2-4 ply openings,
   sampling moves from its policy (temperature --temp) with --eps random moves, so the
   positions are the ones a net like it actually reaches, plus some noise.
2. Unique positions of plies --min-ply..--max-ply are labelled by the Pons solver
   (all 7 column scores; same worker as nn/make_dataset.py).  Benchmark positions and
   their mirrors are excluded.
3. The net's PLAYING head (the one chosen in its model.npz meta, mirror-averaged) is
   checked against the solver.  Every position it gets wrong is kept, plus --keep-right of
   the ones it gets right (so the next round does not forget them).

Output has the nn/make_dataset.py format (cur, mask, scores, ply) and is passed to
nn/train_score.py with --hard.  The .json next to it records the on-policy error rate,
which is a useful number for the paper by itself (accuracy where the net actually plays).
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import time

import numpy as np

from .common import NN_DATA, RUNS, log, pick_device

ILLEGAL = -128


def _key(cur, mask):
    from .encode import mirror_bits
    c = np.asarray(cur, np.uint64)
    m = np.asarray(mask, np.uint64)
    mc, mm = mirror_bits(c), mirror_bits(m)
    a = np.stack([c, m], 1)
    b = np.stack([mc, mm], 1)
    use_b = (mc < c) | ((mc == c) & (mm < m))
    k = np.where(use_b[:, None], b, a)
    return [(int(x), int(y)) for x, y in k]


def selfplay_positions(model, device, n_target, games, temp, eps, min_ply, max_ply, seed, banned):
    """Play batched games; return list of move strings (unique up to mirroring)."""
    import torch
    from engine.board import Board
    from .encode import encode, legal_mask
    rng = random.Random(seed)
    out, seen = [], set(banned)
    gen = torch.Generator(device=device).manual_seed(seed)
    rounds = 0
    while len(out) < n_target:
        rounds += 1
        boards = []
        for _ in range(games):
            b = Board()
            for _ in range(rng.randint(2, 4)):
                b.drop(rng.choice(b.legal_moves()))
            boards.append(b)
        active = [b for b in boards if not b.is_terminal()]
        while active:
            curs = [b.current_bits() for b in active]
            masks = [b.mask for b in active]
            x = torch.from_numpy(encode(curs, masks)).to(device)
            legal = torch.from_numpy(legal_mask(np.array(masks, np.uint64))).to(device)
            with torch.inference_mode():
                p, _ = model(x)
                p = (p.float() / temp).masked_fill(~legal, float("-inf"))
                probs = torch.softmax(p, 1)
                uni = legal.float() / legal.float().sum(1, keepdim=True)
                r = torch.rand(len(active), 1, device=device, generator=gen) < eps
                probs = torch.where(r, uni, probs)
                cols = torch.multinomial(probs, 1, generator=gen).squeeze(1).tolist()
            keys = _key(curs, masks)
            for b, k in zip(active, keys):
                if min_ply <= b.moves_played <= max_ply and k not in seen:
                    seen.add(k)
                    out.append(b.to_move_string())
            for b, c in zip(active, cols):
                b.drop(c)
            active = [b for b in active if not b.is_terminal()]
        log("  self-play round %d: %d unique positions" % (rounds, len(out)))
    rng.shuffle(out)
    return out[:n_target]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="trained run (runs/<run>/checkpoints/best.pt + best.npz)")
    ap.add_argument("--positions", type=int, default=600_000, help="candidate positions to label")
    ap.add_argument("--games", type=int, default=4096, help="games played at once")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--min-ply", type=int, default=6)
    ap.add_argument("--max-ply", type=int, default=34)
    ap.add_argument("--keep-right", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from engine.board import Board
    from .bench import load_accuracy_bench
    from .encode import encode, legal_mask
    from .make_dataset import _label_worker
    from .model import load_checkpoint
    from .train_score import choices, head_logits

    t0 = time.time()
    device = pick_device(a.device)
    ck = os.path.join(RUNS, a.run, "checkpoints", "best.pt")
    model, _ = load_checkpoint(ck, device)
    meta = json.loads(str(np.load(os.path.join(RUNS, a.run, "best.npz"))["meta_json"]))
    head = meta.get("play_head", "policy")
    if head == "combo":
        head = "combo%g" % (meta["play_policy_weight"] * meta["play_temp"])
    log("mining with %s (head %s) on %s" % (a.run, head, device))

    bench = load_accuracy_bench()
    banned = set(_key([r["cur"] for r in bench], [r["mask"] for r in bench]))
    moves = selfplay_positions(model, device, a.positions, a.games, a.temp, a.eps,
                               a.min_ply, a.max_ply, a.seed, banned)

    log("labelling %d positions with the solver on %d workers ..." % (len(moves), a.workers))
    chunks = [moves[i:i + 500] for i in range(0, len(moves), 500)]
    labelled, skipped = [], 0
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(_label_worker, [(c, a.timeout) for c in chunks])):
            for mv, sc in res:
                if sc is None:
                    skipped += 1
                else:
                    labelled.append((mv, sc))
            if (k + 1) % 50 == 0 or k + 1 == len(chunks):
                log("  labelled %d / %d (skipped %d)  %.0fs" % (len(labelled), len(moves), skipped, time.time() - t0))

    n = len(labelled)
    cur = np.zeros(n, np.uint64)
    mask = np.zeros(n, np.uint64)
    scores = np.full((n, 7), ILLEGAL, np.int8)
    ply = np.zeros(n, np.uint8)
    for i, (mv, sc) in enumerate(labelled):
        b = Board.from_moves(mv)
        cur[i], mask[i], ply[i] = b.current_bits(), b.mask, b.moves_played
        for c, v in enumerate(sc):
            if v is not None:
                scores[i, c] = v

    # which of them does the playing head get wrong?
    wrong = np.zeros(n, bool)
    legal_np = scores != ILLEGAL
    best = np.where(legal_np, scores.astype(np.int16), -1000).max(1)
    for i in range(0, n, 8192):
        sl = slice(i, i + 8192)
        x = torch.from_numpy(encode(cur[sl], mask[sl])).to(device)
        legal = torch.from_numpy(legal_mask(mask[sl])).to(device)
        with torch.inference_mode():
            lp, s = head_logits(model, x, legal, mirror=bool(meta.get("play_mirror", True)))
            c = choices(lp, s, legal)[head].cpu().numpy()
        wrong[sl] = scores[sl][np.arange(len(c)), c].astype(np.int16) != best[sl]
    rng = np.random.default_rng(a.seed)
    keep = wrong | (rng.random(n) < a.keep_right)
    by_ply = {}
    for p in range(a.min_ply, a.max_ply + 1):
        m = ply == p
        if m.any():
            by_ply[p] = round(float(1 - wrong[m].mean()), 4)
    info = {"run": a.run, "head": head, "candidates": n, "skipped_timeout": skipped,
            "wrong": int(wrong.sum()), "on_policy_optimal": round(float(1 - wrong.mean()), 4),
            "on_policy_optimal_by_ply": by_ply, "kept": int(keep.sum()), "keep_right": a.keep_right,
            "temp": a.temp, "eps": a.eps, "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, cur=cur[keep], mask=mask[keep], scores=scores[keep], ply=ply[keep],
                        meta_json=np.array(json.dumps(info)))
    with open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    log("on-policy optimal %.4f (%d wrong of %d); kept %d -> %s" % (
        info["on_policy_optimal"], info["wrong"], n, info["kept"], a.out))


if __name__ == "__main__":
    main()
