"""Knowledge distillation: train a small, fast "student" network to copy one or
more trained "teachers".

    # copy AlphaZero + its search into a tiny 2x32 net
    python -m rl.distill --run distill_az_2x32 --teacher nn:runs/az_6x64/model.npz:mcts:100

    # ensemble teacher: AlphaZero+MCTS, PPO and the TD n-tuple player together
    python -m rl.distill --run distill_ens_2x32 \
        --teacher nn:runs/az_6x64/model.npz:mcts:100 \
        --teacher nn:runs/ppo_6x64/model.npz \
        --teacher ntuple:runs/td_ntuple/ntuple.npz \
        --teacher-weights 2,1,1

    # reuse the labels with another student size (no teacher work at all)
    python -m rl.distill --run distill_az_4x48 --labels runs/distill_az_2x32/checkpoints/labels.npz --blocks 4 --channels 48

Teacher specs
  nn:PATH                 a network (.npz or .pt): its policy and WDL value
  nn:PATH:mcts:SIMS       the same network PLUS MCTS: the policy target is the
                          search's visit distribution - the student learns to
                          predict, in one forward pass, what search finds
  ntuple:PATH             the TD n-tuple player: policy = softmax(--ntuple-beta
                          x afterstate values), value mapped to W/D/L
  With several teachers the targets are averaged (--teacher-weights).

Steps
  1. POSITIONS  --positions positions from games played by the first network
     teacher (policy sampled with --temperature, plus --random-move random
     moves for variety); the 3,000 benchmark positions are excluded.
     (--data nn_data/<file>.npz uses the solver dataset's positions instead.)
  2. LABELS     every teacher labels every position (soft targets) -> saved to
     checkpoints/labels.npz so other student sizes can reuse them.
  3. TRAIN      student loss = KL(teacher policy || student policy) +
     value_weight x cross-entropy(teacher WDL, student WDL), mirror
     augmentation, AdamW + one-cycle LR, like nn/train_supervised.py.
  Logged per epoch: accuracy vs perfect play, agreement with the teacher's top
  move, and inference speed (student vs teacher, numpy, single position).

Trade-offs: very cheap to train (no games against itself, labels come from
the teacher) and the student is many times smaller/faster - great for the
website - but it can at best match its teacher (it learns "what", not "why"),
and a teacher with search makes labelling the expensive step.
"""

import argparse
import math
import os
import time

import numpy as np

from nn.common import Run, log, machine_info, now_iso, pick_device

from . import vecenv as V
from .meter import CostMeter


# ================================================================ teachers
class Teacher:
    def __init__(self, spec, device):
        parts = spec.split(":")
        self.spec = spec
        self.kind = parts[0]
        self.path = parts[1]
        self.sims = 0
        if self.kind == "nn":
            from .torchutil import load_gpu_evaluator
            self.ev = load_gpu_evaluator(self.path, device)          # GPU if available (labelling)
            from nn.common import load_evaluator
            self.ev_cpu = load_evaluator(self.path, "cpu") if self.path.endswith(".npz") else self.ev
            if len(parts) >= 4 and parts[2] == "mcts":
                self.sims = int(parts[3])
        elif self.kind == "ntuple":
            from .ntuple import NTupleNet
            self.net = NTupleNet.load(self.path)
        else:
            raise SystemExit("teacher spec must start with nn: or ntuple: (got %r)" % spec)

    def label(self, cur, mask, beta=8.0, batch=4096):
        """-> policy (N,7), wdl (N,3)."""
        n = len(cur)
        pol = np.zeros((n, 7), np.float32)
        wdl = np.zeros((n, 3), np.float32)
        if self.kind == "nn":
            for i in range(0, n, batch):
                p, _, w = self.ev.evaluate(cur[i:i + batch], mask[i:i + batch])
                pol[i:i + batch], wdl[i:i + batch] = p, w
            if self.sims:
                from nn import mcts
                for i in range(0, n, 512):
                    roots = [mcts.Node(int(c), int(m)) for c, m in zip(cur[i:i + 512], mask[i:i + 512])]
                    mcts.search(roots, self.ev, self.sims)
                    for k, r in enumerate(roots):
                        pol[i + k] = mcts.visit_policy(r)
                    if (i // 512) % 20 == 0:
                        log("  %s: %d/%d positions searched" % (self.spec, i + len(roots), n))
        else:
            for i in range(0, n, batch):
                vals, lg = self.net.move_values(cur[i:i + batch], mask[i:i + batch])
                z = np.where(lg, beta * vals, -np.inf)
                z = z - z.max(1, keepdims=True)
                e = np.exp(z)
                pol[i:i + batch] = e / e.sum(1, keepdims=True)
                v = np.where(lg, vals, -np.inf).max(1)
                wdl[i:i + batch] = np.stack([np.maximum(v, 0), 1 - np.abs(v), np.maximum(-v, 0)], 1)
        return pol, wdl

    def move_ms(self, cur, mask):
        """Average milliseconds to choose a move for one position (numpy, as on the website)."""
        t0 = time.perf_counter()
        k = min(50, len(cur))
        if self.kind == "nn" and self.sims:
            from nn import mcts
            k = min(10, k)
            for i in range(k):
                r = mcts.Node(int(cur[i]), int(mask[i]))
                mcts.search([r], self.ev_cpu, self.sims)
        elif self.kind == "nn":
            for i in range(k):
                self.ev_cpu.evaluate(cur[i:i + 1], mask[i:i + 1])
        else:
            for i in range(k):
                self.net.move_values(cur[i:i + 1], mask[i:i + 1])
        return (time.perf_counter() - t0) * 1000 / k


# ============================================================== positions
def gen_positions(teacher, n, temperature, random_move, seed, exclude):
    """Positions visited in games played by the teacher's (raw) policy."""
    rng = np.random.default_rng(seed)
    B = 512
    g = V.VecGames(B, rng, random_opening=(0, 2))
    seen = set()
    cur_out, mask_out = [], []
    while len(cur_out) < n:
        lg = g.legal()
        if teacher is not None and teacher.kind == "nn":
            pr, _, _ = teacher.ev.evaluate(g.cur, g.mask)
            pr = np.where(lg, np.power(np.maximum(pr, 1e-12), 1.0 / temperature), 0)
        elif teacher is not None:
            vals, _ = teacher.net.move_values(g.cur, g.mask)
            z = np.where(lg, 8.0 * vals / temperature, -np.inf)
            pr = np.exp(z - z.max(1, keepdims=True))
        else:
            pr = lg.astype(float)
        cols = V.masked_sample(pr, lg, rng)
        rnd = rng.random(B) < random_move
        if rnd.any():
            cols[rnd] = V.random_legal(lg[rnd], rng)
        for c, m in zip(g.cur, g.mask):
            k = (int(c), int(m))
            if k not in seen and k not in exclude:
                seen.add(k)
                cur_out.append(k[0])
                mask_out.append(k[1])
        won, full = g.step(cols)
        g.reset(np.nonzero(won | full)[0])
    return np.array(cur_out[:n], dtype=np.uint64), np.array(mask_out[:n], dtype=np.uint64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--teacher", action="append", default=[], help="teacher spec (repeatable), see above")
    ap.add_argument("--teacher-weights", default=None, help="comma list, one weight per --teacher")
    ap.add_argument("--labels", default=None, help="reuse a labels.npz from an earlier distill run")
    ap.add_argument("--data", default=None, help="take positions from this nn_data/*.npz instead of teacher games")
    ap.add_argument("--positions", type=int, default=300_000)
    ap.add_argument("--temperature", type=float, default=1.0, help="sampling temperature for position games")
    ap.add_argument("--random-move", type=float, default=0.1, help="share of random moves in position games")
    ap.add_argument("--policy-temp", type=float, default=1.0, help="soften teacher policies: p^(1/T)")
    ap.add_argument("--ntuple-beta", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from nn import bench
    from nn.common import TorchEvaluator
    from nn.export import export_model
    from nn.model import build, count_params, save_checkpoint
    from nn.numpy_net import NumpyNet
    from .torchutil import amp_ctx, masked_log_softmax

    if not a.teacher and not a.labels:
        raise SystemExit("give at least one --teacher (or --labels from an earlier run)")
    run = Run(a.run)
    if run.exists():
        raise SystemExit("runs/%s exists - choose another --run" % a.run)
    device = pick_device(a.device)
    torch.manual_seed(a.seed)
    meter = CostMeter(uses_gpu=device.type == "cuda", label="rl.distill " + a.run).start()
    rows = bench.load_accuracy_bench()
    t_run = time.time()
    teachers = [Teacher(s, a.device) for s in a.teacher]
    tw = [float(x) for x in a.teacher_weights.split(",")] if a.teacher_weights else [1.0] * len(teachers)
    if len(tw) != len(teachers):
        raise SystemExit("--teacher-weights needs one number per --teacher")

    # ------------------------------------------------------------ 1+2 labels
    t0 = time.time()
    lab_path = os.path.join(run.ckpt_dir, "labels.npz")
    if a.labels:
        z = np.load(a.labels)
        cur, mask, pol, wdl = z["cur"], z["mask"], z["pol"], z["wdl"]
        label_sec = 0.0
        log("reusing %d labelled positions from %s" % (len(cur), a.labels))
    else:
        exclude = set()
        for r in rows:                                   # never train on the benchmark (or its mirror)
            exclude.add((r["cur"], r["mask"]))
            exclude.add((int(V.mirror(np.array([r["cur"]], np.uint64))[0]), int(V.mirror(np.array([r["mask"]], np.uint64))[0])))
        if a.data:
            zz = np.load(a.data)
            keep = [i for i, (c, m) in enumerate(zip(zz["cur"], zz["mask"])) if (int(c), int(m)) not in exclude]
            keep = keep[:a.positions]
            cur, mask = zz["cur"][keep].astype(np.uint64), zz["mask"][keep].astype(np.uint64)
        else:
            first_nn = next((t for t in teachers if t.kind == "nn"), teachers[0])
            log("generating %d positions from games of %s ..." % (a.positions, first_nn.spec))
            cur, mask = gen_positions(first_nn, a.positions, a.temperature, a.random_move, a.seed, exclude)
        pol = np.zeros((len(cur), 7), np.float32)
        wdl = np.zeros((len(cur), 3), np.float32)
        for t, w in zip(teachers, tw):
            t1 = time.time()
            log("labelling with %s ..." % t.spec)
            p, v = t.label(cur, mask, a.ntuple_beta)
            if a.policy_temp != 1.0:
                p = np.power(np.maximum(p, 1e-12), 1.0 / a.policy_temp) * (p > 0)
                p = p / p.sum(1, keepdims=True)
            pol += w * p
            wdl += w * v
            log("  done in %.0fs" % (time.time() - t1))
        pol /= sum(tw)
        wdl /= sum(tw)
        np.savez(lab_path, cur=cur, mask=mask, pol=pol, wdl=wdl,
                 teachers=np.array(a.teacher), teacher_weights=np.array(tw))
        label_sec = time.time() - t0
    teacher_ms = {t.spec: round(t.move_ms(cur[:200], mask[:200]), 3) for t in teachers}

    # ------------------------------------------------------------- 3 train
    student = build({"blocks": a.blocks, "channels": a.channels}).to(device)
    n = len(cur)
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(n)
    nv = max(1, int(n * a.val_frac))
    va, tr = perm[:nv], perm[nv:]
    run.write_config({"phase": "distill", "created": now_iso(), "args": vars(a), "device": str(device),
                      "model": student.config, "params": count_params(student), "teachers": a.teacher,
                      "teacher_weights": tw, "positions": int(n), "label_sec": round(label_sec, 1),
                      "teacher_ms": teacher_ms, "machine": machine_info()})
    log("student %dx%d: %d params; %d positions (%d val); labels took %.0fs" % (
        a.blocks, a.channels, count_params(student), n, nv, label_sec))

    from nn.encode import encode, legal_mask
    X = torch.from_numpy(encode(cur, mask)).to(device)
    LG = torch.from_numpy(legal_mask(mask)).to(device)
    P = torch.from_numpy(pol).to(device)
    W = torch.from_numpy(wdl).to(device)
    tr_t, va_t = torch.from_numpy(tr).to(device), torch.from_numpy(va).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps = math.ceil(len(tr) / a.batch) * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    best_val = float("inf")

    def losses(idx, mirror):
        x, lg, p, w = X[idx], LG[idx], P[idx], W[idx]
        if mirror is not None:
            x = torch.where(mirror.view(-1, 1, 1, 1), x.flip(3), x)
            lg = torch.where(mirror.view(-1, 1), lg.flip(1), lg)
            p = torch.where(mirror.view(-1, 1), p.flip(1), p)
        with amp_ctx(device):
            pl, vl = student(x)
        logp = masked_log_softmax(pl, lg)
        lp = -(p * logp).sum(1).mean() + (p * torch.log(p.clamp_min(1e-12))).sum(1).mean()   # KL
        lv = -(w * F.log_softmax(vl.float(), 1)).sum(1).mean()
        agree = (logp.argmax(1) == p.argmax(1)).float().mean()
        return lp, lv, agree

    for epoch in range(1, a.epochs + 1):
        t1 = time.time()
        student.train()
        order = tr_t[torch.randperm(len(tr_t), device=device)]
        s = np.zeros(3)
        for i in range(0, len(order), a.batch):
            idx = order[i:i + a.batch]
            lp, lv, _ = losses(idx, torch.rand(len(idx), device=device) < 0.5)
            loss = lp + a.value_weight * lv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            s += [loss.item() * len(idx), lp.item() * len(idx), lv.item() * len(idx)]
        s /= len(order)
        student.eval()
        with torch.inference_mode():
            vs = np.zeros(3)
            for i in range(0, len(va_t), 4096):
                idx = va_t[i:i + 4096]
                lp, lv, ag = losses(idx, None)
                vs += [lp.item() * len(idx), lv.item() * len(idx), ag.item() * len(idx)]
            vs /= len(va_t)
        t_train = time.time() - t1
        save_checkpoint(os.path.join(run.ckpt_dir, "latest.pt"), student, {"epoch": epoch})
        meta = {"run": a.run, "phase": "distill", "epoch": epoch, "teachers": a.teacher, "exported": now_iso()}
        export_model(student, run.path("model.npz"), meta)
        val_loss = vs[0] + a.value_weight * vs[1]
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(os.path.join(run.ckpt_dir, "best.pt"), student, {"epoch": epoch})
            export_model(student, run.path("best.npz"), dict(meta, best=True))
        pb = bench.policy_eval(TorchEvaluator(student, device), rows)
        sn = NumpyNet(run.path("model.npz"))
        t2 = time.perf_counter()
        for i in range(100):
            sn.evaluate(cur[i:i + 1], mask[i:i + 1])
        student_ms = (time.perf_counter() - t2) * 10
        rec = {"type": "epoch", "phase": "distill", "epoch": epoch, "gen": epoch, "positions_seen": epoch * len(tr),
               "train_loss": round(float(s[0]), 5), "policy_loss": round(float(s[1]), 5),
               "value_loss": round(float(s[2]), 5), "val_policy_kl": round(float(vs[0]), 5),
               "val_value_loss": round(float(vs[1]), 5), "val_teacher_agreement": round(float(vs[2]), 4),
               "bench_optimal": pb["optimal"], "bench_keeps_nl": pb["keeps_nl"], "bench_value_acc": pb["value_acc"],
               "bench_by_bucket": pb["by_bucket"], "student_ms": round(student_ms, 3), "teacher_ms": teacher_ms,
               "label_sec": round(label_sec, 1), "train_sec": round(t_train, 1),
               "elapsed_sec": round(time.time() - t_run, 1)}
        rec.update(meter.snapshot())
        run.log_metrics(rec)
        meter.write(run.dir)
        log("epoch %d/%d  loss %.4f (KL %.4f v %.4f)  val KL %.4f  teacher agreement %.3f  bench optimal %.3f value %.3f  "
            "student %.2f ms vs teacher %s ms  [%.0fs]" % (
                epoch, a.epochs, s[0], s[1], s[2], vs[0], vs[2], pb["optimal"], pb["value_acc"], student_ms,
                "/".join("%.1f" % v for v in teacher_ms.values()), t_train))
    meter.stop()
    meter.write(run.dir)
    log("done. Next: python -m nn.evaluate --run %s --model best.npz   and   python -m rl.compare" % a.run)


if __name__ == "__main__":
    main()
