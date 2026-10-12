"""Training cost meter: wall clock, CPU core-hours, GPU energy, total energy.

Two ways to use it

1. Inside a trainer (all rl/ trainers do this):
       meter = CostMeter(uses_gpu=True).start()
       ...
       run.log_metrics({..., **meter.snapshot()})    # cumulative numbers so far
       meter.write(run.dir)                          # runs/<run>/cost.json

3. For a run trained before this existed (no measurements):
       python -m rl.meter --estimate --run az_6x64 --gpu-watts 100 --cpu-watts 90
   writes an ESTIMATED cost.json = elapsed time (metrics.jsonl) x those watts.
   Get the watts by watching `nvidia-smi` / a wall power meter during a few
   minutes of a --resume'd run, or measure a short metered segment (2.).

2. Around ANY command, e.g. the existing nn/ trainers, so every method is
   measured the same way:
       python -m rl.meter --run az_6x64 -- python -m nn.train_selfplay --run az_6x64 --preset standard
   It runs the command, samples the whole process tree every --interval
   seconds and writes runs/<run>/cost.json (added to what is already there,
   so --resume runs keep accumulating).

What is measured
  wall_sec       elapsed time
  cpu_core_sec   CPU time of the process and all its children (/proc, Linux).
                 1 core busy for 1 hour = 3600.  core-hours = cpu_core_sec / 3600.
  gpu_sec        wall time while the run held the GPU ("GPU-hours" the way a
                 cloud provider bills them)
  gpu_util_avg   average GPU utilisation (nvidia-smi) over those samples
  gpu_wh         GPU energy = integral of nvidia-smi power.draw (watt-hours)
  cpu_wh         CPU package energy from RAPL (/sys/class/powercap) when it is
                 readable (usually needs root); otherwise ESTIMATED as
                 cpu_core_sec x --cpu-watts-per-thread (default 6.5 W: a 105 W
                 Ryzen 7 5800X spread over 16 threads).  cpu_wh_source says which.
  energy_wh      gpu_wh + cpu_wh;  energy_cost_usd = energy_wh / 1000 x price

GPU choice: if CUDA_VISIBLE_DEVICES is a single number (as the runbook
recommends, together with CUDA_DEVICE_ORDER=PCI_BUS_ID) only that GPU is
measured; otherwise every GPU's power is added up (the idle GTX 750 Ti then
adds a few watts).  No nvidia-smi -> GPU numbers stay 0.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time

PRICE_PER_KWH = float(os.environ.get("C4_PRICE_PER_KWH", "0.20"))
CPU_W_PER_THREAD = float(os.environ.get("C4_CPU_WATTS_PER_THREAD", "6.5"))
_TICK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def _gpu_ids():
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if v and "," not in v and v.isdigit():
        return v
    return None


def query_gpus():
    """[(power_w, util_pct, mem_mib)] for the measured GPU(s), or [] without nvidia-smi."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    cmd = [exe, "--query-gpu=power.draw,utilization.gpu,memory.used", "--format=csv,noheader,nounits"]
    ids = _gpu_ids()
    if ids is not None:
        cmd += ["-i", ids]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        try:
            rows.append(tuple(float(p) if p not in ("[N/A]", "N/A", "") else 0.0 for p in parts[:3]))
        except ValueError:
            continue
    return rows


class _Rapl:
    """CPU package energy counter (Linux powercap); None-safe."""

    def __init__(self):
        self.files = []
        base = "/sys/class/powercap"
        try:
            for name in sorted(os.listdir(base)):
                if name.count(":") == 1:                     # top-level package domains only
                    p = os.path.join(base, name, "energy_uj")
                    with open(p) as f:
                        int(f.read())
                    rng = os.path.join(base, name, "max_energy_range_uj")
                    with open(rng) as f:
                        mx = int(f.read())
                    self.files.append((p, mx))
        except (OSError, ValueError):
            self.files = []
        self.last = self._read() if self.files else None
        self.total_uj = 0

    def _read(self):
        out = []
        for p, _ in self.files:
            with open(p) as f:
                out.append(int(f.read()))
        return out

    def update(self):
        if not self.files:
            return
        try:
            cur = self._read()
        except OSError:
            return
        for (p, mx), a, b in zip(self.files, self.last, cur):
            self.total_uj += (b - a) if b >= a else (mx - a + b)
        self.last = cur


def _proc_tree_ticks(root_pid, seen):
    """Update seen[pid] = max utime+stime ticks for root_pid and its descendants."""
    try:
        pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return
    parent, ticks = {}, {}
    for pid in pids:
        try:
            with open("/proc/%d/stat" % pid) as f:
                s = f.read()
        except OSError:
            continue
        rest = s[s.rfind(")") + 2:].split()
        parent[pid] = int(rest[1])
        ticks[pid] = int(rest[11]) + int(rest[12])
    tree, frontier = {root_pid}, [root_pid]
    children = {}
    for pid, pp in parent.items():
        children.setdefault(pp, []).append(pid)
    while frontier:
        p = frontier.pop()
        for c in children.get(p, ()):
            if c not in tree:
                tree.add(c)
                frontier.append(c)
    for pid in tree:
        if pid in ticks:
            seen[pid] = max(seen.get(pid, 0), ticks[pid])


class CostMeter:
    def __init__(self, pid=None, uses_gpu=True, interval=2.0, label=""):
        self.pid = pid or os.getpid()
        self.uses_gpu = uses_gpu
        self.interval = interval
        self.label = label
        self.t0 = None
        self.seen = {}
        self.gpu_j = 0.0
        self.gpu_util_sum = 0.0
        self.gpu_samples = 0
        self.gpu_mem_max = 0.0
        self.rapl = _Rapl()
        self.has_proc = os.path.isdir("/proc/self")
        self._stop = threading.Event()
        self._thread = None
        self._last_t = None
        self._lock = threading.Lock()
        self.timeline = []          # (wall_sec, energy_wh) every ~60 s, for charts

    def start(self):
        self.t0 = time.time()
        self._last_t = self.t0
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        last_tl = 0
        while not self._stop.wait(self.interval):
            self._sample()
            if time.time() - last_tl >= 60:
                s = self.snapshot()
                self.timeline.append((s["wall_sec"], s["energy_wh"]))
                last_tl = time.time()

    def _sample(self):
        now = time.time()
        with self._lock:
            dt = now - self._last_t
            self._last_t = now
            if self.has_proc:
                _proc_tree_ticks(self.pid, self.seen)
            self.rapl.update()
            if self.uses_gpu:
                rows = query_gpus()
                if rows:
                    self.gpu_j += sum(r[0] for r in rows) * dt
                    self.gpu_util_sum += max(r[1] for r in rows)
                    self.gpu_samples += 1
                    self.gpu_mem_max = max(self.gpu_mem_max, sum(r[2] for r in rows))

    def cpu_core_sec(self):
        """Max of (a) the /proc samples of the live process tree and (b) the kernel's
        exact totals for finished, reaped children (catches short-lived workers
        that ended between two samples)."""
        tree = sum(self.seen.values()) / _TICK if (self.has_proc and self.seen) else 0.0
        if self.pid == os.getpid():
            t = os.times()
            exact = t.user + t.system + t.children_user + t.children_system
        else:
            import resource
            r = resource.getrusage(resource.RUSAGE_CHILDREN)      # wrapped command, once it has exited
            exact = r.ru_utime + r.ru_stime
        return max(tree, exact)

    def snapshot(self):
        with self._lock:
            wall = time.time() - self.t0
            cpu = self.cpu_core_sec()
            if self.rapl.files:
                cpu_wh, src = self.rapl.total_uj / 3.6e9, "rapl"
            else:
                cpu_wh, src = cpu * CPU_W_PER_THREAD / 3600.0, "estimate"
            gpu_wh = self.gpu_j / 3600.0
            return {"wall_sec": round(wall, 1), "cpu_core_sec": round(cpu, 1),
                    "gpu_sec": round(wall if self.uses_gpu else 0.0, 1),
                    "gpu_util_avg": round(self.gpu_util_sum / self.gpu_samples, 1) if self.gpu_samples else None,
                    "gpu_mem_max_mib": round(self.gpu_mem_max) if self.gpu_samples else None,
                    "gpu_wh": round(gpu_wh, 2), "cpu_wh": round(cpu_wh, 2), "cpu_wh_source": src,
                    "energy_wh": round(gpu_wh + cpu_wh, 2)}

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._sample()
        return self.snapshot()

    def write(self, run_dir, extra=None):
        """Add this segment to run_dir/cost.json (keeps earlier segments, e.g. before --resume)."""
        if os.environ.get("C4_METER_EXTERNAL") == "1":
            return None                     # rl.meter wraps this process and writes the file itself
        seg = self.snapshot()
        seg["id"] = "%d-%d" % (self.pid, int(self.t0))      # rewriting replaces this session's segment
        return write_cost(run_dir, seg, self.label, self.timeline, extra)


def write_cost(run_dir, seg, label="", timeline=None, extra=None):
    path = os.path.join(run_dir, "cost.json")
    data = {"segments": []}
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {"segments": []}
    seg = dict(seg)
    seg["label"] = label
    seg["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if timeline:
        seg["timeline"] = [[round(a, 1), round(b, 3)] for a, b in timeline]
    segs = data.get("segments", [])
    if seg.get("id"):
        segs = [s for s in segs if s.get("id") != seg["id"]]
    segs.append(seg)
    data["segments"] = segs
    tot = {k: round(sum(s.get(k) or 0 for s in segs), 2)
           for k in ("wall_sec", "cpu_core_sec", "gpu_sec", "gpu_wh", "cpu_wh", "energy_wh")}
    tot["wall_hours"] = round(tot["wall_sec"] / 3600, 3)
    tot["cpu_core_hours"] = round(tot["cpu_core_sec"] / 3600, 3)
    tot["gpu_hours"] = round(tot["gpu_sec"] / 3600, 3)
    tot["energy_kwh"] = round(tot["energy_wh"] / 1000, 4)
    tot["price_per_kwh"] = PRICE_PER_KWH
    tot["energy_cost_usd"] = round(tot["energy_kwh"] * PRICE_PER_KWH, 3)
    tot["cpu_wh_source"] = ",".join(sorted({s.get("cpu_wh_source", "?") for s in segs}))
    data["total"] = tot
    if extra:
        data.update(extra)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    return data


def estimate(run, gpu_watts, cpu_watts, uses_gpu=True, force=False):
    """For runs trained BEFORE rl.meter existed: cost.json from the elapsed time in
    metrics.jsonl x average power (clearly marked as an estimate)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(root, "runs", run)
    p = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.isfile(p):
        raise SystemExit("no %s" % p)
    if os.path.isfile(os.path.join(run_dir, "cost.json")) and not force:
        with open(os.path.join(run_dir, "cost.json"), encoding="utf-8") as f:
            if any(sg.get("id") != "estimate" for sg in json.load(f).get("segments", [])):
                raise SystemExit("runs/%s/cost.json already has measured segments; --force to add an estimate anyway" % run)
    # elapsed_sec restarts at 0 when a run is resumed: add up every session
    el, prev, done = 0.0, 0.0, 0.0
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                e = float(json.loads(line).get("elapsed_sec") or 0)
            except ValueError:
                continue
            if e < prev:
                done += prev
            prev = e
    el = done + prev
    h = el / 3600
    seg = {"id": "estimate", "wall_sec": round(el, 1), "cpu_core_sec": None,
           "gpu_sec": round(el if uses_gpu else 0, 1), "gpu_wh": round(gpu_watts * h if uses_gpu else 0, 2),
           "cpu_wh": round(cpu_watts * h, 2), "cpu_wh_source": "estimate (elapsed x %g W)" % cpu_watts,
           "energy_wh": round((gpu_watts if uses_gpu else 0) * h + cpu_watts * h, 2)}
    write_cost(run_dir, seg, "estimate from metrics.jsonl: %.2f h x (GPU %g W + CPU %g W)" % (h, gpu_watts, cpu_watts),
               extra={"estimated": True})
    print("[meter] %s: %.2f h -> %.0f Wh (estimate) written to runs/%s/cost.json" % (run, h, seg["energy_wh"], run))


def main():
    import argparse
    argv = sys.argv[1:]
    if "--estimate" in argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--estimate", action="store_true")
        ap.add_argument("--run", required=True)
        ap.add_argument("--gpu-watts", type=float, default=100.0, help="average GPU draw while training")
        ap.add_argument("--cpu-watts", type=float, default=90.0, help="average CPU draw while training")
        ap.add_argument("--no-gpu", action="store_true")
        ap.add_argument("--force", action="store_true")
        a = ap.parse_args(argv)
        estimate(a.run, a.gpu_watts, a.cpu_watts, not a.no_gpu, a.force)
        return
    if "--" not in argv:
        raise SystemExit("usage: python -m rl.meter --run NAME [--no-gpu] -- command ...\n"
                         "       python -m rl.meter --estimate --run NAME [--gpu-watts W --cpu-watts W]")
    i = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder runs/<run>/ that gets cost.json")
    ap.add_argument("--no-gpu", action="store_true", help="the command does not use the GPU")
    ap.add_argument("--interval", type=float, default=2.0)
    a = ap.parse_args(argv[:i])
    cmd = argv[i + 1:]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(root, "runs", a.run)
    env = dict(os.environ, C4_METER_EXTERNAL="1")
    proc = subprocess.Popen(cmd, env=env)
    meter = CostMeter(pid=proc.pid, uses_gpu=not a.no_gpu, interval=a.interval, label=" ".join(cmd)).start()
    try:
        rc = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        rc = proc.wait()
    seg = meter.stop()
    os.makedirs(run_dir, exist_ok=True)
    data = write_cost(run_dir, seg, " ".join(cmd), meter.timeline)
    t = data["total"]
    print("[meter] %s: this segment %.2f h wall, %.2f CPU core-h, GPU %.1f Wh, CPU %.1f Wh (%s); "
          "run total %.3f kWh (~$%.2f at $%.2f/kWh) -> %s" % (
              a.run, seg["wall_sec"] / 3600, seg["cpu_core_sec"] / 3600, seg["gpu_wh"], seg["cpu_wh"],
              seg["cpu_wh_source"], t["energy_kwh"], t["energy_cost_usd"], t["price_per_kwh"],
              os.path.join(run_dir, "cost.json")), flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
