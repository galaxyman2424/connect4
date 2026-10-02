#!/usr/bin/env python3
"""Download the public datasets / opening book for the Connect 4 project.

Run from the project root (works on Windows, macOS and Linux, Python 3.10+):

    pip install -r data/requirements-data.txt
    python data/download_data.py --all
    python data/download_data.py --uci
    python data/download_data.py --tonycwang 5000
    python data/download_data.py --leon
    python data/download_data.py --book
    (add --force to redo a step whose output already exists)

Each step is independent: a failure prints a short message and the other steps
still run.  Exit status is 1 if any requested step failed.
"""
import argparse
import csv
import gzip
import io
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
UCI_URL = "https://archive.ics.uci.edu/static/public/26/connect+4.zip"
UCI_ROWS = 67557
BOOK_URLS = ["http://blog.gamesolver.org/data/7x6.book"]
BOOK_RELEASES_API = "https://api.github.com/repos/PascalPons/connect4/releases"
LEON_REPO = "Leon-LLM/Connect-Four-Datasets-Collection"
TONY_REPO = "TonyCWang/ConnectFour"
LEON_MAX_BYTES = 50 * 1024 * 1024
UA = {"User-Agent": "Mozilla/5.0 (connect4-course-project downloader)"}


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------- helpers
def download(url, dest, timeout=60):
    """Download url to dest (via dest.part) with a simple progress line."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers=UA)
    log("  GET %s" % url)
    with urllib.request.urlopen(req, timeout=timeout) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        last = -1
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            pct = int(100 * done / total) if total else done // (1 << 20)
            if pct != last:
                last = pct
                unit = "%" if total else " MB"
                sys.stdout.write("\r  %d%s (%.1f MB)   " % (pct, unit, done / 1e6))
                sys.stdout.flush()
    sys.stdout.write("\n")
    os.replace(part, dest)
    return dest


def short_err(e):
    return "%s: %s" % (type(e).__name__, e)


# ------------------------------------------------------------------- UCI
def decode_blob(data, name=""):
    """Return plain bytes from zip / gzip / .Z (LZW) / plain data."""
    if data[:2] == b"\x1f\x8b":
        return decode_blob(gzip.decompress(data), name)
    if data[:2] == b"\x1f\x9d":
        try:
            import unlzw3
        except ImportError:
            raise RuntimeError("the UCI file is compressed with Unix 'compress' (.Z). "
                               "Install the decoder:  pip install unlzw3   and re-run.")
        return decode_blob(unlzw3.unlzw(data), name)
    if data[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            return decode_blob(z.read(pick_member(z.namelist())), name)
    return data


def pick_member(names):
    cands = [n for n in names if "connect-4" in n.lower() and ".names" not in n.lower()
             and not n.endswith("/")]
    cands.sort(key=lambda n: (".data" not in n.lower(), len(n)))
    if not cands:
        raise RuntimeError("no connect-4 data file inside archive; members: %s" % names)
    return cands[0]


def uci_line_to_row(line):
    """'b,b,...,x,o,win' -> (cells[42] as x/o/b, label, grid 6x7 row0=top, +1=x)."""
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) != 43:
        raise ValueError("expected 43 fields, got %d" % len(parts))
    cells, label = parts[:42], parts[42]
    val = {"x": 1, "o": -1, "b": 0}
    if any(c not in val for c in cells) or label not in ("win", "loss", "draw"):
        raise ValueError("bad cell/label in line %r" % line[:60])
    grid = [[0] * 7 for _ in range(6)]
    for col in range(7):          # a..g
        for h in range(6):        # 1..6, bottom-up
            grid[5 - h][col] = val[cells[col * 6 + h]]
    return cells, label, grid


def convert_uci(data_path, csv_path):
    n = 0
    header = ["c%d" % i for i in range(42)] + ["label", "grid"]
    with open(data_path, "r", encoding="ascii", errors="replace") as f, \
            open(csv_path, "w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow(header)
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                cells, label, grid = uci_line_to_row(line)
            except ValueError as e:
                raise RuntimeError("line %d: %s" % (i, e))
            w.writerow(cells + [label, json.dumps(grid, separators=(",", ":"))])
            n += 1
    return n


def step_uci(force):
    d = DATA / "uci"
    d.mkdir(parents=True, exist_ok=True)
    raw, csv_path = d / "connect-4.data", d / "uci_positions.csv"
    if csv_path.exists() and not force:
        log("[uci] %s exists, skipping (use --force)" % csv_path)
        return True
    if not raw.exists() or force:
        log("[uci] downloading UCI Connect-4")
        z = download(UCI_URL, d / "connect-4.zip")
        try:
            blob = decode_blob(z.read_bytes(), z.name)
        finally:
            pass
        raw.write_bytes(blob)
        z.unlink()
        log("[uci] wrote %s (%d bytes)" % (raw, len(blob)))
    n = convert_uci(raw, csv_path)
    log("[uci] wrote %s with %d rows" % (csv_path, n))
    if n != UCI_ROWS:
        log("[uci] ERROR: expected %d rows, got %d" % (UCI_ROWS, n))
        return False
    log("[uci] row count OK (%d)" % n)
    return True


# ------------------------------------------------------------ TonyCWang
def _num(x):
    try:
        if x is None:
            return None
        v = float(x)
        return None if v != v else int(v) if v == int(v) else v
    except (TypeError, ValueError):
        return None


def _to_cells(v):
    """Try to interpret v as 42 cells; return flat list of ints or None."""
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in "[{":
            try:
                v = json.loads(s)
            except ValueError:
                return None
        else:
            toks = [t for t in s.replace(",", " ").split()] if (" " in s or "," in s) else list(s)
            v = toks
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        flat = []
        for e in v:
            if isinstance(e, (list, tuple)):
                flat.extend(e)
            else:
                flat.append(e)
        if len(flat) == 42:
            return flat
    return None


def _norm_cells(flat):
    """Map cell tokens to 0/+1/-1; returns None if unrecognised."""
    m = {"x": 1, "X": 1, "o": -1, "O": -1, "b": 0, ".": 0, "_": 0, " ": 0,
         "1": 1, "2": -1, "0": 0, "-1": -1, "+1": 1}
    out = []
    for c in flat:
        k = str(int(c)) if isinstance(c, float) and c == int(c) else str(c)
        if k not in m:
            return None
        out.append(m[k])
    return out


def map_tonycwang_row(row):
    """Best-effort map of an unknown schema -> dict(grid, scores, optimal_cols).
    Returns None when the schema can't be recognised."""
    grid = scores = None
    for k, v in row.items():
        if grid is None:
            cells = _to_cells(v)
            if cells is not None:
                n = _norm_cells(cells)
                if n is not None:
                    grid = [n[r * 7:(r + 1) * 7] for r in range(6)]
                    continue
        if scores is None:
            cand = v
            if isinstance(cand, str) and cand.strip()[:1] == "[":
                try:
                    cand = json.loads(cand)
                except ValueError:
                    pass
            if hasattr(cand, "tolist"):
                cand = cand.tolist()
            if isinstance(cand, (list, tuple)) and len(cand) == 7 and \
                    not any(isinstance(e, (list, tuple)) for e in cand):
                scores = [_num(e) for e in cand]
    if scores is None:  # 7 separate scalar keys?
        for pref in (["score_%d" % i for i in range(7)], ["col_%d" % i for i in range(7)],
                     ["c%d" % i for i in range(7)], ["%d" % i for i in range(7)],
                     ["score_%d" % i for i in range(1, 8)], ["col_%d" % i for i in range(1, 8)]):
            if all(p in row for p in pref):
                scores = [_num(row[p]) for p in pref]
                break
    if grid is None or scores is None:
        return None
    vals = [s for s in scores if s is not None]
    best = max(vals) if vals else None
    opt = [i for i, s in enumerate(scores) if s is not None and s == best]
    return {"grid": json.dumps(grid, separators=(",", ":")),
            "scores": json.dumps(scores), "optimal_cols": json.dumps(opt)}


def step_tonycwang(n, force):
    out_csv, out_jsonl = DATA / "tonycwang_sample.csv", DATA / "tonycwang_sample_raw.jsonl"
    if (out_csv.exists() or out_jsonl.exists()) and not force:
        log("[tonycwang] output exists, skipping (use --force)")
        return True
    try:
        from datasets import load_dataset
    except ImportError:
        log("[tonycwang] ERROR: 'datasets' not installed.  pip install -r data/requirements-data.txt")
        return False
    log("[tonycwang] streaming %d rows from %s (shuffle buffer for spread)" % (n, TONY_REPO))
    ds = load_dataset(TONY_REPO, split="train", streaming=True)
    try:
        ds = ds.shuffle(seed=12345, buffer_size=min(max(10 * n, 10000), 100000))
    except Exception as e:
        log("[tonycwang] shuffle unavailable (%s); taking first rows" % short_err(e))
    rows = []
    for i, row in enumerate(ds):
        if i == 0:
            log("[tonycwang] first row keys: %s" % list(row.keys()))
            log("[tonycwang] first row (truncated): %s" % str(row)[:400])
        rows.append(row)
        if len(rows) % 500 == 0:
            log("  %d / %d" % (len(rows), n))
        if len(rows) >= n:
            break
    if not rows:
        log("[tonycwang] ERROR: no rows received")
        return False
    mapped = [map_tonycwang_row(r) for r in rows[:50]]
    if all(m is not None for m in mapped):
        conv = [map_tonycwang_row(r) for r in rows]
        conv = [c for c in conv if c is not None]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["grid", "scores", "optimal_cols"])
            w.writeheader()
            w.writerows(conv)
        log("[tonycwang] wrote %s (%d rows)" % (out_csv, len(conv)))
        log("[tonycwang] NOTE: grid assumed row0=top (+1/-1/0, or x/o/b, or 1/2 mapped to +1/-1) as read "
            "from the dataset; player-to-move and orientation are NOT verified - check against the "
            "keys printed above before trusting.")
    else:
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        log("[tonycwang] Could not map schema automatically. Saved RAW rows to %s." % out_jsonl)
        log("[tonycwang] Send the printed keys / first row back so the converter can be fixed.")
    return True


# ------------------------------------------------------------------ Leon
def step_leon(force):
    out = DATA / "leon"
    if out.exists() and any(out.rglob("*")) and not force:
        log("[leon] %s exists and is non-empty, skipping (use --force)" % out)
        return True
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        log("[leon] ERROR: 'huggingface_hub' not installed.  pip install -r data/requirements-data.txt")
        return False
    api = HfApi()
    info = api.dataset_info(LEON_REPO, files_metadata=True)
    files = [(s.rfilename, s.size or 0) for s in info.siblings]
    log("[leon] %d files in repo:" % len(files))
    for name, size in files[:40]:
        log("   %10.1f KB  %s" % (size / 1024, name))
    if len(files) > 40:
        log("   ... (%d more)" % (len(files) - 40))
    folders = {}
    for name, size in files:
        if "/" in name and not name.endswith((".md", ".gitattributes")):
            folders.setdefault(name.split("/")[0], []).append((name, size))
    if not folders:
        folders = {"": [f for f in files if not f[0].startswith(".") and not f[0].endswith(".md")]}
    out.mkdir(parents=True, exist_ok=True)
    ok = False
    for folder, fl in sorted(folders.items()):
        fl = [f for f in fl if 0 < f[1] <= LEON_MAX_BYTES] or [f for f in fl if f[1] == 0]
        if not fl:
            log("[leon] folder %r: every file exceeds the %d MB cap; skipped" % (folder, LEON_MAX_BYTES >> 20))
            continue
        name, size = min(fl, key=lambda f: f[1])
        log("[leon] downloading %s (%.1f MB)" % (name, size / 1e6))
        p = hf_hub_download(LEON_REPO, name, repo_type="dataset", local_dir=str(out))
        log("[leon] saved %s" % p)
        ok = True
    return ok


# ------------------------------------------------------------------ Book
def _looks_like_book(p):
    p = Path(p)
    if not p.exists() or p.stat().st_size < 10_000:
        return False
    return p.read_bytes()[:64].lstrip()[:1] not in (b"<",)  # not an HTML error page


def step_book(force):
    dest = ROOT / "tools" / "pons" / "7x6.book"
    if dest.exists() and not force:
        log("[book] %s exists, skipping (use --force)" % dest)
        return True
    urls = list(BOOK_URLS)
    try:
        req = urllib.request.Request(BOOK_RELEASES_API, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            for rel in json.load(r):
                for a in rel.get("assets", []):
                    if a.get("name", "").endswith(".book"):
                        urls.insert(0, a["browser_download_url"])
    except Exception as e:
        log("[book] no GitHub release assets found (%s)" % short_err(e))
    for u in urls:
        try:
            download(u, dest)
            if _looks_like_book(dest):
                log("[book] saved %s (%d bytes)" % (dest, dest.stat().st_size))
                return True
            log("[book] %s did not look like a book file; discarding" % u)
            dest.unlink()
        except Exception as e:
            log("[book] failed %s -> %s" % (u, short_err(e)))
    log("[book] ERROR: could not fetch the opening book. Manual: open "
        "http://blog.gamesolver.org/ (Pascal Pons' blog, 'opening book' post), download 7x6.book "
        "and save it as tools/pons/7x6.book. It is optional (speeds up the solver only).")
    return False


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uci", action="store_true", help="UCI Connect-4 (67,557 rows)")
    ap.add_argument("--tonycwang", nargs="?", const=5000, type=int, metavar="N", default=None,
                    help="stream N rows (default 5000) of TonyCWang/ConnectFour")
    ap.add_argument("--leon", action="store_true", help="a small part of Leon-LLM collection")
    ap.add_argument("--book", action="store_true", help="Pascal Pons 7x6.book -> tools/pons/")
    ap.add_argument("--all", action="store_true", help="everything above")
    ap.add_argument("--force", action="store_true", help="redo steps whose output exists")
    a = ap.parse_args(argv)
    steps = []
    if a.uci or a.all:
        steps.append(("uci", lambda: step_uci(a.force)))
    if a.tonycwang is not None or a.all:
        n = a.tonycwang if a.tonycwang is not None else 5000
        steps.append(("tonycwang", lambda: step_tonycwang(n, a.force)))
    if a.leon or a.all:
        steps.append(("leon", lambda: step_leon(a.force)))
    if a.book or a.all:
        steps.append(("book", lambda: step_book(a.force)))
    if not steps:
        ap.print_help()
        return 2
    failed = []
    for name, fn in steps:
        try:
            if not fn():
                failed.append(name)
        except Exception as e:
            log("[%s] ERROR: %s" % (name, short_err(e)))
            failed.append(name)
    log("\nDone. %s" % ("Failed steps: " + ", ".join(failed) if failed else "All steps OK."))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
