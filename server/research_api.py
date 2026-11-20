"""JSON endpoints behind the Research page (/research).

The page only reads files that the research scripts write (research/*.py):

  experiments/results/research/arena_analysis.json   arena re-analysis (research/arena_analysis.py)
  experiments/results/research/roster.json           top-15 roster
  experiments/results/research/sweep.json            supervised size x teacher sweep (research/sweep.py)
  experiments/results/research/interp.json           interpretability (research/interp.py)
  experiments/results/research/interp_channels.json  per-channel mean maps (research/interp.py)

and keeps one file of its own, selection.json: which findings / bots / runs the owner
wants in the paper, plus a free-text note per finding.

GET  /research                    the page
GET  /api/research/index          which data files exist (generated time, size)
GET  /api/research/data/<id>      one data file (id in DATASETS), 404 JSON when missing
GET  /api/research/selection      the saved selection
POST /api/research/selection      replace the sections given ("findings", "roster", "runs")

The paper outline is no longer built on the page: `python -m research.paper_outline`
writes docs/PAPER_OUTLINE.md (gitignored) with build_outline() below.
"""

import datetime
import json
import os
import re
import tempfile

from flask import Blueprint, jsonify, request, send_from_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(ROOT, "web")
# module-level so tests (and a scratch server) can point them somewhere else
RESULTS_DIR = os.environ.get("C4_RESEARCH_DIR") or os.path.join(ROOT, "experiments", "results", "research")
DOCS_DIR = os.path.join(ROOT, "docs")
# the clean top-15 re-run of the arena (arena/tournament.py, same format as part3/summary.json)
ARENA_TOP15 = os.environ.get("C4_ARENA_TOP15") or os.path.join(ROOT, "experiments", "results", "arena", "top15", "summary.json")

DATASETS = {
    "arena_analysis": "Arena analysis (part3 round robin re-analysed)",
    "roster": "Top-15 roster",
    "arena_top15": "Top-15 tournament re-run (arena/top15/summary.json)",
    "sweep": "Supervised sweep (student size x teacher)",
    "interp": "Interpretability (probes, channels, ablations)",
    "interp_channels": "Interpretability: per-channel mean maps",
}
# which data file each paper section's findings come from
SECTIONS = [("arena_analysis", "Arena: which bot is strongest, and why"),
            ("sweep", "Supervised sweep: network size and teacher"),
            ("interp", "Interpretability: what the network computes")]
SELECTION_FILE = "selection.json"
STATIC = ("research.js", "research.css")

MAX_BODY = 512 * 1024
MAX_KEYS = 1000
MAX_NOTE = 5000
KEY_RE = re.compile(r"^[A-Za-z0-9_.:+\-]{1,160}$")

bp = Blueprint("research", __name__)


# ------------------------------------------------------------------ files
def _path(name):
    if name == "arena_top15":
        return ARENA_TOP15
    return os.path.join(RESULTS_DIR, name + ".json")


_CACHE = {}   # path -> ((mtime_ns, size), parsed) so the 30 s polling does not re-parse unchanged files


def _load(name):
    p = _path(name)
    try:
        st = os.stat(p)
        key = (st.st_mtime_ns, st.st_size)
        hit = _CACHE.get(p)
        if hit and hit[0] == key:
            return hit[1]
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    if st.st_size < 20 * 1024 * 1024:
        _CACHE[p] = (key, d)
    return d


def _empty_selection():
    return {"findings": {}, "roster": {}, "runs": {}, "updated": None}


def load_selection():
    sel = _empty_selection()
    try:
        with open(os.path.join(RESULTS_DIR, SELECTION_FILE), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return sel
    if isinstance(data, dict):
        for k in ("findings", "roster", "runs"):
            try:
                sel[k] = _validate_section(k, data.get(k, {}))
            except ValueError:
                pass
        if isinstance(data.get("updated"), str):
            sel["updated"] = data["updated"]
    return sel


def _atomic_write(path, text):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".part", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _validate_section(kind, val):
    if not isinstance(val, dict):
        raise ValueError("%s must be an object" % kind)
    if len(val) > MAX_KEYS:
        raise ValueError("%s has too many entries" % kind)
    out = {}
    for k, v in val.items():
        if not isinstance(k, str) or not KEY_RE.match(k):
            raise ValueError("bad key in %s: %r" % (kind, str(k)[:40]))
        if kind == "findings":
            if not isinstance(v, dict):
                raise ValueError("findings.%s must be an object" % k)
            inc, note = v.get("include", False), v.get("note", "")
            if not isinstance(inc, bool):
                raise ValueError("findings.%s.include must be true/false" % k)
            if not isinstance(note, str) or len(note) > MAX_NOTE:
                raise ValueError("findings.%s.note must be a string of at most %d characters" % (k, MAX_NOTE))
            if set(v) - {"include", "note"}:
                raise ValueError("findings.%s: only 'include' and 'note' are allowed" % k)
            out[k] = {"include": inc, "note": note}
        else:
            if not isinstance(v, bool):
                raise ValueError("%s.%s must be true/false" % (kind, k))
            out[k] = v
    return out


# ------------------------------------------------------------------ routes
@bp.get("/research")
def research_page():
    return send_from_directory(WEB_DIR, "research.html")


@bp.get("/research.js")
def research_js():
    return send_from_directory(WEB_DIR, "research.js")


@bp.get("/research.css")
def research_css():
    return send_from_directory(WEB_DIR, "research.css")


@bp.get("/api/research/index")
def research_index():
    out = {}
    for name, label in DATASETS.items():
        p = _path(name)
        ent = {"id": name, "label": label, "exists": os.path.isfile(p)}
        if ent["exists"]:
            st = os.stat(p)
            ent["bytes"] = st.st_size
            ent["mtime"] = datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
            if name != "interp_channels":       # big file: do not parse it just for the timestamp
                d = _load(name)
                ent["valid"] = isinstance(d, dict)
                if isinstance(d, dict):
                    ent["generated"] = d.get("generated") or (d.get("meta") or {}).get("created")
                    ent["title"] = d.get("title")
                    if isinstance(d.get("findings"), list):
                        ent["n_findings"] = len(d["findings"])
                    if name == "sweep" and isinstance(d.get("status"), dict):
                        ent["status"] = d["status"]
        out[name] = ent
    return jsonify({"datasets": out, "results_dir": os.path.relpath(RESULTS_DIR, ROOT)})


@bp.get("/api/research/data/<name>")
def research_data(name):
    if name not in DATASETS:
        return jsonify({"error": "unknown dataset"}), 404
    p = _path(name)
    if not os.path.isfile(p):
        return jsonify({"error": "not available yet", "id": name}), 404
    # big per-channel file is sent as is (parsing it per request would be slow); the others are checked
    if name != "interp_channels" and _load(name) is None:   # being rewritten right now, or broken
        return jsonify({"error": "file could not be parsed (it may be mid-write; retry)", "id": name}), 503
    resp = send_from_directory(os.path.dirname(p), os.path.basename(p), mimetype="application/json", max_age=0)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@bp.get("/api/research/selection")
def get_selection():
    return jsonify(load_selection())


@bp.post("/api/research/selection")
def post_selection():
    if request.content_length is not None and request.content_length > MAX_BODY:
        return jsonify({"error": "selection too large"}), 413
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "expected a JSON object"}), 400
    extra = set(data) - {"findings", "roster", "runs", "updated"}
    if extra:
        return jsonify({"error": "unknown keys: %s" % ", ".join(sorted(extra))}), 400
    sel = load_selection()
    try:
        for k in ("findings", "roster", "runs"):
            if k in data:
                sel[k] = _validate_section(k, data[k])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    sel["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    text = json.dumps(sel, indent=1, sort_keys=True)
    if len(text) > MAX_BODY:
        return jsonify({"error": "selection too large"}), 413
    _atomic_write(os.path.join(RESULTS_DIR, SELECTION_FILE), text)
    return jsonify(sel)


def write_outline(path=None, include_all=True):
    """Write the Markdown outline (default docs/PAPER_OUTLINE.md); returns (path, bytes)."""
    text = build_outline(include_all=include_all)
    path = path or os.path.join(DOCS_DIR, "PAPER_OUTLINE.md")
    _atomic_write(path, text)
    return path, len(text.encode("utf-8"))


# ------------------------------------------------------------------ outline
def _one_line(s):
    return " ".join(str(s if s is not None else "").split())


def _md_cell(s):
    return _one_line(s).replace("|", "\\|")


def build_outline(include_all=False):
    """include_all=True: every finding, every top-15 bot and every sweep run (notes from
    selection.json are still added); False: only what selection.json marks as included."""
    sel = load_selection()
    data = {name: _load(name) for name, _ in SECTIONS}
    roster = _load("roster")
    fsel = sel["findings"]
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    scope = "all findings, bots and runs" if include_all else "the findings selected in selection.json"
    L = ["# [Paper title]", "",
         "*Draft outline generated by `python -m research.paper_outline` on %s from %s. "
         "Every number below is copied from the results files in `experiments/results/research/`.*" % (now, scope), "",
         "**Author:** [name]  ", "**Course:** [course]", "",
         "## Research questions", "",
         "1. Which Connect 4 bot in our 20-bot arena is strongest, and is our best neural network better than the "
         "open-source AlphaZero-style nets for a reason we can identify (training signal, search, size, time)?",
         "2. How does a supervised student's quality depend on its size (parameters) and on the teacher that labels "
         "its training positions? Can a student exceed its teacher?",
         "3. What does the trained network represent internally: which layer first encodes which game concept, and "
         "which blocks and channels does play depend on?", ""]
    total = 0
    for k, (name, heading) in enumerate(SECTIONS, 1):
        d = data[name]
        L.append("## %d. %s" % (k, heading))
        L.append("")
        if not isinstance(d, dict):
            L += ["*Data not available yet (`%s.json` missing).*" % name, ""]
            continue
        src = [d.get("title"), "generated %s" % d.get("generated") if d.get("generated") else None]
        L += ["Source: `%s.json`%s" % (name, (" (" + ", ".join(s for s in src if s) + ")") if any(src) else ""), ""]
        if name == "sweep" and isinstance(d.get("status"), dict) and d["status"].get("state") != "done":
            c = d["status"].get("counts") or {}
            L += ["*Sweep still in progress: %s of %s runs done.*" % (c.get("done", "?"), d["status"].get("total", "?")), ""]
        found = [f for f in (d.get("findings") or []) if isinstance(f, dict) and f.get("id")
                 and (include_all or fsel.get(f.get("id"), {}).get("include"))]
        if not found:
            L += ["*No findings selected from this section.*", ""]
            continue
        for f in found:
            total += 1
            L.append("### %s" % _one_line(f.get("title") or f.get("id")))
            L.append("")
            L.append("- **Claim:** %s" % _one_line(f.get("claim")))
            if f.get("evidence"):
                L.append("- **Evidence:** %s" % _one_line(f.get("evidence")))
            if f.get("caveat"):
                L.append("- **Caveat:** %s" % _one_line(f.get("caveat")))
            note = (fsel.get(f["id"]) or {}).get("note", "").strip()
            if note:
                L.append("- **My note:** %s" % note.replace("\n", "\n  "))
            L.append("- *Finding id:* `%s`" % f.get("id"))
            L.append("")
    # roster
    L += ["## Bots used (roster)", ""]
    chosen = [b for b, v in sel["roster"].items() if v]
    if include_all and isinstance(roster, dict):
        chosen = [r["name"] for r in roster.get("top15") or [] if isinstance(r, dict) and r.get("name")]
    if not chosen:
        L += ["*No bots selected.*", ""]
    else:
        info = {}
        if isinstance(roster, dict):
            for r in (roster.get("top15") or []) + (roster.get("dropped") or []):
                if isinstance(r, dict) and r.get("name"):
                    info[r["name"]] = r
        L += ["| Bot | Label | Family | Search | Top-15 Elo [95% CI] | Part3 Elo |", "|---|---|---|---|---|---|"]
        for b in chosen:
            r = info.get(b, {})
            ci = ("%s [%s, %s]" % (r.get("top15_elo"), r.get("top15_lo"), r.get("top15_hi"))
                  if r.get("top15_elo") is not None else "")
            L.append("| `%s` | %s | %s | %s | %s | %s |" % (b, _md_cell(r.get("label", "")), _md_cell(r.get("family", "")),
                                                          _md_cell(r.get("search_budget", "")), ci,
                                                          r.get("part3_elo", "")))
        if isinstance(roster, dict) and roster.get("anchor"):
            L += ["", "Elo anchor: %s" % _one_line(roster["anchor"])]
        L.append("")
    # runs
    L += ["## Training runs used (supervised sweep)", ""]
    chosen = [r for r, v in sel["runs"].items() if v]
    if include_all and isinstance(data["sweep"], dict):
        chosen = [r["name"] for r in data["sweep"].get("runs") or [] if isinstance(r, dict) and r.get("name")]
    if not chosen:
        L += ["*No runs selected.*", ""]
    else:
        runs = {}
        if isinstance(data["sweep"], dict):
            runs = {r.get("name"): r for r in data["sweep"].get("runs") or [] if isinstance(r, dict)}
        L += ["| Run | Teacher | Size | Params | Status | Bench optimal | Gauntlet score |", "|---|---|---|---|---|---|---|"]
        for n in chosen:
            r = runs.get(n, {})
            b = r.get("bench") or {}
            g = r.get("gauntlet") or {}
            size = "%sx%s" % (r.get("blocks"), r.get("channels")) if r.get("blocks") else ""
            L.append("| `%s` | %s | %s | %s | %s | %s | %s |" % (n, _md_cell(r.get("teacher", "")), size, r.get("params", ""),
                                                               r.get("status", "not in sweep.json"),
                                                               b.get("optimal", ""), g.get("mean_score", "")))
        L.append("")
    # limitations
    L += ["## Limitations / experiments still needed", ""]
    a = data["arena_analysis"]
    need = []
    if isinstance(a, dict):
        need = (((a.get("best_nn") or {}).get("answer") or {}).get("needed_experiments")) or []
    for n in need:
        L.append("- %s" % _one_line(n))
    if isinstance(data["sweep"], dict):
        st = data["sweep"].get("status") or {}
        if st.get("state") != "done":
            c = st.get("counts") or {}
            L.append("- Supervised sweep not finished (%s of %s runs done when this outline was generated)."
                     % (c.get("done", "?"), st.get("total", "?")))
    else:
        L.append("- Supervised sweep results not available yet.")
    if not isinstance(data["interp"], dict):
        L.append("- Interpretability results not available yet.")
    if not need and isinstance(a, dict):
        L.append("- (arena_analysis.json lists no needed experiments)")
    L += ["", "---", "*%d finding(s) selected.*" % total, ""]
    return "\n".join(L)
