/* Connect 4 Lab page: fetches /api/lab/* and draws the charts as inline SVG
   (no external libraries, works offline). */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var C = { blue: "#56B4E9", orange: "#E69F00", green: "#009E73", yellow: "#F0E442", red: "#D55E00",
            purple: "#CC79A7", sky: "#0072B2", grey: "#8e96b5", pink: "#ef476f", gold: "#ffd166", white: "#e9ecf8" };
  var SEQ = [C.blue, C.orange, C.green, C.pink, C.purple, C.gold, C.red, C.sky, C.grey];
  var REF = null, RUNS = [], ARENAS = [], CACHE = {};

  // ------------------------------------------------------------ helpers
  function el(tag, attrs, html) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) { if (attrs[k] != null) e.setAttribute(k, attrs[k]); }
    if (html != null) e.innerHTML = html;
    return e;
  }
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function pct(v, d) { return v == null || isNaN(v) ? "–" : (100 * v).toFixed(d == null ? 1 : d) + "%"; }
  function num(v, d) { return v == null || isNaN(v) ? "–" : Number(v).toFixed(d == null ? 0 : d); }
  function fmtInt(v) { return v == null ? "–" : Math.round(v).toLocaleString("en-US"); }
  function dur(sec) { if (sec == null) return "–"; if (sec < 90) return Math.round(sec) + " s"; if (sec < 5400) return Math.round(sec / 60) + " min"; return (sec / 3600).toFixed(1) + " h"; }
  function getJSON(url) {
    return fetch(url, { cache: "no-store" }).then(function (r) { if (!r.ok) throw new Error(url + ": " + r.status); return r.json(); });
  }
  function card(title, desc, body, wide) {
    var c = el("div", { "class": "card" + (wide ? " wide" : "") });
    c.appendChild(el("h3", null, esc(title)));
    if (desc) c.appendChild(el("p", { "class": "desc" }, desc));
    if (body) c.appendChild(body);
    return c;
  }
  function tile(k, v, s) { return '<div class="tile"><div class="k">' + k + '</div><div class="v">' + v + "</div>" + (s ? '<div class="s">' + s + "</div>" : "") + "</div>"; }
  function niceTicks(lo, hi, n) {
    if (hi === lo) { hi = lo + 1; }
    var span = hi - lo, step = Math.pow(10, Math.floor(Math.log10(span / n))), err = n / span * step;
    if (err <= 0.15) step *= 10; else if (err <= 0.35) step *= 5; else if (err <= 0.75) step *= 2;
    var t = [], v = Math.ceil(lo / step) * step;
    for (; v <= hi + 1e-9; v += step) t.push(+v.toFixed(10));
    return t;
  }
  function legend(items) {
    var d = el("div", { "class": "legend" });
    items.forEach(function (it) {
      d.appendChild(el("span", null, '<i class="' + (it.dash ? "dash" : "") + '" style="background:' + it.color + ";color:" + it.color + '"></i>' + esc(it.name)));
    });
    return d;
  }

  // ------------------------------------------------------------ line chart
  function lineChart(o) {
    var W = 640, H = o.height || 250, m = { l: 52, r: 16, t: 10, b: 38 };
    var pts = [];
    o.series.forEach(function (s) { s.points.forEach(function (p) { if (p[1] != null && !isNaN(p[1])) pts.push(p); }); });
    var wrap = el("div");
    if (!pts.length) { wrap.appendChild(el("p", { "class": "muted" }, o.emptyText || "No data yet.")); return wrap; }
    var xs = pts.map(function (p) { return p[0]; }), ys = pts.map(function (p) { return p[1]; });
    (o.refs || []).forEach(function (r) { ys.push(r.y); });
    var x0 = o.xMin != null ? o.xMin : Math.min.apply(null, xs), x1 = o.xMax != null ? o.xMax : Math.max.apply(null, xs);
    if (x1 === x0) x1 = x0 + 1;
    var y0 = o.yMin != null ? o.yMin : Math.min.apply(null, ys), y1 = o.yMax != null ? o.yMax : Math.max.apply(null, ys);
    if (o.yMin == null || o.yMax == null) { var pad = (y1 - y0) * 0.08 || 1; if (o.yMin == null) y0 -= pad; if (o.yMax == null) y1 += pad; }
    var X = function (v) { return m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r); };
    var Y = function (v) { return H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b); };
    var yf = o.yFmt || function (v) { return v; };
    var s = '<svg class="chart" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' + esc(o.title || "chart") + '">';
    s += '<g class="grid">';
    niceTicks(y0, y1, 5).forEach(function (t) { s += '<line x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(t) + '" y2="' + Y(t) + '"/>'; s += '<text x="' + (m.l - 6) + '" y="' + (Y(t) + 4) + '" text-anchor="end">' + yf(t) + "</text>"; });
    niceTicks(x0, x1, 7).forEach(function (t) { if (o.intX && t % 1) return; s += '<text x="' + X(t) + '" y="' + (H - m.b + 16) + '" text-anchor="middle">' + (o.xFmt ? o.xFmt(t) : t) + "</text>"; });
    s += "</g>";
    s += '<g class="axis"><line x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + (H - m.b) + '" y2="' + (H - m.b) + '"/></g>';
    if (o.xLabel) s += '<text x="' + ((m.l + W - m.r) / 2) + '" y="' + (H - 4) + '" text-anchor="middle">' + esc(o.xLabel) + "</text>";
    if (o.yLabel) s += '<text transform="translate(12,' + ((m.t + H - m.b) / 2) + ') rotate(-90)" text-anchor="middle">' + esc(o.yLabel) + "</text>";
    (o.refs || []).forEach(function (r, k) {
      if (r.y < y0 || r.y > y1) return;
      var left = k % 2 === 1;
      s += '<g class="ref"><line x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(r.y) + '" y2="' + Y(r.y) + '" stroke="' + (r.color || C.grey) + '"/>';
      s += '<text x="' + (left ? m.l + 4 : W - m.r - 2) + '" y="' + (Y(r.y) - 3) + '" text-anchor="' + (left ? "start" : "end") + '" style="fill:' + (r.color || C.grey) + '">' + esc(r.label) + "</text></g>";
    });
    o.series.forEach(function (se, i) {
      var col = se.color || SEQ[i % SEQ.length];
      var p = se.points.filter(function (q) { return q[1] != null && !isNaN(q[1]); });
      if (!p.length) return;
      if (!se.dotsOnly) {
        var d = p.map(function (q, k) { return (k ? "L" : "M") + X(q[0]).toFixed(1) + " " + Y(q[1]).toFixed(1); }).join("");
        s += '<path d="' + d + '" fill="none" stroke="' + col + '" stroke-width="2"' + (se.dash ? ' stroke-dasharray="6 4"' : "") + "/>";
      }
      if (se.dots || se.dotsOnly || p.length < 25) p.forEach(function (q) { s += '<circle cx="' + X(q[0]) + '" cy="' + Y(q[1]) + '" r="' + (se.dotsOnly ? 4 : 2.6) + '" fill="' + col + '"><title>' + esc(se.name) + ": " + yf(q[1]) + " @ " + q[0] + "</title></circle>"; });
    });
    s += "</svg>";
    wrap.innerHTML = s;
    wrap.appendChild(legend(o.series.map(function (se, i) { return { name: se.name, color: se.color || SEQ[i % SEQ.length], dash: se.dash }; })
      .concat((o.refs || []).filter(function (r) { return r.legend; }).map(function (r) { return { name: r.legend, color: r.color || C.grey, dash: true }; }))));
    return wrap;
  }

  // ------------------------------------------------- grouped bar chart
  function groupedBars(o) {
    var W = 640, H = o.height || 240, m = { l: 48, r: 10, t: 10, b: 40 };
    var cats = o.categories, ser = o.series;
    var vals = []; ser.forEach(function (s) { s.values.forEach(function (v) { if (v != null) vals.push(v); }); });
    var y0 = o.yMin != null ? o.yMin : 0, y1 = o.yMax != null ? o.yMax : Math.max.apply(null, vals.concat([1e-9])) * 1.1;
    var Y = function (v) { return H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b); };
    var gw = (W - m.l - m.r) / cats.length, bw = Math.min(28, gw * 0.8 / ser.length);
    var yf = o.yFmt || function (v) { return v; };
    var s = '<svg class="chart" viewBox="0 0 ' + W + " " + H + '"><g class="grid">';
    niceTicks(y0, y1, 5).forEach(function (t) { s += '<line x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(t) + '" y2="' + Y(t) + '"/><text x="' + (m.l - 6) + '" y="' + (Y(t) + 4) + '" text-anchor="end">' + yf(t) + "</text>"; });
    s += "</g>";
    (o.refs || []).forEach(function (r) { s += '<g class="ref"><line x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(r.y) + '" y2="' + Y(r.y) + '" stroke="' + (r.color || C.white) + '"/><text x="' + (W - m.r) + '" y="' + (Y(r.y) - 3) + '" text-anchor="end">' + esc(r.label) + "</text></g>"; });
    cats.forEach(function (c, i) {
      var gx = m.l + i * gw + (gw - bw * ser.length) / 2;
      ser.forEach(function (se, j) {
        var v = se.values[i]; if (v == null) return;
        var col = se.color || SEQ[j % SEQ.length];
        s += '<rect x="' + (gx + j * bw) + '" y="' + Y(Math.max(v, y0)) + '" width="' + (bw - 2) + '" height="' + Math.max(0, Y(y0) - Y(Math.max(v, y0))) + '" rx="2" fill="' + col + '"><title>' + esc(se.name + " / " + c + ": " + yf(v)) + "</title></rect>";
      });
      s += '<text x="' + (m.l + i * gw + gw / 2) + '" y="' + (H - m.b + 16) + '" text-anchor="middle">' + esc(c) + "</text>";
    });
    if (o.xLabel) s += '<text x="' + ((m.l + W - m.r) / 2) + '" y="' + (H - 4) + '" text-anchor="middle">' + esc(o.xLabel) + "</text>";
    s += "</svg>";
    var wrap = el("div"); wrap.innerHTML = s;
    wrap.appendChild(legend(ser.map(function (se, j) { return { name: se.name, color: se.color || SEQ[j % SEQ.length] }; })));
    return wrap;
  }

  // ------------------------------------- horizontal bars with error bars
  function hBars(items, o) {
    // very wide bootstrap intervals (bots that almost never lose/win) would squash the chart: clip at +-600
    items = items.map(function (it) { var c = {}; for (var k in it) c[k] = it[k];
      if (c.hi != null && c.hi > c.value + 600) { c.hi = c.value + 600; c.clipped = true; }
      if (c.lo != null && c.lo < c.value - 600) { c.lo = c.value - 600; c.clipped = true; } return c; });
    var W = 640, rowH = 26, m = { l: 230, r: 60, t: 6, b: 28 }, H = m.t + m.b + rowH * items.length;
    var lo = Math.min.apply(null, items.map(function (i) { return i.lo != null ? i.lo : i.value; }));
    var hi = Math.max.apply(null, items.map(function (i) { return i.hi != null ? i.hi : i.value; }));
    if (o && o.min != null) lo = Math.min(lo, o.min);
    var pad = (hi - lo) * 0.06 || 50; lo -= pad; hi += pad;
    var X = function (v) { return m.l + (v - lo) / (hi - lo) * (W - m.l - m.r); };
    var s = '<svg class="chart" viewBox="0 0 ' + W + " " + H + '"><g class="grid">';
    niceTicks(lo, hi, 6).forEach(function (t) { s += '<line x1="' + X(t) + '" x2="' + X(t) + '" y1="' + m.t + '" y2="' + (H - m.b) + '"/><text x="' + X(t) + '" y="' + (H - m.b + 15) + '" text-anchor="middle">' + Math.round(t) + "</text>"; });
    s += "</g>";
    items.forEach(function (it, i) {
      var y = m.t + i * rowH + rowH / 2;
      s += '<text x="' + (m.l - 8) + '" y="' + (y + 4) + '" text-anchor="end" style="fill:' + (it.labelColor || "#e9ecf8") + '">' + esc(it.label) + "</text>";
      s += '<line x1="' + X(lo) + '" x2="' + X(it.value) + '" y1="' + y + '" y2="' + y + '" stroke="' + it.color + '" stroke-width="10" stroke-opacity=".55"/>';
      if (it.lo != null) s += '<line x1="' + X(it.lo) + '" x2="' + X(it.hi) + '" y1="' + y + '" y2="' + y + '" stroke="#e9ecf8" stroke-width="1.5"/><line x1="' + X(it.lo) + '" x2="' + X(it.lo) + '" y1="' + (y - 5) + '" y2="' + (y + 5) + '" stroke="#e9ecf8"/><line x1="' + X(it.hi) + '" x2="' + X(it.hi) + '" y1="' + (y - 5) + '" y2="' + (y + 5) + '" stroke="#e9ecf8"/>';
      s += '<circle cx="' + X(it.value) + '" cy="' + y + '" r="4.5" fill="' + it.color + '"/>';
      s += '<text x="' + (W - m.r + 6) + '" y="' + (y + 4) + '">' + Math.round(it.value) + (it.clipped ? "*" : "") + "</text>";
    });
    if (o && o.xLabel) s += '<text x="' + ((m.l + W - m.r) / 2) + '" y="' + (H - 2) + '" text-anchor="middle">' + esc(o.xLabel) + "</text>";
    s += "</svg>";
    var d = el("div"); d.innerHTML = s; return d;
  }

  // ---------------------------------------------------- scatter (log x)
  function scatter(items, o) {
    var W = 640, H = 280, m = { l: 52, r: 16, t: 12, b: 40 };
    var xs = items.map(function (i) { return Math.log10(Math.max(i.x, 0.01)); });
    var x0 = Math.floor(Math.min.apply(null, xs)), x1 = Math.ceil(Math.max.apply(null, xs)); if (x1 === x0) x1++;
    var y0 = o.yMin, y1 = o.yMax;
    var X = function (v) { return m.l + (Math.log10(Math.max(v, 0.01)) - x0) / (x1 - x0) * (W - m.l - m.r); };
    var Y = function (v) { return H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b); };
    var s = '<svg class="chart" viewBox="0 0 ' + W + " " + H + '"><g class="grid">';
    niceTicks(y0, y1, 5).forEach(function (t) { s += '<line x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(t) + '" y2="' + Y(t) + '"/><text x="' + (m.l - 6) + '" y="' + (Y(t) + 4) + '" text-anchor="end">' + (o.yFmt ? o.yFmt(t) : t) + "</text>"; });
    for (var e = x0; e <= x1; e++) { var xv = Math.pow(10, e); s += '<line x1="' + X(xv) + '" x2="' + X(xv) + '" y1="' + m.t + '" y2="' + (H - m.b) + '"/><text x="' + X(xv) + '" y="' + (H - m.b + 15) + '" text-anchor="middle">' + (xv >= 1000 ? xv / 1000 + " s" : xv + " ms") + "</text>"; }
    s += "</g>";
    items.forEach(function (it) {
      s += '<circle cx="' + X(it.x) + '" cy="' + Y(it.y) + '" r="6" fill="' + it.color + '" fill-opacity=".85"><title>' + esc(it.label) + "</title></circle>";
      s += '<text x="' + (X(it.x) + 9) + '" y="' + (Y(it.y) + 4) + '" style="fill:#cfd5ee">' + esc(it.short || it.label) + "</text>";
    });
    s += '<text x="' + ((m.l + W - m.r) / 2) + '" y="' + (H - 4) + '" text-anchor="middle">' + esc(o.xLabel) + '</text><text transform="translate(12,' + ((m.t + H - m.b) / 2) + ') rotate(-90)" text-anchor="middle">' + esc(o.yLabel) + "</text></svg>";
    var d = el("div"); d.innerHTML = s; return d;
  }

  // -------------------------------------------------------- tables
  function table(head, rows, cls) {
    var h = '<div class="scroll"><table class="t ' + (cls || "") + '"><thead><tr>' + head.map(function (x) { return "<th>" + x + "</th>"; }).join("") + "</tr></thead><tbody>";
    rows.forEach(function (r) { var c = r.cls ? ' class="' + r.cls + '"' : ""; var cells = r.cells || r; h += "<tr" + c + ">" + cells.map(function (x) { return "<td>" + (x == null ? "–" : x) + "</td>"; }).join("") + "</tr>"; });
    return el("div", null, h + "</tbody></table></div>");
  }
  function heatColor(v) { // 0 red -> 0.5 grey -> 1 green
    if (v == null) return "#1a1f38";
    var a = [213, 94, 0], b = [142, 150, 181], c = [0, 158, 115], t, x, y;
    if (v < 0.5) { t = v / 0.5; x = a; y = b; } else { t = (v - 0.5) / 0.5; x = b; y = c; }
    return "rgb(" + [0, 1, 2].map(function (i) { return Math.round(x[i] + (y[i] - x[i]) * t); }).join(",") + ")";
  }

  // ---------------------------------------------------- first-move strip
  function firstMoveStrip(metrics) {
    var rows = metrics.filter(function (r) { return r.first_move_hist; });
    var W = 640, m = { l: 52, r: 16, t: 8, b: 34 }, H = 7 * 22 + m.t + m.b;
    if (!rows.length) return el("p", { "class": "muted" }, "No self-play data yet.");
    var cw = (W - m.l - m.r) / rows.length;
    var s = '<svg class="chart" viewBox="0 0 ' + W + " " + H + '">';
    for (var c = 0; c < 7; c++) s += '<text x="' + (m.l - 8) + '" y="' + (m.t + c * 22 + 15) + '" text-anchor="end">col ' + (c + 1) + "</text>";
    rows.forEach(function (r, i) {
      var tot = r.first_move_hist.reduce(function (a, b) { return a + b; }, 0) || 1;
      for (var c = 0; c < 7; c++) {
        var f = r.first_move_hist[c] / tot;
        s += '<rect x="' + (m.l + i * cw) + '" y="' + (m.t + c * 22) + '" width="' + Math.max(cw - 0.5, 0.5) + '" height="20" fill="' + C.gold + '" fill-opacity="' + (0.05 + 0.95 * f).toFixed(3) + '"><title>gen ' + r.gen + ", column " + (c + 1) + ": " + pct(f) + "</title></rect>";
      }
    });
    niceTicks(rows[0].gen, rows[rows.length - 1].gen, 6).forEach(function (t) {
      var i = rows.findIndex(function (r) { return r.gen >= t; }); if (i < 0) return;
      s += '<text x="' + (m.l + i * cw + cw / 2) + '" y="' + (H - m.b + 15) + '" text-anchor="middle">' + t + "</text>";
    });
    s += '<text x="' + ((m.l + W - m.r) / 2) + '" y="' + (H - 4) + '" text-anchor="middle">generation</text></svg>';
    var d = el("div"); d.innerHTML = s; return d;
  }

  // ============================================================ training
  function refAcc(depth, bucket) { var d = REF && REF.accuracy["d" + depth]; return d && d[bucket || "all"] ? d[bucket || "all"] : null; }
  function accRefs() {
    var out = [], cols = { 1: C.grey, 4: C.purple, 8: C.white };
    [1, 4, 8].forEach(function (d) { var r = refAcc(d); if (r) out.push({ y: r.optimal, label: "minimax depth " + d, color: cols[d] }); });
    return out;
  }

  function renderRunSelect() {
    var sel = $("runSel"), hs = $("hypSel");
    sel.innerHTML = ""; hs.innerHTML = "";
    RUNS.forEach(function (r) {
      var lab = r.name + "  (" + (r.phase === "selfplay" ? "self-play" : r.phase || "?") + ", " + r.points + (r.phase === "selfplay" ? " gens" : " epochs") + ")";
      sel.appendChild(el("option", { value: r.name }, esc(lab)));
      if (r.has_eval) hs.appendChild(el("option", { value: r.name }, esc(r.name)));
    });
    var want = (location.hash.match(/run=([^&]+)/) || [])[1];
    if (want && RUNS.some(function (r) { return r.name === decodeURIComponent(want); })) sel.value = decodeURIComponent(want);
  }

  function emptyTraining() {
    $("trainBody").innerHTML = "";
    $("trainBody").appendChild(el("div", { "class": "empty" },
      "<b>No training runs yet.</b> Train on the GPU machine (see docs/NEURAL_NET.md), commit the <code>runs/&lt;name&gt;/</code> folder, pull it here and refresh. Quick start:" +
      "<pre>python -m nn.make_dataset --positions 300000\npython -m nn.train_supervised --run sup_6x64\npython -m nn.train_selfplay --run az_6x64 --preset standard\npython -m nn.evaluate --run az_6x64</pre>"));
  }

  function loadRun(name) {
    if (!name) return emptyTraining();
    $("trainBody").innerHTML = '<p class="muted">Loading ' + esc(name) + "...</p>";
    getJSON("/api/lab/run/" + encodeURIComponent(name)).then(function (d) { CACHE[name] = d; renderRun(d); })
      .catch(function (e) { $("trainBody").innerHTML = '<div class="empty">Could not load run: ' + esc(e.message) + "</div>"; });
  }

  function renderRun(d) {
    var body = $("trainBody"); body.innerHTML = "";
    var cfg = d.config || {}, M = d.metrics || [], sp = cfg.phase === "selfplay";
    var xk = sp ? "gen" : "epoch", xl = sp ? "generation" : "epoch";
    var last = M[M.length - 1] || {};
    var mc = cfg.model || {};
    $("runInfo").textContent = (sp ? "AlphaZero self-play" : "Supervised (perfect-play labels)") + " · " + (mc.blocks || "?") + "×" + (mc.channels || "?") + " ResNet, " + fmtInt(cfg.params) + " params · " + (cfg.device || "") + " · started " + (cfg.created || "").replace("T", " ");
    var best = M.reduce(function (a, r) { return Math.max(a, r.bench_optimal || 0); }, 0);
    var mctsBest = M.reduce(function (a, r) { return Math.max(a, r.bench_mcts_optimal || 0); }, 0);
    var elos = M.filter(function (r) { return r.elo != null; });
    var ev = d.eval;
    var t = '<div class="tiles">';
    t += tile(sp ? "Generations" : "Epochs", fmtInt(last[xk]), sp ? fmtInt(last.total_games) + " self-play games" : fmtInt(last.positions_seen) + " positions seen");
    t += tile("Training time", dur(last.elapsed_sec), last.time ? "last update " + last.time.replace("T", " ") : "");
    t += tile("Best move accuracy", pct(best), "policy only, no search");
    if (mctsBest) t += tile("With MCTS", pct(mctsBest), (last.bench_mcts_sims || (M.filter(function (r) { return r.bench_mcts_sims; }).slice(-1)[0] || {}).bench_mcts_sims || "") + " simulations");
    t += tile("Value accuracy", pct(last.bench_value_acc), "win/draw/loss vs solver");
    var r8 = refAcc(8); if (r8) t += tile("Minimax depth 8", pct(r8.optimal), "same test, for comparison");
    if (ev && ev.ladder && ev.ladder.elo != null) t += tile("Elo (evaluation)", num(ev.ladder.elo), "minimax d1 = 1000, d8 ≈ " + num((REF.elo.d8 || {}).elo));
    else if (elos.length) t += tile("Elo (latest)", num(elos[elos.length - 1].elo), "minimax d1 = 1000");
    body.appendChild(el("div", null, t + "</div>"));

    var g = el("div", { "class": "grid g2" });
    body.appendChild(g);
    var pts = function (k, f) { return M.filter(function (r) { return r[k] != null; }).map(function (r) { return [r[xk], f ? f(r[k]) : r[k]]; }); };

    g.appendChild(card("Accuracy vs perfect play", "Share of 3,000 held-out positions where the network picks a move the perfect solver also rates best. Dashed lines: our minimax at depth 1/4/8 on the same positions.",
      lineChart({ series: [{ name: "policy head (no search)", points: pts("bench_optimal"), color: C.blue },
                           { name: "network + MCTS", points: pts("bench_mcts_optimal"), color: C.orange, dots: true },
                           { name: "non-blunder rate (policy)", points: pts("bench_keeps_nl"), color: C.green, dash: true }],
                  refs: accRefs(), yMin: 0, yMax: 1, yFmt: function (v) { return Math.round(v * 100) + "%"; }, xLabel: xl, intX: true })));
    var lossSeries = sp ? [{ name: "policy loss", points: pts("policy_loss"), color: C.blue }, { name: "value loss", points: pts("value_loss"), color: C.orange }]
      : [{ name: "train policy", points: pts("train_policy_loss"), color: C.blue }, { name: "train value", points: pts("train_value_loss"), color: C.orange },
         { name: "val policy", points: pts("val_policy_loss"), color: C.blue, dash: true }, { name: "val value", points: pts("val_value_loss"), color: C.orange, dash: true }];
    g.appendChild(card("Training loss", sp ? "Cross-entropy of the policy against MCTS visit counts and of the value against game results. In self-play the targets move as the network improves, so the loss does not have to go to zero." : "Cross-entropy on training data (solid) and on the 5% validation split (dashed). Val going up while train goes down = overfitting.",
      lineChart({ series: lossSeries, xLabel: xl, intX: true, yFmt: function (v) { return v.toFixed(2); } })));
    g.appendChild(card("Value head accuracy", "Does the network's win/draw/loss prediction match the solver? (Majority-class baseline: always 'win' = 63%.)",
      lineChart({ series: [{ name: "held-out benchmark", points: pts("bench_value_acc"), color: C.green }].concat(sp ? [] : [{ name: "validation split", points: pts("val_value_acc"), color: C.green, dash: true }]),
                  refs: [{ y: 0.6333, label: "always 'win'", color: C.grey }], yMin: 0.5, yMax: 1, yFmt: function (v) { return Math.round(v * 100) + "%"; }, xLabel: xl, intX: true })));

    // by game phase (latest)
    var lb = M.filter(function (r) { return r.bench_by_bucket; }).slice(-1)[0];
    if (lb) {
      var bk = ["8-11", "12-15", "16-19", "20-23", "24-27", "28-30"];
      var ser = [{ name: "network (policy)", values: bk.map(function (b) { return (lb.bench_by_bucket[b] || {}).optimal; }), color: C.blue }];
      [4, 8].forEach(function (dd, i) { if (refAcc(dd)) ser.push({ name: "minimax depth " + dd, values: bk.map(function (b) { return (refAcc(dd, b) || {}).optimal; }), color: i ? C.white : C.purple }); });
      g.appendChild(card("Accuracy by game phase", "Latest " + xl + ", split by how many stones are on the board. Early positions are hardest for search (the win is far away); the network can be strong there.",
        groupedBars({ categories: bk, series: ser, yMin: 0, yMax: 1, yFmt: function (v) { return Math.round(v * 100) + "%"; }, xLabel: "stones on the board" })));
    }

    if (sp) {
      var ladderRefs = [];
      if (REF && REF.elo) ["d1", "d2", "d4", "d6", "d8"].forEach(function (k) { if (REF.elo[k]) ladderRefs.push({ y: REF.elo[k].elo, label: "minimax " + k, color: C.grey }); });
      g.appendChild(card("Strength (Elo) over training", "Estimated from games against our minimax at fixed depths, on the same Elo scale as the depth tournament (depth 1 = 1000). Dashed lines: minimax depths.",
        lineChart({ series: [{ name: "network + MCTS", points: pts("elo"), color: C.gold, dots: true }], refs: ladderRefs, xLabel: xl, intX: true, emptyText: "Measured every --eval-every generations." })));
      var depthKeys = {};
      M.forEach(function (r) { if (r.vs_minimax) Object.keys(r.vs_minimax).forEach(function (k) { depthKeys[k] = 1; }); });
      g.appendChild(card("Score vs minimax", "Share of points (draw = ½) against minimax at each depth.",
        lineChart({ series: Object.keys(depthKeys).sort().map(function (k, i) {
          return { name: "vs " + k.replace("d", "depth "), points: M.filter(function (r) { return r.vs_minimax && r.vs_minimax[k]; }).map(function (r) { return [r.gen, r.vs_minimax[k].score]; }), color: SEQ[i] };
        }), refs: [{ y: 0.5, label: "even", color: C.grey }], yMin: 0, yMax: 1, yFmt: function (v) { return Math.round(v * 100) + "%"; }, xLabel: xl, intX: true })));
      g.appendChild(card("Self-play games", "Who wins the network's games against itself. With perfect play the first player always wins, so a P1 share near 100% late in training is a sign of strong play.",
        lineChart({ series: [{ name: "first player wins", points: pts("p1_win"), color: C.pink }, { name: "draws", points: pts("draw"), color: C.grey }, { name: "second player wins", points: pts("p2_win"), color: C.gold }],
                    yMin: 0, yMax: 1, yFmt: function (v) { return Math.round(v * 100) + "%"; }, xLabel: xl, intX: true })));
      g.appendChild(card("Game length", "Average number of moves per self-play game.",
        lineChart({ series: [{ name: "moves per game", points: pts("avg_length"), color: C.blue }], xLabel: xl, intX: true })));
      g.appendChild(card("Where does it play first?", "First move of every self-play game, per generation (brighter = more often). Perfect play starts in the centre (column 4).",
        firstMoveStrip(M)));
      g.appendChild(card("What it thinks of the empty board", "Value head on the empty board. Connect 4 is a first-player win, so P(win) should rise toward 1.",
        lineChart({ series: [{ name: "P(first player wins)", points: M.filter(function (r) { return r.empty_board_wdl; }).map(function (r) { return [r.gen, r.empty_board_wdl[0]]; }), color: C.green },
                             { name: "P(draw)", points: M.filter(function (r) { return r.empty_board_wdl; }).map(function (r) { return [r.gen, r.empty_board_wdl[1]]; }), color: C.grey },
                             { name: "P(centre first) by policy", points: M.filter(function (r) { return r.empty_board_policy; }).map(function (r) { return [r.gen, r.empty_board_policy[3]]; }), color: C.gold, dash: true }],
                    yMin: 0, yMax: 1, yFmt: function (v) { return Math.round(v * 100) + "%"; }, xLabel: xl, intX: true })));
      g.appendChild(card("Time per generation", "Seconds spent on self-play, training and evaluation.",
        lineChart({ series: [{ name: "self-play", points: pts("selfplay_sec"), color: C.blue }, { name: "training", points: pts("train_sec"), color: C.orange }, { name: "evaluation", points: pts("eval_sec"), color: C.grey }], xLabel: xl, intX: true })));
    } else {
      g.appendChild(card("Validation accuracy", "On the held-out 5% of the generated data: is the policy's top move optimal, and is the value's call right?",
        lineChart({ series: [{ name: "policy top move optimal", points: pts("val_policy_acc"), color: C.blue }, { name: "value correct", points: pts("val_value_acc"), color: C.green }],
                    yMin: 0, yMax: 1, yFmt: function (v) { return Math.round(v * 100) + "%"; }, xLabel: xl, intX: true })));
    }

    if (ev) body.appendChild(renderEval(ev));
    if (d.notes) { var n = el("div", { "class": "grid", style: "margin-top:14px" }); var nd = el("div", { "class": "notes" }); nd.textContent = d.notes; n.appendChild(card("Notes (runs/" + d.name + "/notes.md)", null, nd, true)); body.appendChild(n); }
    body.appendChild(renderRunsOverview());
  }

  function renderEval(ev) {
    var g = el("div", { "class": "grid g2", style: "margin-top:14px" });
    var acc = ev.accuracy || {};
    var rows = [];
    Object.keys(acc).filter(function (k) { return k !== "minimax_reference"; }).forEach(function (k) {
      rows.push({ cls: "ours", cells: [k.replace("nn_", "network: ").replace("mcts", "MCTS ").replace("policy", "policy only"), pct(acc[k].optimal), pct(acc[k].keeps_nl)] });
    });
    var mr = acc.minimax_reference || {};
    Object.keys(mr).sort(function (a, b) { return +a.slice(1) - +b.slice(1); }).forEach(function (k) { rows.push([k.replace("d", "minimax depth "), pct(mr[k].optimal), pct(mr[k].keeps_nl)]); });
    if (rows.length) g.appendChild(card("Final evaluation: accuracy vs perfect play", "From nn/evaluate.py on " + esc(ev.model) + ".", table(["player", "optimal move", "non-blunder"], rows)));
    if (ev.value) {
      var cm = ev.value.confusion, lab = ev.value.labels;
      g.appendChild(card("Value head: confusion matrix", "Rows: true result (solver). Columns: network's call. Accuracy " + pct(ev.value.accuracy) + ".",
        table(["truth \\ predicted"].concat(lab), cm.map(function (r, i) { return [lab[i]].concat(r.map(fmtInt)); }))));
    }
    if (ev.tactics) {
      var cats = ["win1", "block", "win2", "win3", "all"], tr = [];
      Object.keys(ev.tactics).filter(function (k) { return k !== "reference"; }).forEach(function (k) { tr.push({ cls: "ours", cells: [k.replace("nn_", "network: ")].concat(cats.map(function (c) { return pct(ev.tactics[k][c], 0); })) }); });
      var ref = ev.tactics.reference || {};
      ["minimax_d1", "minimax_d2", "minimax_d4", "minimax_d8"].forEach(function (k) { if (ref[k]) tr.push([k.replace("minimax_d", "minimax depth ")].concat(cats.map(function (c) { return pct(ref[k][c], 0); }))); });
      g.appendChild(card("Tactics suite (52 positions)", "Win in 1, forced block, win in 2 and win in 3 moves; share solved.", table(["player", "win in 1", "block", "win in 2", "win in 3", "all"], tr)));
    }
    if (ev.ladder) {
      var L = ev.ladder, lr = Object.keys(L.vs).map(function (k) { var v = L.vs[k]; return [k.replace("minimax_d", "minimax depth "), v.win + "-" + v.draw + "-" + v.loss, pct(v.score), num(v.a_ms) + " / " + num(v.b_ms)]; });
      g.appendChild(card("Games vs minimax (" + esc(L.player) + ")", "W-D-L from the network's side, " + L.openings + " openings × 2 colours each. Elo on the minimax ladder: <b>" + num(L.elo) + "</b>.",
        table(["opponent", "W-D-L", "score", "ms/move (net / minimax)"], lr)));
    }
    return g;
  }

  function renderRunsOverview() {
    var w = el("div", { "class": "grid", style: "margin-top:14px" });
    var rows = RUNS.map(function (r) {
      return [esc(r.name), r.phase === "selfplay" ? "self-play" : "supervised", (r.model ? r.model.blocks + "×" + r.model.channels : "–"), fmtInt(r.params),
              fmtInt(r.last.gen || r.last.epoch), r.last.total_games ? fmtInt(r.last.total_games) : "–", dur(r.last.elapsed_sec), pct(r.best_bench_optimal), num(r.eval_elo || r.last_elo)];
    });
    var series = RUNS.map(function (r, i) { var d = CACHE[r.name]; if (!d) return null; return { name: r.name, color: SEQ[i % SEQ.length], points: d.metrics.filter(function (m) { return m.bench_optimal != null && m.elapsed_sec != null; }).map(function (m) { return [m.elapsed_sec / 3600, m.bench_optimal]; }) }; }).filter(Boolean);
    var c = card("All runs", "Compare training runs. The chart shows runs you have opened on this page; x = hours of training.", table(["run", "phase", "net", "params", "gen/epoch", "games", "time", "best accuracy", "Elo"], rows), true);
    if (series.length > 1) c.appendChild(lineChart({ series: series, refs: accRefs(), yMin: 0, yMax: 1, yFmt: function (v) { return Math.round(v * 100) + "%"; }, xLabel: "hours of training", xFmt: function (v) { return v.toFixed(1); } }));
    w.appendChild(c);
    return w;
  }

  // ========================================================== hypothesis
  function loadHyp(name) {
    var body = $("hypBody");
    if (!name) {
      body.innerHTML = '<div class="empty"><b>No evaluated network yet.</b> After training, run<pre>python -m nn.evaluate --run &lt;run&gt;</pre>It plays minimax-with-the-network\'s-evaluation against minimax-with-the-hand-crafted-heuristic at the same depths and writes runs/&lt;run&gt;/eval.json.</div>';
      return;
    }
    var p = CACHE[name] ? Promise.resolve(CACHE[name]) : getJSON("/api/lab/run/" + encodeURIComponent(name));
    p.then(function (d) { CACHE[name] = d; renderHyp(d); });
  }
  function renderHyp(d) {
    var body = $("hypBody"); body.innerHTML = "";
    var h = d.eval && d.eval.hypothesis;
    if (!h || !h.rows || !h.rows.length) { body.innerHTML = '<div class="empty">This run\'s eval.json has no hypothesis section (it was skipped or is still running).</div>'; return; }
    var ok = h.rows.filter(function (r) { return r.supported; }).length;
    body.appendChild(el("div", { "class": "tiles" },
      tile("Hypothesis", ok === h.rows.length ? '<span class="yes">supported</span>' : ok ? '<span class="yes">partly</span>' : '<span class="no">not supported</span>', esc(h.verdict || "")) +
      tile("Depths tested", h.rows.map(function (r) { return r.depth; }).join(", "), h.openings + " openings × 2 colours per depth") +
      tile("Accuracy positions", fmtInt(h.positions), "held-out, solver-labelled")));
    var g = el("div", { "class": "grid g2" }); body.appendChild(g);
    var cats = h.rows.map(function (r) { return "depth " + r.depth; });
    g.appendChild(card("Head to head at the same depth", "Score of minimax using the network's value vs minimax using our hand-crafted heuristic (draw = ½). Above 50% = the network's evaluation is better.",
      groupedBars({ categories: cats, series: [{ name: "network evaluation", values: h.rows.map(function (r) { return r.head_to_head.score; }), color: C.gold }], refs: [{ y: 0.5, label: "50%" }], yMin: 0, yMax: 1, yFmt: function (v) { return Math.round(v * 100) + "%"; } })));
    g.appendChild(card("Accuracy vs perfect play", "Same held-out positions for both (paired; McNemar test in the table).",
      groupedBars({ categories: cats, series: [{ name: "network evaluation", values: h.rows.map(function (r) { return r.nn_accuracy.optimal; }), color: C.gold }, { name: "hand-crafted heuristic", values: h.rows.map(function (r) { return r.hand_accuracy.optimal; }), color: C.blue }], yMin: 0, yMax: 1, yFmt: function (v) { return Math.round(v * 100) + "%"; } })));
    var rows = h.rows.map(function (r) {
      var m = r.head_to_head, c = r.criteria || {};
      return ["depth " + r.depth, m.win + "-" + m.draw + "-" + m.loss, pct(m.score), num(r.binom_p, 3), pct(r.nn_accuracy.optimal) + " / " + pct(r.hand_accuracy.optimal), num(r.mcnemar_p, 3),
              pct(r.nn_tactics.win1, 0) + " / " + pct((r.hand_tactics || {}).win1, 0), pct(r.nn_tactics.block, 0) + " / " + pct((r.hand_tactics || {}).block, 0), num(r.nn_ms, 1) + " / " + num(r.hand_ms, 1),
              r.supported ? '<span class="yes">yes</span>' : '<span class="no">no</span>' + (c.score_at_least_50pct === false ? " (score)" : "") + ((c.wins_and_blocks_at_least_as_hand || c.takes_wins_and_blocks) === false ? " (tactics)" : "") + (c.fast_enough_2s === false ? " (time)" : "")];
    });
    g.appendChild(card("Details", "W-D-L and score are from the network-evaluation side. p = exact binomial test on decisive games (small p = the difference is unlikely to be luck). Accuracy, tactics and time: network / hand-crafted. Supported = score ≥ 50%, wins and blocks at least as often as the hand-crafted version, and under 2 s per move.",
      table(["depth", "W-D-L", "score", "p", "optimal", "McNemar p", "win in 1", "forced block", "ms per move", "supported"], rows), true));
  }

  // =============================================================== arena
  function loadArenaList() {
    return getJSON("/api/lab/arena").then(function (list) {
      ARENAS = list; var s = $("arenaSel"); s.innerHTML = "";
      list.forEach(function (a) { s.appendChild(el("option", { value: a.name }, esc(a.name + " (" + a.bots + " bots, " + fmtInt(a.games) + " games)"))); });
      if (!list.length) {
        $("arenaBody").innerHTML = '<div class="empty"><b>No tournament results yet.</b> On a machine with the bots installed:<pre>pip install -r arena/requirements.txt\npython -m arena.fetch\ncd tools/pons &amp;&amp; make c4solver &amp;&amp; cd ../..\npython -m arena.tournament --list\npython -m arena.tournament --openings 20</pre>Results land in experiments/results/arena/main/summary.json.</div>';
        return;
      }
      loadArena(s.value);
    });
  }
  function groupOf(info) { return info.group === "ours" ? "ours" : info.group === "open source" ? "os" : "base"; }
  function loadArena(name) {
    getJSON("/api/lab/arena/" + encodeURIComponent(name)).then(renderArena)
      .catch(function (e) { $("arenaBody").innerHTML = '<div class="empty">' + esc(e.message) + "</div>"; });
  }
  function renderArena(s) {
    var body = $("arenaBody"); body.innerHTML = "";
    var info = s.bot_info || {}, P = s.per_bot, meta = s.meta || {};
    $("arenaInfo").textContent = fmtInt(s.games) + " games · " + (meta.openings || []).length + " openings × 2 colours per pairing · " + (meta.created || "").replace("T", " ") + " · " + dur(meta.seconds);
    var order = s.bots.slice().sort(function (a, b) { var ea = (s.elo[a] || {}).elo, eb = (s.elo[b] || {}).elo; if (P[a].score === 1 && P[b].score !== 1) return -1; if (P[b].score === 1 && P[a].score !== 1) return 1; if (ea != null && eb != null) return eb - ea; return (P[b].score || 0) - (P[a].score || 0); });
    var lbl = function (n) { return (info[n] && info[n].label) || n; };
    var top = order[0];
    body.appendChild(el("div", { "class": "tiles" },
      tile("Bots", s.bots.length, Object.keys(info).filter(function (n) { return info[n].group === "open source"; }).length + " open source") +
      tile("Games", fmtInt(s.games), fmtInt(s.decisive) + " decisive") +
      tile("Top bot", esc(lbl(top)), pct(P[top].score) + " of points") +
      tile("First-mover score", pct(s.first_player_score), "all games, draw = ½") +
      tile("Forfeits", fmtInt(s.forfeits.length), "crashes or illegal moves")));
    var g = el("div", { "class": "grid g2" }); body.appendChild(g);
    var rows = order.map(function (n, i) {
      var p = P[n], e = s.elo[n], inf = info[n] || {};
      var gb = groupOf(inf), badge = '<span class="badge ' + gb + '">' + (gb === "ours" ? "ours" : gb === "os" ? "open source" : "baseline") + "</span>";
      return { cls: gb === "ours" ? "ours" : "", cells: [(i + 1) + ". " + esc(lbl(n)) + badge, e ? num(e.elo) + ' <span class="muted">[' + num(e.lo) + ", " + num(e.hi) + "]</span>" : '<span class="muted">' + (p.score === 1 ? "undefeated" : p.score === 0 ? "never scored" : "–") + "</span>",
        pct(p.score), p.wins + "-" + p.draws + "-" + p.losses, num(p.avg_ms, 1), num(p.max_ms, 0), p.forfeits] };
    });
    g.appendChild(card("Leaderboard", "Score = share of points (win 1, draw ½). Elo: Bradley-Terry fit to all games, " + esc(s.elo_anchor || "") + " fixed at 1000, 95% bootstrap interval in brackets. Bots that won or lost every game have no finite Elo.",
      table(["bot", "Elo [95% CI]", "score", "W-D-L", "ms/move", "max ms", "forfeits"], rows), true));
    var items = order.filter(function (n) { return s.elo[n]; }).map(function (n) { var e = s.elo[n], gb = groupOf(info[n] || {}); return { label: lbl(n), value: e.elo, lo: e.lo, hi: e.hi, color: gb === "ours" ? C.gold : gb === "os" ? C.blue : C.grey }; });
    if (items.length) g.appendChild(card("Elo with 95% confidence intervals", "Blue = open-source bots, gold = ours. Overlapping intervals = not clearly different. * = interval wider than ±600, clipped (see table).", hBars(items, { xLabel: "Elo" })));
    var sc = order.map(function (n) { var gb = groupOf(info[n] || {}); return { x: Math.max(P[n].avg_ms || 0.01, 0.01), y: P[n].score, label: lbl(n) + ": " + num(P[n].avg_ms, 1) + " ms/move", short: (lbl(n).length > 22 ? lbl(n).slice(0, 21) + "…" : lbl(n)), color: gb === "ours" ? C.gold : gb === "os" ? C.blue : C.grey }; });
    g.appendChild(card("Strength vs thinking time", "Average time per move (log scale) against score. Up and to the left = efficient.", scatter(sc, { yMin: 0, yMax: 1, yFmt: function (v) { return Math.round(v * 100) + "%"; }, xLabel: "average time per move", yLabel: "score" })));
    // heatmap
    var h = '<div class="scroll"><table class="t heat"><thead><tr><th>row vs column</th>' + order.map(function (n) { return '<th class="rot"><div>' + esc(lbl(n)) + "</div></th>"; }).join("") + "</tr></thead><tbody>";
    var idx = {}; s.bots.forEach(function (n, i) { idx[n] = i; });
    order.forEach(function (a) {
      h += "<tr><td>" + esc(lbl(a)) + "</td>";
      order.forEach(function (b) {
        var v = a === b ? null : s.matrix[idx[a]][idx[b]], w = (s.wdl[a] || {})[b];
        h += a === b ? '<td style="background:#141832"></td>' : '<td style="background:' + heatColor(v) + '" title="' + esc(lbl(a) + " vs " + lbl(b) + (w ? ": " + w[0] + "-" + w[1] + "-" + w[2] : "")) + '">' + (v == null ? "–" : Math.round(v * 100)) + "</td>";
      });
      h += "</tr>";
    });
    h += "</tbody></table></div>";
    g.appendChild(card("Head-to-head score matrix", "Each cell: % of points the ROW bot took from the COLUMN bot (hover for W-D-L). Green = row bot better.", el("div", null, h), true));
    // bot cards
    var cards = el("div", { "class": "botcards" });
    order.forEach(function (n) {
      var inf = info[n] || {}, prm = inf.params || {};
      var ps = Object.keys(prm).map(function (k) { return k + "=" + prm[k]; }).join(", ");
      cards.appendChild(el("div", { "class": "botcard" }, "<b>" + esc(lbl(n)) + "</b> <span class='badge " + groupOf(inf) + "'>" + esc(inf.group || "") + "</span><br>" +
        (inf.author ? esc(inf.author) + " · " : "") + (inf.license ? esc(inf.license) + " · " : "") + esc(inf.lang || "") + "<br>" +
        (inf.url ? '<a href="' + esc(inf.url) + '" target="_blank" rel="noopener">' + esc(inf.url.replace("https://github.com/", "github: ")) + "</a>" + (inf.commit ? " @ " + esc(inf.commit) : "") + "<br>" : "") +
        '<span class="muted">' + esc(inf.algorithm || "") + "</span>" + (ps ? "<br>settings: <code>" + esc(ps) + "</code>" : "")));
    });
    g.appendChild(card("The bots", "Source, licence, algorithm and the settings used in this tournament.", cards, true));
    if (s.forfeits.length) { var f = el("div", { "class": "notes" }); f.textContent = s.forfeits.join("\n"); g.appendChild(card("Forfeits", "Games a bot lost by crashing or playing an illegal move.", f, true)); }
  }

  // ================================================================ boot
  function tabs() {
    var btns = document.querySelectorAll(".tabs button");
    function show(name) {
      btns.forEach(function (b) { b.classList.toggle("on", b.getAttribute("data-tab") === name); b.setAttribute("aria-selected", b.getAttribute("data-tab") === name); });
      document.querySelectorAll(".tab").forEach(function (t) { t.classList.toggle("on", t.id === "tab-" + name); });
      if (history.replaceState) history.replaceState(null, "", "#" + name);
    }
    btns.forEach(function (b) { b.addEventListener("click", function () { show(b.getAttribute("data-tab")); }); });
    function fromHash() { var h = location.hash.replace("#", "").split("&")[0]; if (h && $("tab-" + h)) show(h); }
    window.addEventListener("hashchange", fromHash);
    fromHash();
  }
  function loadAll() {
    return Promise.all([getJSON("/api/lab/reference").catch(function () { return { accuracy: {}, tactics: {}, elo: {} }; }), getJSON("/api/lab/runs")])
      .then(function (r) {
        REF = r[0]; RUNS = r[1];
        renderRunSelect();
        if (!RUNS.length) { emptyTraining(); loadHyp(null); } else { loadRun($("runSel").value); loadHyp($("hypSel").value); }
      }).catch(function (e) { $("trainBody").innerHTML = '<div class="empty">Could not reach the server: ' + esc(e.message) + "</div>"; });
  }
  tabs();
  $("runSel").addEventListener("change", function () { loadRun(this.value); });
  $("hypSel").addEventListener("change", function () { loadHyp(this.value); });
  $("arenaSel").addEventListener("change", function () { loadArena(this.value); });
  $("reload").addEventListener("click", function () { CACHE = {}; loadAll(); loadArenaList(); });
  loadAll();
  loadArenaList();
})();
