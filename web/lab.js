/* Connect 4 Lab page: fetches /api/lab/* and draws the charts as inline SVG
   (no external libraries, works offline). */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var C = { blue: "#56B4E9", orange: "#E69F00", green: "#009E73", yellow: "#F0E442", red: "#D55E00",
            purple: "#CC79A7", sky: "#0072B2", grey: "#8e96b5", pink: "#ef476f", gold: "#ffd166", white: "#e9ecf8" };
  var SEQ = [C.blue, C.orange, C.green, C.pink, C.purple, C.gold, C.red, C.sky, C.grey];
  var REF = null, RUNS = [], ARENAS = [], CACHE = {};
  // training methods: label, what one point on the x axis is
  var PHASE = {
    supervised: { label: "Supervised (perfect-play labels)", short: "supervised", x: "epoch", unit: "epochs" },
    selfplay: { label: "AlphaZero self-play", short: "self-play", x: "generation", unit: "gens" },
    evolution: { label: "Evolution (CMA-ES) of the heuristic weights", short: "evolution", x: "generation", unit: "gens" },
    ntuple: { label: "TD(λ) n-tuple network", short: "n-tuple TD", x: "generation", unit: "gens" },
    dqn: { label: "Deep Q-learning (double DQN)", short: "DQN", x: "iteration", unit: "iters" },
    ppo: { label: "PPO self-play (policy gradient)", short: "PPO", x: "iteration", unit: "iters" },
    distill: { label: "Knowledge distillation", short: "distillation", x: "epoch", unit: "epochs" },
    supervised_score: { label: "Supervised + score head (3M Pons positions)", short: "score head", x: "epoch", unit: "epochs" }
  };
  // label sources ("families") from research/bench_all.json, in its order, with short names and colours
  var FAMILY = [
    ["perfect labels + score head (Pons, 3M)", "score head (3M Pons)", "#ffd166"],
    ["perfect labels (Pons)", "Pons labels", "#56B4E9"],
    ["net teacher (distillation)", "distilled from a net", "#0072B2"],
    ["search teacher (minimax d4)", "minimax-d4 teacher", "#CC79A7"],
    ["heuristic teacher (no search)", "heuristic teacher", "#8e96b5"],
    ["RL-net teacher", "PPO-net teacher", "#E69F00"],
    ["self-play (AlphaZero)", "AlphaZero self-play", "#009E73"],
    ["self-play RL (PPO / DQN)", "PPO / DQN self-play", "#D55E00"],
    ["evolved heuristic (CMA-ES)", "evolution (CMA-ES)", "#F0E442"],
    ["TD(λ) n-tuple", "TD n-tuple", "#ef476f"]];
  function famIdx(f) { for (var i = 0; i < FAMILY.length; i++) if (FAMILY[i][0] === f) return i; return FAMILY.length; }
  function famShort(f) { var i = famIdx(f); return i < FAMILY.length ? FAMILY[i][1] : (f || "?"); }
  function famColor(f) { var i = famIdx(f); return i < FAMILY.length ? FAMILY[i][2] : "#e9ecf8"; }
  function phaseOf(p) { return PHASE[p] || { label: p || "?", short: p || "?", x: "step", unit: "points" }; }
  function isRL(p) { return ["evolution", "ntuple", "dqn", "ppo", "distill"].indexOf(p) >= 0; }
  function kwh(wh) { return wh == null ? "–" : wh >= 1000 ? (wh / 1000).toFixed(2) + " kWh" : wh.toFixed(wh < 10 ? 2 : 0) + " Wh"; }

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
    var W = 640, H = o.height || 280, m = { l: 52, r: 16, t: 12, b: 40 };
    if (!items.length) { var d0 = el("div", { "class": "empty" }, "No data yet."); return d0; }
    var X, s, x0, x1;
    var y0 = o.yMin, y1 = o.yMax;
    var Y = function (v) { return H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b); };
    s = '<svg class="chart" viewBox="0 0 ' + W + " " + H + '"><g class="grid">';
    niceTicks(y0, y1, 5).forEach(function (t) { s += '<line x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(t) + '" y2="' + Y(t) + '"/><text x="' + (m.l - 6) + '" y="' + (Y(t) + 4) + '" text-anchor="end">' + (o.yFmt ? o.yFmt(t) : t) + "</text>"; });
    if (o.linear) {
      var lx = items.map(function (i) { return i.x; });
      x0 = o.xMin != null ? o.xMin : Math.min.apply(null, lx); x1 = o.xMax != null ? o.xMax : Math.max.apply(null, lx);
      var pd = (x1 - x0) * 0.06 || 1; if (o.xMin == null) x0 -= pd; if (o.xMax == null) x1 += pd;
      X = function (v) { return m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r); };
      niceTicks(x0, x1, 6).forEach(function (t) { s += '<line x1="' + X(t) + '" x2="' + X(t) + '" y1="' + m.t + '" y2="' + (H - m.b) + '"/><text x="' + X(t) + '" y="' + (H - m.b + 15) + '" text-anchor="middle">' + (o.xTick ? o.xTick(t) : t) + "</text>"; });
    } else {
      var xs = items.map(function (i) { return Math.log10(Math.max(i.x, 0.01)); });
      x0 = Math.floor(Math.min.apply(null, xs)); x1 = Math.ceil(Math.max.apply(null, xs)); if (x1 === x0) x1++;
      X = function (v) { return m.l + (Math.log10(Math.max(v, 0.01)) - x0) / (x1 - x0) * (W - m.l - m.r); };
      for (var e = x0; e <= x1; e++) { var xv = Math.pow(10, e); s += '<line x1="' + X(xv) + '" x2="' + X(xv) + '" y1="' + m.t + '" y2="' + (H - m.b) + '"/><text x="' + X(xv) + '" y="' + (H - m.b + 15) + '" text-anchor="middle">' + (o.xTick ? o.xTick(xv) : xv >= 1000 ? xv / 1000 + " s" : xv + " ms") + "</text>"; }
    }
    s += "</g>";
    (o.refs || []).forEach(function (r) { if (r.y < y0 || r.y > y1) return; s += '<g class="ref"><line x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(r.y) + '" y2="' + Y(r.y) + '" stroke="' + (r.color || C.grey) + '"/><text x="' + (m.l + 4) + '" y="' + (Y(r.y) - 3) + '" style="fill:' + (r.color || C.grey) + '">' + esc(r.label) + "</text></g>"; });
    items.forEach(function (it) {
      s += '<circle cx="' + X(it.x) + '" cy="' + Y(it.y) + '" r="' + (it.r || 6) + '" fill="' + it.color + '" fill-opacity=".85"' + (it.ring ? ' stroke="#fff" stroke-width="1.5"' : "") + "><title>" + esc(it.label) + "</title></circle>";
      if (!o.noLabels || it.showLabel) s += '<text x="' + (X(it.x) + 9) + '" y="' + (Y(it.y) + 4 + (it.dy || 0)) + '" style="fill:#cfd5ee">' + esc(it.short || it.label) + "</text>";
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

  // ---- training-run filters: model size -> method -> run
  var F = { size: "all", fam: "all", top: false };
  function sizeOf(r) { return r.size || "?"; }
  function sizeList() {
    var by = {};
    RUNS.forEach(function (r) { var k = sizeOf(r); if (!by[k]) by[k] = { size: k, n: 0, params: r.params || 0 }; by[k].n++; });
    return Object.keys(by).map(function (k) { return by[k]; }).sort(function (a, b) { return a.params - b.params; });
  }
  function filteredRuns(ignoreFam) {
    return RUNS.filter(function (r) {
      return (F.size === "all" || sizeOf(r) === F.size) && (ignoreFam || F.fam === "all" || r.family === F.fam) && (!F.top || r.top8);
    });
  }
  function runAcc(r) { return r.bench && r.bench.policy_optimal != null ? r.bench.policy_optimal : r.best_bench_optimal; }
  function runOption(r) {
    var a = runAcc(r);
    return el("option", { value: r.name }, esc(r.name + "  (" + famShort(r.family) + (a ? ", " + pct(a) : "") + (r.top8 ? ", top 8" : "") + ")"));
  }
  function renderFilters() {
    var ss = $("sizeSel"), fs = $("famSel");
    ss.innerHTML = "";
    var base = RUNS.filter(function (r) { return !F.top || r.top8; });
    ss.appendChild(el("option", { value: "all" }, "All sizes (" + base.length + " runs)"));
    sizeList().forEach(function (z) {
      var n = base.filter(function (r) { return sizeOf(r) === z.size; }).length;
      if (n) ss.appendChild(el("option", { value: z.size }, esc(z.size + "  (" + n + (n === 1 ? " run" : " runs") + (z.params ? ", " + (z.params >= 1e6 ? (z.params / 1e6).toFixed(1) + " M" : Math.round(z.params / 1000) + " k") + " params" : "") + ")")));
    });
    if (!Array.prototype.some.call(ss.options, function (o) { return o.value === F.size; })) F.size = "all";
    ss.value = F.size;
    var fams = {};
    filteredRuns(true).forEach(function (r) { fams[r.family] = (fams[r.family] || 0) + 1; });
    fs.innerHTML = "";
    fs.appendChild(el("option", { value: "all" }, "All methods"));
    Object.keys(fams).sort(function (a, b) { return famIdx(a) - famIdx(b); }).forEach(function (f) { fs.appendChild(el("option", { value: f }, esc(famShort(f) + " (" + fams[f] + ")"))); });
    if (!fams[F.fam]) F.fam = "all";
    fs.value = F.fam;
    $("top8Only").checked = F.top;
  }
  function renderRunSelect() {
    renderFilters();
    var sel = $("runSel"), prev = sel.value;
    sel.innerHTML = "";
    var list = filteredRuns();
    if (F.size === "all") {      // group the menu by model size
      sizeList().forEach(function (z) {
        var rs = list.filter(function (r) { return sizeOf(r) === z.size; });
        if (!rs.length) return;
        var g = el("optgroup", { label: z.size });
        rs.sort(function (a, b) { return (runAcc(b) || 0) - (runAcc(a) || 0); }).forEach(function (r) { g.appendChild(runOption(r)); });
        sel.appendChild(g);
      });
    } else {
      list.sort(function (a, b) { return famIdx(a.family) - famIdx(b.family) || (runAcc(b) || 0) - (runAcc(a) || 0); }).forEach(function (r) { sel.appendChild(runOption(r)); });
    }
    var want = decodeURIComponent((location.hash.match(/run=([^&]+)/) || [])[1] || "");
    if (want && list.some(function (r) { return r.name === want; })) sel.value = want;
    else if (prev && list.some(function (r) { return r.name === prev; })) sel.value = prev;
    else if (list.length) sel.value = list.slice().sort(function (a, b) { return (b.created || "") < (a.created || "") ? -1 : 1; })[0].name;   // newest run
    $("runInfo").textContent = "";
    renderGroup(list);
    return sel.value;
  }
  function onFilter() {
    var before = $("runSel").value, now = renderRunSelect();
    if (now !== before || !$("trainBody").children.length) loadRun(now);
    else if (CACHE[now]) renderRun(CACHE[now]);
  }

  // ranked horizontal bars as HTML (clickable rows)
  function rankList(items, o) {
    o = o || {};
    var lo = o.min != null ? o.min : 0, hi = o.max != null ? o.max : Math.max.apply(null, items.map(function (i) { return i.value || 0; }).concat([1e-9]));
    var h = '<div class="rank">';
    items.forEach(function (it) {
      var w = it.value == null ? 0 : Math.max(0, Math.min(1, (it.value - lo) / (hi - lo || 1)));
      h += '<div class="rrow' + (it.key ? " click" : "") + (it.on ? " on" : "") + '"' + (it.key ? ' data-key="' + esc(it.key) + '"' : "") + ' title="' + esc(it.title || it.label) + '">' +
        '<span class="rl">' + esc(it.label) + (it.badge ? ' <span class="badge ours">' + esc(it.badge) + "</span>" : "") + "</span>" +
        '<span class="rt"><span class="rf" style="width:' + (100 * w).toFixed(1) + "%;background:" + (it.color || C.blue) + '"></span></span>' +
        '<span class="rv">' + (it.text != null ? it.text : (o.fmt || pct)(it.value)) + "</span></div>";
    });
    var d = el("div", null, h + "</div>");
    if (o.onClick) d.addEventListener("click", function (e) { var r = e.target.closest(".rrow.click"); if (r) o.onClick(r.getAttribute("data-key")); });
    return d;
  }

  // overview of the runs the filters selected (above the single run)
  function renderGroup(list) {
    var gb = $("groupBody"); gb.innerHTML = "";
    if (F.size === "all" && F.fam === "all" && !F.top) return;
    var title = (F.size === "all" ? "All sizes" : F.size) + (F.fam === "all" ? "" : " · " + famShort(F.fam)) + (F.top ? " · top-8 arena nets" : "");
    var withAcc = list.filter(function (r) { return runAcc(r) != null; }).sort(function (a, b) { return runAcc(b) - runAcc(a); });
    var lo = Math.min.apply(null, withAcc.map(runAcc).concat([1])), cur = $("runSel").value;
    var g = el("div", { "class": "grid g2", style: "margin-bottom:14px" });
    g.appendChild(card(title + ": " + list.length + (list.length === 1 ? " run" : " runs"),
      "Optimal-move rate on the same 3,000 held-out positions (research/bench_all.json, policy head, no search; nets not in that file use their best training-time benchmark). Click a row to open the run.",
      withAcc.length ? rankList(withAcc.map(function (r) {
        return { key: r.name, label: r.name, value: runAcc(r), color: famColor(r.family), on: r.name === cur, badge: r.top8 ? "top 8" : null,
                 title: r.name + " · " + famShort(r.family) + (r.teacher ? " · " + r.teacher : "") + " · " + (r.size || "") };
      }), { min: Math.max(0, Math.floor((lo - 0.05) * 20) / 20), max: 1, onClick: function (k) { $("runSel").value = k; loadRun(k); renderGroup(filteredRuns()); } })
        : el("p", { "class": "muted" }, "No accuracy numbers for these runs.")));
    // per method at this size (or per size for one method)
    var by = {}, keyOf = F.size === "all" ? sizeOf : function (r) { return r.family; };
    withAcc.forEach(function (r) { var k = keyOf(r); if (!by[k] || runAcc(r) > runAcc(by[k])) by[k] = r; });
    var keys = Object.keys(by).sort(function (a, b) { return runAcc(by[b]) - runAcc(by[a]); });
    if (keys.length > 1)
      g.appendChild(card(F.size === "all" ? "Best run per size" : "Best run per method at " + F.size,
        F.size === "all" ? "How accuracy grows with the network (same label source if a method is selected)." : "Which training signal works best when the network size is held fixed.",
        rankList(keys.map(function (k) { var r = by[k]; return { key: r.name, label: (F.size === "all" ? k : famShort(k)) + " · " + r.name, value: runAcc(r), color: famColor(r.family) }; }),
          { min: Math.max(0, Math.floor((lo - 0.05) * 20) / 20), max: 1, onClick: function (k) { $("runSel").value = k; loadRun(k); renderGroup(filteredRuns()); } })));
    gb.appendChild(g);
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
    var mc = cfg.model || {};
    var modelDesc = mc.kind === "n-tuple" ? mc.tuples + " tuples × " + mc.length + " cells, " + fmtInt(cfg.params) + " table weights"
      : mc.kind === "weighted heuristic" ? (mc.features || []).length + " heuristic weights, searched at minimax depth " + mc.search_depth
      : (mc.blocks || "?") + "×" + (mc.channels || "?") + " ResNet, " + fmtInt(cfg.params) + " params";
    var rs = RUNS.filter(function (r) { return r.name === d.name; })[0] || {};
    $("runInfo").textContent = phaseOf(cfg.phase).label + (rs.teacher ? " (" + rs.teacher + ")" : "") + " · " + modelDesc + " · " + (cfg.device || "") + " · started " + (cfg.created || "").replace("T", " ") + (rs.top8 ? " · in the top-8 arena" : "");
    if (isRL(cfg.phase)) return renderRL(d, body);
    var xk = sp ? "gen" : "epoch", xl = sp ? "generation" : "epoch";
    var last = M[M.length - 1] || {};
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
    } else if (cfg.phase === "supervised_score") {
      var heads = (M.filter(function (r) { return r.bench_by_head_mirror; }).slice(-1)[0] || {});
      var hk = Object.keys(heads.bench_by_head_mirror || {});
      g.appendChild(card("Score and rank losses", "The extra head predicts the exact Pons score of all 7 columns: smooth-L1 on the score and a listwise loss on their order (faster wins rank above slower ones).",
        lineChart({ series: [{ name: "score loss", points: pts("train_score_loss"), color: C.gold }, { name: "rank loss", points: pts("train_rank_loss"), color: C.purple }], xLabel: xl, intX: true, yFmt: function (v) { return v.toFixed(2); } })));
      g.appendChild(card("Validation: optimal move per head", "Validation positions (plies 8-30, never the benchmark). The best head is the one the website, arena and MCTS play with.",
        lineChart({ series: ["policy", "score", "combo1"].map(function (h, i) { return { name: h, color: [C.blue, C.gold, C.green][i], points: M.filter(function (r) { return r.val_optimal_by_head && r.val_optimal_by_head[h] != null; }).map(function (r) { return [r[xk], r.val_optimal_by_head[h]]; }) }; }),
                    yFmt: function (v) { return (v * 100).toFixed(1) + "%"; }, xLabel: xl, intX: true })));
      if (hk.length) g.appendChild(card("Benchmark by head (latest epoch)", "Optimal-move rate on the 3,000 held-out positions for every way of reading the net, with and without left-right mirror averaging. combo w = score + w·log(policy). Chosen head: <b>" + esc(heads.chosen_head || "?") + "</b>.",
        groupedBars({ categories: hk, series: [{ name: "mirror-averaged", values: hk.map(function (h) { return heads.bench_by_head_mirror[h].optimal; }), color: C.gold },
                                              { name: "no mirror", values: hk.map(function (h) { return (heads.bench_by_head_no_mirror || {})[h] ? heads.bench_by_head_no_mirror[h].optimal : null; }), color: C.blue }],
                      yMin: 0.95, yMax: 0.98, yFmt: function (v) { return (v * 100).toFixed(1) + "%"; } })));
    } else {
      g.appendChild(card("Validation accuracy", "On the held-out 5% of the generated data: is the policy's top move optimal, and is the value's call right?",
        lineChart({ series: [{ name: "policy top move optimal", points: pts("val_policy_acc"), color: C.blue }, { name: "value correct", points: pts("val_value_acc"), color: C.green }],
                    yMin: 0, yMax: 1, yFmt: function (v) { return Math.round(v * 100) + "%"; }, xLabel: xl, intX: true })));
    }

    if (ev) body.appendChild(renderEval(ev));
    if (d.notes) { var n = el("div", { "class": "grid", style: "margin-top:14px" }); var nd = el("div", { "class": "notes" }); nd.textContent = d.notes; n.appendChild(card("Notes (runs/" + d.name + "/notes.md)", null, nd, true)); body.appendChild(n); }
    body.appendChild(renderRunsOverview());
  }

  // ======================================================= other methods (rl/)
  var METHOD_COLOR = { supervised: C.blue, selfplay: C.gold, evolution: C.green, ntuple: C.purple,
                       dqn: C.red, ppo: C.orange, distill: C.sky, hand: C.grey, custom: C.white };
  var PCT = function (v) { return Math.round(v * 100) + "%"; };

  function renderRL(d, body) {
    var cfg = d.config || {}, M = d.metrics || [], ph = cfg.phase, P = phaseOf(ph);
    var xk = "gen", xl = P.x;
    var last = M[M.length - 1] || {};
    var cost = d.cost && d.cost.total;
    var pts = function (k, f) { return M.filter(function (r) { return r[k] != null; }).map(function (r) { return [r[xk], f ? f(r[k]) : r[k]]; }); };
    var best = M.reduce(function (a, r) { return Math.max(a, r.bench_optimal || 0); }, 0);
    var elos = M.filter(function (r) { return r.elo != null; });
    var what = { evolution: "minimax d" + ((cfg.model || {}).search_depth || "?") + " with the evolved weights", ntuple: "greedy, no search",
                 dqn: "argmax Q, no search", ppo: "policy, no search", distill: "student policy, no search" }[ph] || "";
    var t = '<div class="tiles">';
    t += tile(P.x.charAt(0).toUpperCase() + P.x.slice(1) + "s", fmtInt(last[xk]), last.total_games ? fmtInt(last.total_games) + " games" : last.positions_seen ? fmtInt(last.positions_seen) + " positions seen" : "");
    t += tile("Training time", dur(cost ? cost.wall_sec : last.elapsed_sec), cost ? num(cost.cpu_core_hours, 1) + " CPU core-h · " + num(cost.gpu_hours, 2) + " GPU-h" : "");
    if (cost) t += tile("Energy", kwh(cost.energy_wh), "GPU " + kwh(cost.gpu_wh) + " + CPU " + kwh(cost.cpu_wh) + (cost.cpu_wh_source === "estimate" ? " (est.)" : "") + " · $" + num(cost.energy_cost_usd, 2));
    t += tile("Best move accuracy", pct(best), what);
    if (last.bench_value_acc != null && ph !== "evolution") t += tile("Value accuracy", pct(last.bench_value_acc), "win/draw/loss vs solver");
    if (ph === "distill") t += tile("Agrees with teacher", pct(last.val_teacher_agreement), "student " + num(last.student_ms, 2) + " ms vs teacher " + Object.values(last.teacher_ms || {}).map(function (v) { return num(v, 1); }).join(" / ") + " ms");
    if (elos.length) t += tile("Elo (latest)", num(elos[elos.length - 1].elo), "minimax d1 = 1000");
    if (ph === "evolution" && last.weights) t += tile("Step size σ", num(last.sigma, 3), "CMA-ES search radius");
    body.appendChild(el("div", null, t + "</div>"));

    var g = el("div", { "class": "grid g2" });
    body.appendChild(g);
    g.appendChild(card("Accuracy vs perfect play", "Share of held-out positions (" + (last.bench_positions || "3,000") + ") where it picks a move the perfect solver also rates best (" + what + "). Dashed: our hand-crafted minimax at depth 1/4/8.",
      lineChart({ series: [{ name: "optimal move", points: pts("bench_optimal"), color: C.blue, dots: ph === "evolution" },
                           { name: "non-blunder rate", points: pts("bench_keeps_nl"), color: C.green, dash: true, dots: ph === "evolution" }],
                  refs: accRefs(), yMin: 0, yMax: 1, yFmt: PCT, xLabel: xl, intX: true })));
    if (elos.length) {
      var ladderRefs = [];
      if (REF && REF.elo) ["d1", "d2", "d4", "d6", "d8"].forEach(function (k) { if (REF.elo[k]) ladderRefs.push({ y: REF.elo[k].elo, label: "minimax " + k, color: C.grey }); });
      g.appendChild(card("Strength (Elo) over training", "From games against our minimax at fixed depths (depth 1 = 1000). Dashed lines: minimax depths.",
        lineChart({ series: [{ name: what, points: pts("elo"), color: C.gold, dots: true }], refs: ladderRefs, xLabel: xl, intX: true })));
      var depthKeys = {};
      M.forEach(function (r) { if (r.vs_minimax) Object.keys(r.vs_minimax).forEach(function (k) { depthKeys[k] = 1; }); });
      g.appendChild(card("Score vs minimax", "Share of points (draw = ½) against minimax at each depth.",
        lineChart({ series: Object.keys(depthKeys).sort().map(function (k, i) {
          return { name: "vs " + k.replace("d", "depth "), points: M.filter(function (r) { return r.vs_minimax && r.vs_minimax[k]; }).map(function (r) { return [r[xk], r.vs_minimax[k].score]; }), color: SEQ[i], dots: true };
        }), refs: [{ y: 0.5, label: "even", color: C.grey }], yMin: 0, yMax: 1, yFmt: PCT, xLabel: xl, intX: true })));
    }
    var f2 = function (v) { return v.toFixed(2); };
    if (ph === "evolution") {
      g.appendChild(card("Fitness per generation", "Score of the candidates (games vs the hand-crafted minimax and vs the current mean; or accuracy). Best / mean / worst of each generation's population.",
        lineChart({ series: [{ name: "best", points: pts("fitness_best"), color: C.green }, { name: "mean", points: pts("fitness_mean"), color: C.blue }, { name: "worst", points: pts("fitness_worst"), color: C.red, dash: true }], xLabel: xl, intX: true, yFmt: f2 })));
      var names = last.weights ? Object.keys(last.weights) : [];
      g.appendChild(card("Weights over generations", "The mean of the search distribution - the current answer. Hand-crafted start: centre 3, three 5, two 2, opponent three −4, opponent two −2, the rest 0.",
        lineChart({ series: names.map(function (n, i) { return { name: n, color: SEQ[i % SEQ.length], dash: i >= SEQ.length, points: M.filter(function (r) { return r.weights; }).map(function (r) { return [r[xk], r.weights[n]]; }) }; }), xLabel: xl, intX: true, yFmt: f2 })));
      g.appendChild(card("Step size σ", "How far CMA-ES samples from the mean. Shrinks as it converges.",
        lineChart({ series: [{ name: "sigma", points: pts("sigma"), color: C.purple }], xLabel: xl, intX: true, yFmt: f2 })));
    } else if (ph === "ntuple") {
      g.appendChild(card("TD error", "Average |λ-return − value| per position: how surprised the table still is by what happens next.",
        lineChart({ series: [{ name: "TD error", points: pts("td_error"), color: C.purple }], xLabel: xl, intX: true, yFmt: function (v) { return v.toFixed(3); } })));
      g.appendChild(card("Speed and exploration", "Self-play games per second (CPU only) and the share of random moves.",
        lineChart({ series: [{ name: "games / s", points: pts("games_per_sec"), color: C.blue }], xLabel: xl, intX: true })));
    } else if (ph === "dqn") {
      g.appendChild(card("Training loss", "Huber loss of Q(s,a) against the Bellman target, and cross-entropy of the side value head against game results.",
        lineChart({ series: [{ name: "Q loss", points: pts("q_loss"), color: C.red }, { name: "value loss", points: pts("value_loss"), color: C.orange }], xLabel: xl, intX: true, yFmt: function (v) { return v.toFixed(3); } })));
      g.appendChild(card("Q values and exploration", "Average |Q| of the moves trained on, and epsilon (share of random moves).",
        lineChart({ series: [{ name: "mean |Q|", points: pts("q_abs_mean"), color: C.red }, { name: "epsilon", points: pts("eps"), color: C.grey, dash: true }], yMin: 0, yMax: 1, xLabel: xl, intX: true, yFmt: f2 })));
    } else if (ph === "ppo") {
      g.appendChild(card("Training loss", "Clipped policy loss, value cross-entropy, and the policy's entropy (how spread out its move choice is).",
        lineChart({ series: [{ name: "policy loss", points: pts("policy_loss"), color: C.orange }, { name: "value loss", points: pts("value_loss"), color: C.blue }, { name: "entropy", points: pts("entropy"), color: C.grey, dash: true }], xLabel: xl, intX: true, yFmt: f2 })));
      g.appendChild(card("Update size", "Approximate KL divergence between the old and new policy, and the share of moves whose probability ratio hit the clip. Small and steady = healthy PPO.",
        lineChart({ series: [{ name: "approx KL", points: pts("approx_kl"), color: C.purple }, { name: "clip fraction", points: pts("clip_frac"), color: C.pink }], xLabel: xl, intX: true, yFmt: function (v) { return v.toFixed(3); } })));
      if (M.some(function (r) { return r.score_vs_pool != null; }))
        g.appendChild(card("Score vs past selves", "Games against frozen snapshots from the opponent pool. Staying above 50% = it keeps beating its older versions.",
          lineChart({ series: [{ name: "score vs pool", points: pts("score_vs_pool"), color: C.gold, dots: true }], refs: [{ y: 0.5, label: "even", color: C.grey }], yMin: 0, yMax: 1, yFmt: PCT, xLabel: xl, intX: true })));
    } else if (ph === "distill") {
      g.appendChild(card("Distillation loss", "KL divergence between the teacher's and the student's move distribution (train / validation), and the value cross-entropy.",
        lineChart({ series: [{ name: "train KL", points: pts("policy_loss"), color: C.sky }, { name: "val KL", points: pts("val_policy_kl"), color: C.sky, dash: true }, { name: "value loss", points: pts("value_loss"), color: C.orange }], xLabel: xl, intX: true, yFmt: function (v) { return v.toFixed(3); } })));
      g.appendChild(card("Agreement with the teacher", "Validation positions where the student's top move is the teacher's top move.",
        lineChart({ series: [{ name: "same top move", points: pts("val_teacher_agreement"), color: C.green }], yMin: 0, yMax: 1, yFmt: PCT, xLabel: xl, intX: true })));
    }
    if (M.some(function (r) { return r.p1_win != null; }))
      g.appendChild(card("Training games", "Who wins the games the method learns from, and how long they are.",
        lineChart({ series: [{ name: "first player wins", points: pts("p1_win"), color: C.pink }, { name: "draws", points: pts("draw"), color: C.grey }, { name: "second player wins", points: pts("p2_win"), color: C.gold }],
                    yMin: 0, yMax: 1, yFmt: PCT, xLabel: xl, intX: true })));
    if (ph !== "evolution" && M.some(function (r) { return r.bench_value_acc != null; }))
      g.appendChild(card("Value accuracy", "Win/draw/loss call vs the solver on the held-out positions. (Always 'win' = 63%.)",
        lineChart({ series: [{ name: "value", points: pts("bench_value_acc"), color: C.green }], refs: [{ y: 0.6333, label: "always 'win'", color: C.grey }], yMin: 0, yMax: 1, yFmt: PCT, xLabel: xl, intX: true })));
    if (M.some(function (r) { return r.energy_wh != null; }))
      g.appendChild(card("Compute spent so far", "Cumulative energy (Wh, GPU + CPU) and CPU core-hours, measured by rl/meter.py while training.",
        lineChart({ series: [{ name: "energy (Wh)", points: pts("energy_wh"), color: C.gold }, { name: "CPU core-hours", points: pts("cpu_core_sec", function (v) { return v / 3600; }), color: C.blue }], xLabel: xl, intX: true, yFmt: function (v) { return String(Number(v.toPrecision(2))); } })));
    var lb = M.filter(function (r) { return r.bench_by_bucket; }).slice(-1)[0];
    if (lb) {
      var bk = ["8-11", "12-15", "16-19", "20-23", "24-27", "28-30"];
      var ser = [{ name: P.short, values: bk.map(function (b) { return (lb.bench_by_bucket[b] || {}).optimal; }), color: METHOD_COLOR[ph] || C.blue }];
      [4, 8].forEach(function (dd, i) { if (refAcc(dd)) ser.push({ name: "minimax depth " + dd, values: bk.map(function (b) { return (refAcc(dd, b) || {}).optimal; }), color: i ? C.white : C.purple }); });
      g.appendChild(card("Accuracy by game phase", "Latest " + xl + ", by stones on the board.",
        groupedBars({ categories: bk, series: ser, yMin: 0, yMax: 1, yFmt: PCT, xLabel: "stones on the board" })));
    }
    if (d.eval) body.appendChild(renderEval(d.eval));
    if (d.notes) { var n = el("div", { "class": "grid", style: "margin-top:14px" }); var nd = el("div", { "class": "notes" }); nd.textContent = d.notes; n.appendChild(card("Notes (runs/" + d.name + "/notes.md)", null, nd, true)); body.appendChild(n); }
    body.appendChild(renderRunsOverview());
  }

  // ===================================================== method comparison
  var COMPARE = null;
  function loadMethodsList() {
    return Promise.all([getJSON("/api/lab/methods").catch(function () { return []; }), getJSON("/api/lab/compare").catch(function () { return null; })]).then(function (r) {
      var list = r[0]; COMPARE = r[1];
      var s = $("methSel"), prev = s.value; s.innerHTML = "";
      if (COMPARE && COMPARE.players.length) s.appendChild(el("option", { value: "top8" }, esc("Top-8 nets: arena + benchmark + cost (" + COMPARE.arena.name + ")")));
      if (COMPARE && COMPARE.nets.length) {
        s.appendChild(el("option", { value: "sizes" }, esc("By model size: every net, label source × size (" + COMPARE.nets.length + " nets)")));
        s.appendChild(el("option", { value: "bench" }, esc("Every net on the 3,000-position benchmark (" + COMPARE.nets.length + " nets)")));
      }
      if (list.length) {
        var g = el("optgroup", { label: "Minimax-ladder comparisons (rl.compare)" });
        list.forEach(function (m) { g.appendChild(el("option", { value: "ladder:" + m.name }, esc(m.name + " (" + m.entries + " entries" + (m.created ? ", " + m.created.slice(0, 10) : "") + ")"))); });
        s.appendChild(g);
      }
      if (!s.options.length) {
        $("methBody").innerHTML = '<div class="empty"><b>No method comparison yet.</b> After training some of the methods (docs/OTHER_METHODS.md), run<pre>python -m rl.compare</pre>It plays every trained method against the minimax ladder, measures accuracy and speed, and reads each run\'s training cost.</div>';
        return;
      }
      if (prev && Array.prototype.some.call(s.options, function (o) { return o.value === prev; })) s.value = prev;
      loadMethods(s.value);
    }).catch(function () {});
  }
  function loadMethods(name) {
    if (name === "top8") return renderTop8(COMPARE);
    if (name === "bench") return renderBench(COMPARE);
    if (name === "sizes") return renderSizes(COMPARE);
    var n = name.replace(/^ladder:/, "");
    getJSON("/api/lab/methods/" + encodeURIComponent(n)).then(renderMethods)
      .catch(function (e) { $("methBody").innerHTML = '<div class="empty">' + esc(e.message) + "</div>"; });
  }

  // ---- top-8 view: one row per arena player, everything we know about it
  function shortBot(p) {
    if (p.bot.indexOf("pons") === 0) return "Pons solver";
    if (p.bot.indexOf("bruneton") === 0) return "Bruneton AlphaZero";
    var m = p.mode === "policy" ? "policy" : "+ " + (p.mode || "").replace("mcts", "MCTS ");
    return (p.runs.length > 1 ? "ens3 score 8x128" : p.runs[0]) + " " + m;
  }
  function playerColor(p) { return p.group !== "ours" ? C.blue : p.runs.length > 1 ? "#ffffff" : famColor(p.family); }
  function playerAcc(p) {
    if (p.ensemble_bench) return p.ensemble_bench.best.optimal;
    return p.bench && p.bench.length ? p.bench[0].played_optimal || p.bench[0].policy_optimal : null;
  }
  function ladderFor(p) {
    if (!COMPARE || !p.runs.length) return null;
    var E = COMPARE.ladder.entries || [];
    var bySpec = E.filter(function (e) {      // exact model file + mode (entries added with rl.compare --add)
      var sp = e.spec || [];
      return sp[0] === "nn" && sp[1] === p.model && (sp[2] === "policy" ? p.mode === "policy" : sp[2] === "mcts" && "mcts" + sp[3] === p.mode);
    })[0];
    if (bySpec || p.runs.length > 1) return bySpec || null;
    var want = p.mode === "policy" ? "none" : p.mode;
    return E.filter(function (e) { return e.run === p.runs[0] && e.search === want; })[0] || null;
  }
  function renderTop8(D) {
    var body = $("methBody"); body.innerHTML = "";
    if (!D || !D.players.length) { body.innerHTML = '<div class="empty">No top-8 arena run found (experiments/results/arena/top8_*).</div>'; return; }
    var A = D.arena, P = D.players.slice().sort(function (a, b) { return ((b.elo || {}).elo || 0) - ((a.elo || {}).elo || 0); });
    $("methInfo").textContent = "Arena " + A.name + " · " + fmtInt(A.games) + " games · " + A.openings + " openings × 2 colours per pairing · Elo anchor " + (A.anchor || "") + " · benchmark: 3,000 held-out positions · " + ((A.meta || {}).created || "").replace("T", " ");
    var ours = P.filter(function (p) { return p.group === "ours"; });
    var pons = P.filter(function (p) { return p.bot.indexOf("pons") === 0; })[0];
    var best = function (arr, f, hi) { return arr.filter(function (p) { return f(p) != null; }).sort(function (a, b) { return hi ? f(b) - f(a) : f(a) - f(b); })[0]; };
    var bElo = best(ours, function (p) { return (p.elo || {}).elo; }, true), bPons = best(ours, function (p) { return p.vs_pons && p.vs_pons.score; }, true),
        bAcc = best(ours, playerAcc, true), bMs = best(ours, function (p) { return p.avg_ms; }), bCost = best(ours, function (p) { return p.cost && p.cost.sec ? p.cost.sec : null; });
    body.appendChild(el("div", { "class": "tiles" },
      tile("Strongest (Elo)", esc(shortBot(bElo)), num(bElo.elo.elo) + (pons && pons.elo ? " vs Pons " + num(pons.elo.elo) : "")) +
      tile("Best vs Pons", esc(shortBot(bPons)), pct(bPons.vs_pons.score) + " (" + bPons.vs_pons.w + "-" + bPons.vs_pons.d + "-" + bPons.vs_pons.l + ")") +
      tile("Most accurate", esc(shortBot(bAcc)), pct(playerAcc(bAcc)) + " optimal moves") +
      tile("Fastest", esc(shortBot(bMs)), num(bMs.avg_ms, 1) + " ms / move") +
      tile("Cheapest to train", esc(shortBot(bCost)), dur(bCost.cost.sec) + " incl. its parents")));
    var g = el("div", { "class": "grid g2" }); body.appendChild(g);
    var rows = P.map(function (p) {
      var e = p.elo || {}, vp = p.vs_pons, b = (p.bench || [])[0] || {}, ev = D.evals[p.runs.join("+")], lad = ladderFor(p), acc = playerAcc(p);
      var accTxt = acc == null ? "–" : pct(acc) + (p.ensemble_bench ? ' <span class="muted">(' + esc(p.ensemble_bench.best_head) + ")</span>" : "");
      return { cls: p.group === "ours" ? "ours" : "", cells: ['<span style="color:' + playerColor(p) + '">●</span> ' + esc(shortBot(p)),
        p.group === "ours" ? esc(famShort(p.family)) : '<span class="muted">' + esc(p.group || "") + "</span>",
        p.size ? p.size + (p.runs.length > 1 ? " ×3" : "") : "–", p.params ? fmtInt(p.params * p.runs.length) : "–",
        e.elo != null ? num(e.elo) + ' <span class="muted">[' + num(e.lo) + ", " + num(e.hi) + "]</span>" : "–", pct(p.score),
        vp ? pct(vp.score) + ' <span class="muted">' + vp.w + "-" + vp.d + "-" + vp.l + "</span>" : "–",
        accTxt, b.value_acc != null ? pct(b.value_acc) : "–", b.value_acc_D != null ? pct(b.value_acc_D, 0) : "–",
        num(p.avg_ms, p.avg_ms < 10 ? 1 : 0), p.cost && p.cost.sec ? dur(p.cost.sec) : "–",
        p.group !== "ours" ? "–" : ev ? (ev.rows.every(function (r) { return r.supported; }) ? '<span class="yes">all depths</span>' : '<span class="no">' + ev.rows.filter(function (r) { return r.supported; }).length + "/" + ev.rows.length + "</span>") : '<span class="muted">not run</span>',
        lad ? num(lad.elo) : "–"] };
    });
    g.appendChild(card("The top-8 nets side by side", "Arena Elo: Bradley-Terry over all " + fmtInt(A.games) + " games, 95% bootstrap interval. vs Pons: score against the perfect solver (W-D-L). Optimal / value / draws: the 3,000 held-out positions with no search (ensemble optimal: the 3 nets averaged, best head, from nn.eval_score; its value columns are those of the first member). ms/move: arena average on one CPU thread. Training time includes the runs a net was initialised or distilled from. Hypothesis: nn.evaluate (Hypothesis test tab). Ladder Elo: rl.compare " + esc(D.ladder.name) + " (minimax depth 1 = 1000, a different scale).",
      table(["player", "method", "size", "params", "arena Elo [95% CI]", "score", "vs Pons", "optimal", "value acc", "draws", "ms/move", "training", "hypothesis", "ladder Elo"], rows), true));
    var items = P.filter(function (p) { return p.elo; }).map(function (p) { return { label: shortBot(p), value: p.elo.elo, lo: p.elo.lo, hi: p.elo.hi, color: p.group === "ours" ? (p.runs.length > 1 ? "#ffffff" : famColor(p.family)) : C.blue, labelColor: p.group === "ours" ? "#e9ecf8" : "#9fd0ff" }; });
    g.appendChild(card("Arena Elo with 95% intervals", "Blue = not ours (Pons solver, Bruneton AlphaZero). Overlapping intervals = not clearly different. The two score-net players sit level with Pons on Elo but still lose to it head to head (next chart): Elo also counts how hard they punish the rest of the field.", hBars(items, { xLabel: "Elo" })));
    var vpItems = P.filter(function (p) { return p.vs_pons; }).sort(function (a, b) { return b.vs_pons.score - a.vs_pons.score; }).map(function (p) {
      return { label: shortBot(p), value: p.vs_pons.score, color: playerColor(p), text: pct(p.vs_pons.score) + " · " + p.vs_pons.w + "-" + p.vs_pons.d + "-" + p.vs_pons.l };
    });
    var vpCard = card("Head to head against the perfect solver", "Share of points each bot took from Pons (draw = ½). 50% would mean playing it even; with balanced openings a perfect opponent can only be drawn or beaten by its own mistakes, and it makes none.", rankList(vpItems, { min: 0, max: 0.6 }));
    g.appendChild(vpCard);
    var lo = Math.min.apply(null, items.map(function (i) { return i.value; })), hi = Math.max.apply(null, items.map(function (i) { return i.value; }));
    var yMin = Math.floor((lo - 60) / 100) * 100, yMax = Math.ceil((hi + 60) / 100) * 100;
    var dot = function (p, x) { return { x: x, y: p.elo.elo, label: shortBot(p) + ": Elo " + num(p.elo.elo), short: shortBot(p), color: playerColor(p) }; };
    g.appendChild(card("Strength vs training time", "Hours of training, including the parent runs (log scale; not counting the time to solve the training positions with Pons). AlphaZero self-play is the most expensive and the distilled 2x32 inherits its cost; solver-labelled training buys by far the most strength per hour.",
      scatter(ours.filter(function (p) { return p.elo && p.cost && p.cost.sec; }).map(function (p) { return dot(p, p.cost.sec / 3600); }),
        { yMin: yMin, yMax: yMax, yFmt: function (v) { return Math.round(v); }, xLabel: "training time (h, incl. parents)", yLabel: "arena Elo", xTick: function (v) { return v < 1 ? Math.round(v * 60) + " min" : v + " h"; } })));
    g.appendChild(card("Strength vs thinking time", "Arena milliseconds per move (log scale). MCTS 200 costs ~100-200× the time of a single forward pass; the score nets pay for it with the biggest Elo jump.",
      scatter(P.filter(function (p) { return p.elo && p.avg_ms; }).map(function (p) { return dot(p, Math.max(p.avg_ms, 0.1)); }),
        { yMin: yMin, yMax: yMax, yFmt: function (v) { return Math.round(v); }, xLabel: "time per move", yLabel: "arena Elo" })));
    var accP = ours.filter(function (p) { return p.elo && playerAcc(p) != null; });
    g.appendChild(card("Accuracy vs strength", "Optimal-move rate without search (x) against arena Elo (y). Ring = played with MCTS 200, so its Elo includes the search.",
      scatter(accP.map(function (p) { var d = dot(p, playerAcc(p)); d.ring = p.mode !== "policy"; return d; }),
        { linear: true, yMin: yMin, yMax: yMax, yFmt: function (v) { return Math.round(v); }, xLabel: "optimal moves, no search (3,000 positions)", yLabel: "arena Elo", xTick: function (v) { return Math.round(v * 100) + "%"; } })));
    // accuracy by phase, one bar per distinct net
    var seen = {}, phases = ["8-15", "16-23", "24-30"], ser = [];
    ours.forEach(function (p) {
      var k = p.runs.join("+"); if (seen[k] || !(p.bench || []).length) return; seen[k] = 1;
      var vals = phases.map(function (ph) { var xs = p.bench.map(function (b) { return (b.policy_optimal_by_ply || {})[ph]; }).filter(function (v) { return v != null; }); return xs.length ? xs.reduce(function (a, b) { return a + b; }, 0) / xs.length : null; });
      ser.push({ name: p.runs.length > 1 ? "score 8x128 (mean of 3)" : p.runs[0], values: vals, color: p.runs.length > 1 ? "#ffffff" : famColor(p.family) });
    });
    if (ser.length) g.appendChild(card("Accuracy by game phase", "Optimal-move rate by stones on the board (policy head, no search). The gap between label sources is largest in the opening, where the result is decided far ahead.",
      groupedBars({ categories: phases.map(function (x) { return x + " stones"; }), series: ser, yMin: 0.4, yMax: 1, yFmt: PCT })));
    var draws = []; seen = {};
    ours.forEach(function (p) { var k = p.runs.join("+"); if (seen[k] || !(p.bench || []).length) return; seen[k] = 1; var b = p.bench[0]; draws.push({ name: p.runs.length > 1 ? "score 8x128 s0_r1" : p.runs[0], vals: [b.value_acc_W, b.value_acc_D, b.value_acc_L], color: p.runs.length > 1 ? "#ffffff" : famColor(p.family) }); });
    if (draws.length) g.appendChild(card("Value head: win / draw / loss", "How often the value head's call is right, split by the true result. Draws (234 of 3,000 positions) are every net's weak spot.",
      groupedBars({ categories: ["true win", "true draw", "true loss"], series: draws.map(function (d) { return { name: d.name, values: d.vals, color: d.color }; }), yMin: 0, yMax: 1, yFmt: PCT })));
  }

  // ---- every net on the benchmark
  function netLabel(n) { return n.run; }
  function renderBench(D) {
    var body = $("methBody"); body.innerHTML = "";
    var N = (D && D.nets || []).slice().sort(function (a, b) { return b.policy_optimal - a.policy_optimal; });
    $("methInfo").textContent = N.length + " nets · " + fmtInt((D.bench_meta || {}).n) + " held-out positions (" + ((D.bench_meta || {}).benchmark || "") + ") · " + ((D.bench_meta || {}).generated || "");
    var fams = {}; N.forEach(function (n) { if (!fams[n.family] || n.policy_optimal > fams[n.family].policy_optimal) fams[n.family] = n; });
    var fk = Object.keys(fams).sort(function (a, b) { return famIdx(a) - famIdx(b); });
    body.appendChild(el("div", { "class": "tiles" },
      tile("Nets", N.length, fk.length + " label sources") +
      tile("Most accurate", esc(N[0].run), pct(N[0].policy_optimal) + " optimal") +
      tile("Best value head", esc(N.slice().sort(function (a, b) { return b.value_acc - a.value_acc; })[0].run), pct(N.slice().sort(function (a, b) { return b.value_acc - a.value_acc; })[0].value_acc)) +
      tile("Smallest above 90%", (function () { var x = N.filter(function (n) { return n.policy_optimal >= 0.9; }).sort(function (a, b) { return a.params - b.params; })[0]; return x ? esc(x.run) : "–"; })(), (function () { var x = N.filter(function (n) { return n.policy_optimal >= 0.9; }).sort(function (a, b) { return a.params - b.params; })[0]; return x ? fmtInt(x.params) + " params" : ""; })())));
    var g = el("div", { "class": "grid g2" }); body.appendChild(g);
    var sc = scatter(N.map(function (n) { return { x: n.params, y: n.policy_optimal, color: famColor(n.family), r: 5, ring: n.top8, label: n.run + " · " + famShort(n.family) + " · " + pct(n.policy_optimal) }; }),
        { noLabels: true, yMin: 0.4, yMax: 1, yFmt: PCT, xLabel: "parameters", yLabel: "optimal moves", xTick: function (v) { return v >= 1e6 ? v / 1e6 + " M" : v >= 1000 ? v / 1000 + " k" : v; } });
    sc.appendChild(legend(fk.map(function (f) { return { name: famShort(f), color: famColor(f) }; })));
    g.appendChild(card("Accuracy vs network size", "Every net: parameters (log scale) against optimal-move rate, coloured by what it learned from. Ringed = in the top-8 arena. Hover a dot for the run. The label source moves accuracy far more than size does.", sc));
    g.appendChild(card("Best net per label source", "Highest optimal-move rate for each training signal (policy head, no search). Value accuracy in the label.",
      rankList(fk.map(function (f) { var n = fams[f]; return { label: famShort(f) + " · " + n.run, value: n.policy_optimal, color: famColor(f), text: pct(n.policy_optimal) + " / " + pct(n.value_acc, 0) }; }).sort(function (a, b) { return b.value - a.value; }), { min: 0.5, max: 1 })));
    g.appendChild(card("Value head by true result (best net per source)", "Policy can be good while the value head is poor: teachers that never saw the true result (minimax, heuristics) give value labels that are only as good as the teacher.",
      groupedBars({ categories: ["true win", "true draw", "true loss"], yMin: 0, yMax: 1, yFmt: PCT,
        series: fk.map(function (f) { var n = fams[f]; return { name: famShort(f), color: famColor(f), values: [n.value_acc_W, n.value_acc_D, n.value_acc_L] }; }) })));
    var rows = N.map(function (n) {
      return { cls: n.top8 ? "ours" : "", cells: [esc(n.run) + (n.top8 ? ' <span class="badge ours">top 8</span>' : ""), '<span style="color:' + famColor(n.family) + '">●</span> ' + esc(famShort(n.family)),
        '<span class="muted">' + esc(n.teacher_label || "") + "</span>", n.blocks + "×" + n.channels, fmtInt(n.params), n.seed != null ? n.seed : "–",
        pct(n.policy_optimal), pct(n.policy_keeps), pct(n.value_acc), pct(n.value_acc_D, 0), n.train_sec ? dur(n.train_sec) : "–"] };
    });
    g.appendChild(card("All nets", "Sorted by optimal-move rate. Keeps = the move does not throw away a won or drawn position. Draws = value-head accuracy on the 234 drawn positions. Training time includes the run a net was initialised from.",
      table(["run", "label source", "teacher", "size", "params", "seed", "optimal", "keeps", "value", "draws", "training"], rows), true));
  }

  // ---- the same nets lumped by size: label source x size grid
  function renderSizes(D) {
    var body = $("methBody"); body.innerHTML = "";
    var N = (D && D.nets || []);
    var sz = {}; N.forEach(function (n) { var k = n.blocks + "x" + n.channels; if (!sz[k]) sz[k] = { k: k, params: n.params }; });
    var sizes = Object.keys(sz).sort(function (a, b) { return sz[a].params - sz[b].params; });
    var fk = {}; N.forEach(function (n) { fk[n.family] = 1; }); fk = Object.keys(fk).sort(function (a, b) { return famIdx(a) - famIdx(b); });
    $("methInfo").textContent = N.length + " nets · " + sizes.length + " sizes · " + fk.length + " label sources · best seed per cell";
    var cell = function (f, k) { var xs = N.filter(function (n) { return n.family === f && n.blocks + "x" + n.channels === k; }); if (!xs.length) return null; return xs.sort(function (a, b) { return b.policy_optimal - a.policy_optimal; })[0]; };
    var g = el("div", { "class": "grid g2" }); body.appendChild(g);
    var h = '<div class="scroll"><table class="t heat"><thead><tr><th>label source \\ size</th>' + sizes.map(function (k) { return "<th>" + k + '<br><span class="muted">' + (sz[k].params >= 1e6 ? (sz[k].params / 1e6).toFixed(1) + "M" : Math.round(sz[k].params / 1000) + "k") + "</span></th>"; }).join("") + "</tr></thead><tbody>";
    fk.forEach(function (f) {
      h += '<tr><td><span style="color:' + famColor(f) + '">●</span> ' + esc(famShort(f)) + "</td>";
      sizes.forEach(function (k) { var c = cell(f, k); h += c ? '<td style="background:' + heatColor(Math.max(0, Math.min(1, (c.policy_optimal - 0.55) / 0.42))) + '" title="' + esc(c.run + ": " + pct(c.policy_optimal) + " optimal, value " + pct(c.value_acc)) + '">' + (100 * c.policy_optimal).toFixed(1) + "</td>" : '<td style="background:#141832"></td>'; });
      h += "</tr>";
    });
    h += "</tbody></table></div>";
    g.appendChild(card("Optimal moves (%) by label source and size", "Each cell: the best net with that training signal at that size (hover for the run). Read across a row to see what size buys; read down a column to see what the teacher buys at a fixed size.", el("div", null, h), true));
    var series = fk.map(function (f) { return { name: famShort(f), color: famColor(f), dots: true, points: sizes.map(function (k, i) { var c = cell(f, k); return [i, c ? c.policy_optimal : null]; }).filter(function (p) { return p[1] != null; }) }; }).filter(function (s) { return s.points.length > 1; });
    g.appendChild(card("Scaling with size", "Optimal-move rate against network size, one line per label source that was trained at more than one size (x = size, small to large).",
      lineChart({ series: series, yMin: 0.5, yMax: 1, yFmt: PCT, xMin: 0, xMax: sizes.length - 1, intX: true, xFmt: function (v) { return sizes[v] || ""; }, xLabel: "network size" })));
    var vseries = fk.map(function (f) { return { name: famShort(f), color: famColor(f), dots: true, points: sizes.map(function (k, i) { var c = cell(f, k); return [i, c ? c.value_acc : null]; }).filter(function (p) { return p[1] != null; }) }; }).filter(function (s) { return s.points.length > 1; });
    g.appendChild(card("Value accuracy with size", "Same nets, value head (win/draw/loss right). Teacher-labelled nets stay low at every size: the ceiling is the teacher, not the network.",
      lineChart({ series: vseries, yMin: 0.6, yMax: 1, yFmt: PCT, xMin: 0, xMax: sizes.length - 1, intX: true, xFmt: function (v) { return sizes[v] || ""; }, xLabel: "network size" })));
  }

  function renderMethods(s) {
    var body = $("methBody"); body.innerHTML = "";
    var E = (s.entries || []).slice().sort(function (a, b) { return (b.elo || 0) - (a.elo || 0); });
    var st = s.settings || {};
    $("methInfo").textContent = E.length + " entries · Elo from " + st.openings + " openings × 2 colours vs minimax depths " + (st.depths || []).join(", ") + " · accuracy on " + fmtInt(st.positions) + " positions · " + (s.created || "").replace("T", " ");
    var short = function (e) { var l = e.label.split(" [")[0]; return l.length > 34 ? l.slice(0, 33) + "…" : l; };
    var withCost = E.filter(function (e) { return e.cost && e.cost.energy_wh != null; });
    var top = E[0] || {};
    var cheapest = withCost.slice().sort(function (a, b) { return a.cost.energy_wh - b.cost.energy_wh; })[0];
    body.appendChild(el("div", { "class": "tiles" },
      tile("Strongest", esc(short(top)), "Elo " + num(top.elo)) +
      tile("Cheapest to train", cheapest ? esc(short(cheapest)) : "–", cheapest ? kwh(cheapest.cost.energy_wh) + ", Elo " + num(cheapest.elo) : "needs cost.json") +
      tile("Fastest per move", esc(short(E.slice().sort(function (a, b) { return a.ms_per_move - b.ms_per_move; })[0] || {})), num((E.slice().sort(function (a, b) { return a.ms_per_move - b.ms_per_move; })[0] || {}).ms_per_move, 2) + " ms")));
    var g = el("div", { "class": "grid g2" }); body.appendChild(g);
    var rows = E.map(function (e) {
      var c = e.cost || {}, inc = c.including_teachers;
      return { cls: e.method === "hand" ? "" : "ours", cells: ['<span style="color:' + (METHOD_COLOR[e.method] || C.white) + '">●</span> ' + esc(e.label), num(e.elo), pct(e.optimal), num(e.ms_per_move, 1),
        c.wall_hours != null ? num(c.wall_hours, 2) : "–", c.gpu_hours != null ? num(c.gpu_hours, 2) : "–", c.cpu_core_hours != null ? num(c.cpu_core_hours, 1) : "–",
        c.energy_wh != null ? kwh(c.energy_wh) + (inc ? ' <span class="muted">(' + kwh(inc.energy_kwh * 1000) + " incl. teachers)</span>" : "") : "–",
        c.energy_cost_usd != null ? "$" + num(c.energy_cost_usd, 2) : "–"] };
    });
    g.appendChild(card("All methods", "Sorted by strength. Training cost from runs/&lt;run&gt;/cost.json (wall time, GPU-hours, CPU core-hours, energy, electricity price). A distilled student's cost is shown alone and including its teachers. ms/move: CPU, numpy, one position at a time (like the website).",
      table(["method", "Elo", "optimal", "ms/move", "train h", "GPU-h", "CPU core-h", "energy", "cost"], rows), true));
    var lo = Math.min.apply(null, E.map(function (e) { return e.elo || 0; })), hi = Math.max.apply(null, E.map(function (e) { return e.elo || 0; }));
    var yMin = Math.floor((lo - 100) / 100) * 100, yMax = Math.ceil((hi + 100) / 100) * 100;
    var dot = function (e, x) { return { x: x, y: e.elo, label: e.label + ": Elo " + num(e.elo), short: short(e), color: METHOD_COLOR[e.method] || C.white }; };
    g.appendChild(card("Strength vs training energy", "Up and to the left = more strength per kWh. Log scale. Methods without a cost.json are missing (train them under rl.meter).",
      scatter(withCost.filter(function (e) { return e.elo != null && e.cost.energy_wh > 0; }).map(function (e) { return dot(e, e.cost.energy_wh); }),
        { yMin: yMin, yMax: yMax, yFmt: function (v) { return Math.round(v); }, xLabel: "training energy (GPU + CPU)", yLabel: "Elo", xTick: function (v) { return v >= 1000 ? v / 1000 + " kWh" : v + " Wh"; } })));
    g.appendChild(card("Strength vs training time", "Wall-clock hours of training (log scale).",
      scatter(E.filter(function (e) { return e.elo != null && e.cost && e.cost.wall_hours > 0; }).map(function (e) { return dot(e, e.cost.wall_hours); }),
        { yMin: yMin, yMax: yMax, yFmt: function (v) { return Math.round(v); }, xLabel: "training time", yLabel: "Elo", xTick: function (v) { return v < 1 ? Math.round(v * 60) + " min" : v + " h"; } })));
    g.appendChild(card("Strength vs thinking time", "Milliseconds per move at play time (log scale). Search costs time at every move; learning costs time once.",
      scatter(E.filter(function (e) { return e.elo != null; }).map(function (e) { return dot(e, Math.max(e.ms_per_move, 0.01)); }),
        { yMin: yMin, yMax: yMax, yFmt: function (v) { return Math.round(v); }, xLabel: "time per move", yLabel: "Elo" })));
    g.appendChild(card("Elo", "On the minimax ladder (depth 1 = 1000). Grey = our hand-crafted minimax at the same depth (reference).",
      hBars(E.filter(function (e) { return e.elo != null; }).map(function (e) { return { label: short(e), value: e.elo, color: METHOD_COLOR[e.method] || C.white }; }), { xLabel: "Elo" })));
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
    var list = filteredRuns().slice().sort(function (a, b) { return famIdx(a.family) - famIdx(b.family) || (runAcc(b) || 0) - (runAcc(a) || 0); });
    var rows = list.map(function (r) {
      var net = r.model && r.model.blocks ? r.model.blocks + "×" + r.model.channels : r.model && r.model.kind === "n-tuple" ? "n-tuple" : r.model && r.model.kind ? "10 weights" : "–";
      return { cls: r.top8 ? "ours" : "", cells: [esc(r.name) + (r.top8 ? ' <span class="badge ours">top 8</span>' : ""), '<span style="color:' + famColor(r.family) + '">●</span> ' + esc(famShort(r.family)),
              net, fmtInt(r.params), fmtInt(r.last.gen || r.last.epoch), dur((r.cost && r.cost.wall_sec) || r.last.elapsed_sec),
              r.cost ? kwh(r.cost.energy_wh) : "–", r.bench ? pct(r.bench.policy_optimal) : "–", r.bench ? pct(r.bench.value_acc) : "–", pct(r.best_bench_optimal), num(r.eval_elo || r.last_elo)] };
    });
    var series = list.map(function (r, i) { var d = CACHE[r.name]; if (!d) return null; return { name: r.name, color: SEQ[i % SEQ.length], points: d.metrics.filter(function (m) { return m.bench_optimal != null && m.elapsed_sec != null; }).map(function (m) { return [m.elapsed_sec / 3600, m.bench_optimal]; }) }; }).filter(Boolean);
    var c = card("Runs in this view (" + list.length + ")", "Benchmark columns: every net re-measured on the same 3,000 positions (research/bench_all.json), so they compare across runs. \"Best during training\" is the run's own benchmark log. The chart shows runs you have opened on this page; x = hours of training.",
      table(["run", "method", "model", "params", "gen/epoch", "time", "energy", "benchmark optimal", "value acc", "best during training", "Elo"], rows), true);
    if (series.length > 1) c.appendChild(lineChart({ series: series, refs: accRefs(), yMin: 0, yMax: 1, yFmt: function (v) { return Math.round(v * 100) + "%"; }, xLabel: "hours of training", xFmt: function (v) { return v.toFixed(1); } }));
    w.appendChild(c);
    return w;
  }

  // ========================================================== hypothesis
  var EVALS = { evals: [], missing: [] }, EVCACHE = {};
  function evalCommand(m) {
    var runs = m.runs || [], file = (m.model || "").split("+")[0].split("/").pop() || "best.npz";
    if (runs.length > 1)
      return "python -m nn.evaluate --run " + runs[0] + " --out eval_ens" + runs.length + ".json \\\n    --model " + m.model;
    return "python -m nn.evaluate --run " + runs[0] + " --model " + file;
  }
  function loadEvals() {
    return getJSON("/api/lab/evals").then(function (d) {
      EVALS = d; var s = $("hypSel"), prev = s.value; s.innerHTML = "";
      s.appendChild(el("option", { value: "__all" }, "All evaluated nets (overview)"));
      var top = d.evals.filter(function (e) { return e.top8 && e.has_hypothesis; }), rest = d.evals.filter(function (e) { return !e.top8 && e.has_hypothesis; });
      [["Top-8 arena nets", top], ["Other runs", rest]].forEach(function (gp) {
        if (!gp[1].length) return;
        var g = el("optgroup", { label: gp[0] });
        gp[1].forEach(function (e) { g.appendChild(el("option", { value: e.run + "/" + e.file }, esc(e.label + (e.verdict ? " — " + e.verdict.split(" (")[0] : "")))); });
        s.appendChild(g);
      });
      if (prev && Array.prototype.some.call(s.options, function (o) { return o.value === prev; })) s.value = prev;
      loadHyp(s.value);
    }).catch(function (e) { $("hypBody").innerHTML = '<div class="empty">Could not load evaluations: ' + esc(e.message) + "</div>"; });
  }
  function getEval(key) {
    if (EVCACHE[key]) return Promise.resolve(EVCACHE[key]);
    var p = key.split("/");
    return getJSON("/api/lab/eval/" + encodeURIComponent(p[0]) + "/" + encodeURIComponent(p[1])).then(function (d) { EVCACHE[key] = d; return d; });
  }
  function missingCard() {
    if (!EVALS.missing || !EVALS.missing.length) return null;
    var cmds = EVALS.missing.map(function (m) { return "# " + m.label + "\n" + evalCommand(m); }).join("\n\n");
    return card("Not evaluated yet (" + EVALS.missing.length + " of the top-8 arena nets)",
      "These nets are in the " + esc(EVALS.arena || "top-8") + " arena run but have no <code>eval.json</code>, so they are missing from this tab. Run on the training PC from the repo folder, commit <code>runs/&lt;run&gt;/eval*.json</code>, pull and refresh.",
      el("pre", { "class": "cmd" }, esc(cmds)), true);
  }
  function loadHyp(key) {
    var body = $("hypBody");
    if (!EVALS.evals.filter(function (e) { return e.has_hypothesis; }).length) {
      body.innerHTML = '<div class="empty"><b>No evaluated network yet.</b> After training, run<pre>python -m nn.evaluate --run &lt;run&gt;</pre>It plays minimax-with-the-network\'s-evaluation against minimax-with-the-hand-crafted-heuristic at the same depths and writes runs/&lt;run&gt;/eval.json.</div>';
      var mc = missingCard(); if (mc) { var g0 = el("div", { "class": "grid", style: "margin-top:14px" }); g0.appendChild(mc); body.appendChild(g0); }
      return;
    }
    body.innerHTML = '<p class="muted">Loading...</p>';
    if (!key || key === "__all") return renderHypOverview();
    getEval(key).then(function (ev) {
      var meta = EVALS.evals.filter(function (e) { return e.run + "/" + e.file === key; })[0] || {};
      renderHyp(ev, meta);
    }).catch(function (e) { body.innerHTML = '<div class="empty">' + esc(e.message) + "</div>"; });
  }
  function hypShort(e) { return e.runs && e.runs.length > 1 ? "ensemble ×" + e.runs.length : e.run; }

  function renderHypOverview() {
    var list = EVALS.evals.filter(function (e) { return e.has_hypothesis; });
    Promise.all(list.map(function (e) { return getEval(e.run + "/" + e.file).then(function (ev) { return { meta: e, ev: ev }; }); })).then(function (all) {
      var body = $("hypBody"); body.innerHTML = "";
      var runInfo = {}; RUNS.forEach(function (r) { runInfo[r.name] = r; });
      var depths = {}; all.forEach(function (x) { x.ev.hypothesis.rows.forEach(function (r) { depths[r.depth] = 1; }); });
      depths = Object.keys(depths).map(Number).sort(function (a, b) { return a - b; });
      var full = all.filter(function (x) { return x.ev.hypothesis.rows.every(function (r) { return r.supported; }); }).length;
      var topDone = all.filter(function (x) { return x.meta.top8; }).length;
      body.appendChild(el("div", { "class": "tiles" },
        tile("Nets evaluated", all.length, topDone + " of them in the top-8 arena") +
        tile("Supported at every depth", full + " / " + all.length, "score ≥ 50%, wins/blocks, < 2 s") +
        tile("Still to run", (EVALS.missing || []).length, (EVALS.missing || []).length ? "commands below" : "all top-8 nets done") +
        tile("Depths", depths.join(", "), "minimax with NN value vs hand-crafted")));
      var g = el("div", { "class": "grid g2" }); body.appendChild(g);
      var col = function (x, i) { var r = runInfo[x.meta.runs[0]] || {}; return x.meta.runs.length > 1 ? C.white : r.family ? famColor(r.family) : SEQ[i % SEQ.length]; };
      g.appendChild(card("Score against the hand-crafted heuristic", "Same search, same depth, only the evaluation differs (draw = ½). Above 50% = the learned evaluation is better.",
        groupedBars({ categories: depths.map(function (d) { return "depth " + d; }), yMin: 0, yMax: 1, yFmt: PCT, refs: [{ y: 0.5, label: "50%" }],
          series: all.map(function (x, i) { return { name: hypShort(x.meta), color: col(x, i), values: depths.map(function (d) { var r = x.ev.hypothesis.rows.filter(function (q) { return q.depth === d; })[0]; return r ? r.head_to_head.score : null; }) }; }) })));
      g.appendChild(card("Thinking time per move", "Milliseconds per move of the NN-evaluation minimax (log scale). The 2 s line is the success criterion; bigger nets pay for every leaf they evaluate.",
        lineChart({ series: all.map(function (x, i) { return { name: hypShort(x.meta), color: col(x, i), dots: true, points: x.ev.hypothesis.rows.map(function (r) { return [r.depth, r.nn_ms > 0 ? Math.log10(r.nn_ms) : null]; }) }; }),
          refs: [{ y: Math.log10(2000), label: "2 s limit", color: C.red }], xLabel: "search depth", intX: true, yFmt: function (v) { var x = Math.pow(10, v); return x >= 1000 ? (x / 1000).toFixed(x >= 10000 ? 0 : 1) + " s" : Math.round(x) + " ms"; } })));
      var rows = all.map(function (x) {
        var r0 = runInfo[x.meta.runs[0]] || {}, H = x.ev.hypothesis.rows;
        return { cls: x.meta.top8 ? "ours" : "", cells: [esc(x.meta.label) + (x.meta.top8 ? ' <span class="badge ours">top 8</span>' : ""), r0.size || "–", esc(famShort(r0.family))]
          .concat(depths.map(function (d) { var r = H.filter(function (q) { return q.depth === d; })[0]; return r ? pct(r.head_to_head.score, 0) + " " + (r.supported ? '<span class="yes">✓</span>' : '<span class="no">✗</span>') : "–"; }))
          .concat([(function () { var r = H[H.length - 1]; return r ? pct(r.nn_accuracy.optimal) + " / " + pct(r.hand_accuracy.optimal) : "–"; })(),
                   (function () { var r = H[H.length - 1]; return r ? num(r.nn_ms, 0) : "–"; })(), esc((x.ev.hypothesis.verdict || "").split(" (")[0])]) };
      });
      g.appendChild(card("All evaluated nets", "Score per depth (✓ = all three criteria met at that depth). Last two columns at the deepest depth tested: optimal-move rate NN / hand-crafted on " + fmtInt((all[0] || {}).ev.hypothesis.positions) + " positions, and NN ms per move. Pick a net in the menu for the full breakdown.",
        table(["net", "size", "method"].concat(depths.map(function (d) { return "d" + d; })).concat(["optimal (deepest)", "ms (deepest)", "verdict"]), rows), true));
      var mc = missingCard(); if (mc) g.appendChild(mc);
    });
  }

  function renderHyp(ev, meta) {
    var body = $("hypBody"); body.innerHTML = "";
    var h = ev && ev.hypothesis;
    if (!h || !h.rows || !h.rows.length) { body.innerHTML = '<div class="empty">This eval file has no hypothesis section (it was skipped or is still running).</div>'; return; }
    var ok = h.rows.filter(function (r) { return r.supported; }).length;
    body.appendChild(el("div", { "class": "tiles" },
      tile("Hypothesis", ok === h.rows.length ? '<span class="yes">supported</span>' : ok ? '<span class="yes">partly</span>' : '<span class="no">not supported</span>', esc(h.verdict || "")) +
      tile("Network", esc(hypShort(meta)), esc(ev.model || "")) +
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
    if (ev.accuracy || ev.tactics || ev.ladder) body.appendChild(renderEval(ev));
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
        var hs = (location.hash.match(/size=([^&]+)/) || [])[1]; if (hs) F.size = decodeURIComponent(hs);
        var v = renderRunSelect();
        if (!RUNS.length) emptyTraining(); else loadRun(v);
        loadEvals();
      }).catch(function (e) { $("trainBody").innerHTML = '<div class="empty">Could not reach the server: ' + esc(e.message) + "</div>"; });
  }
  tabs();
  $("runSel").addEventListener("change", function () { loadRun(this.value); renderGroup(filteredRuns()); });
  $("sizeSel").addEventListener("change", function () { F.size = this.value; F.fam = "all"; onFilter(); });
  $("famSel").addEventListener("change", function () { F.fam = this.value; onFilter(); });
  $("top8Only").addEventListener("change", function () { F.top = this.checked; onFilter(); });
  $("hypSel").addEventListener("change", function () { loadHyp(this.value); });
  $("arenaSel").addEventListener("change", function () { loadArena(this.value); });
  $("methSel").addEventListener("change", function () { loadMethods(this.value); });
  $("reload").addEventListener("click", function () { CACHE = {}; EVCACHE = {}; loadAll(); loadArenaList(); loadMethodsList(); });
  loadAll();
  loadArenaList();
  loadMethodsList();
})();
