/* Connect 4 Research page: reads /api/research/* and draws everything as inline SVG
   (no libraries, works offline). Every number shown comes from the JSON files; when a
   file or a field is missing the chart says "not available yet" instead of failing. */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  // Okabe-Ito (colour-blind safe) + the site's own colours
  var C = { blue: "#56B4E9", orange: "#E69F00", green: "#009E73", yellow: "#F0E442", red: "#D55E00",
            purple: "#CC79A7", sky: "#0072B2", grey: "#8e96b5", white: "#e9ecf8", gold: "#ffd166" };
  var SEQ = [C.blue, C.orange, C.green, C.purple, C.yellow, C.sky, C.red, C.grey, C.white];
  var SHAPES = ["circle", "square", "diamond", "triangle"];
  var BEST = "nn_sup_6x64_mcts200";
  var D = { arena_analysis: null, roster: null, sweep: null, interp: null, interp_channels: null, report: null };
  var IDX = null, SEL = { findings: {}, roster: {}, runs: {} };
  var UI = { tab: "overview" };       // UI state kept across re-renders

  // ------------------------------------------------------------ helpers
  function el(tag, attrs, html) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) { if (attrs[k] != null) e.setAttribute(k, attrs[k]); }
    if (html != null) e.innerHTML = html;
    return e;
  }
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); }
  function isNum(v) { return typeof v === "number" && isFinite(v); }
  function pct(v, d) { return isNum(v) ? (100 * v).toFixed(d == null ? 1 : d) + "%" : "–"; }
  function num(v, d) { return isNum(v) ? Number(v).toFixed(d == null ? 0 : d) : "–"; }
  function sgn(v, d) { return isNum(v) ? (v > 0 ? "+" : "") + Number(v).toFixed(d == null ? 0 : d) : "–"; }
  function fmtInt(v) { return isNum(v) ? Math.round(v).toLocaleString("en-US") : "–"; }
  function g(o, path) { // safe nested get: g(obj, "a.b.0.c")
    var p = path.split("."), x = o;
    for (var i = 0; i < p.length; i++) { if (x == null) return null; x = x[p[i]]; }
    return x === undefined ? null : x;
  }
  function ms(v) { if (!isNum(v)) return "–"; if (v < 10) return v.toFixed(2) + " ms"; if (v < 1000) return Math.round(v) + " ms"; if (v < 60000) return (v / 1000).toFixed(1) + " s"; return (v / 60000).toFixed(1) + " min"; }
  function params(v) { if (!isNum(v)) return "–"; if (v >= 1e6) return (v / 1e6).toFixed(2) + " M"; if (v >= 1e3) return (v / 1e3).toFixed(v >= 1e5 ? 0 : 1) + " k"; return String(v); }
  function dur(sec) { if (!isNum(sec)) return "–"; if (sec < 90) return Math.round(sec) + " s"; if (sec < 5400) return Math.round(sec / 60) + " min"; return (sec / 3600).toFixed(1) + " h"; }
  function when(iso) { if (!iso) return "–"; return String(iso).replace("T", " ").slice(0, 16); }
  function getJSON(url) {
    return fetch(url, { cache: "no-store" }).then(function (r) {
      if (!r.ok) { var e = new Error(url + ": " + r.status); e.status = r.status; throw e; }
      return r.json();
    });
  }
  function card(title, desc, body, wide) {
    var c = el("div", { "class": "card" + (wide ? " wide" : "") });
    if (title) c.appendChild(el("h3", null, esc(title)));
    if (desc) c.appendChild(el("p", { "class": "desc" }, desc));
    if (body) (Array.isArray(body) ? body : [body]).forEach(function (b) { if (b) c.appendChild(b); });
    return c;
  }
  function empty(msg) { return el("div", { "class": "empty" }, msg || "Not available yet."); }
  function na(what) { return el("p", { "class": "muted" }, esc(what || "Not available yet") + "."); }
  function grid(cls, kids) { var d = el("div", { "class": "grid " + (cls || "") }); kids.forEach(function (k) { if (k) d.appendChild(k); }); return d; }
  function tile(k, v, s) { return '<div class="tile"><div class="k">' + k + '</div><div class="v">' + v + "</div>" + (s ? '<div class="s">' + s + "</div>" : "") + "</div>"; }
  function niceTicks(lo, hi, n) {
    if (!(hi > lo)) { hi = lo + 1; }
    var span = hi - lo, step = Math.pow(10, Math.floor(Math.log10(span / n))), err = n / span * step;
    if (err <= 0.15) step *= 10; else if (err <= 0.35) step *= 5; else if (err <= 0.75) step *= 2;
    var t = [], v = Math.ceil(lo / step) * step;
    for (; v <= hi + 1e-9; v += step) t.push(+v.toFixed(10));
    return t;
  }
  function tickFmt(t) { var a = Math.abs(t); if (a >= 1e6) return (t / 1e6) + "M"; if (a >= 1e4) return (t / 1e3) + "k"; return String(+t.toFixed(4)); }
  function shapeSvg(shape, x, y, r, col, tip, extra) {
    var t = tip != null ? ' data-tip="' + esc(tip) + '"' : "";
    extra = extra || "";
    if (shape === "square") return '<rect x="' + (x - r) + '" y="' + (y - r) + '" width="' + 2 * r + '" height="' + 2 * r + '" fill="' + col + '"' + t + extra + "/>";
    if (shape === "diamond") return '<path d="M' + x + " " + (y - r * 1.3) + "L" + (x + r * 1.3) + " " + y + "L" + x + " " + (y + r * 1.3) + "L" + (x - r * 1.3) + " " + y + 'Z" fill="' + col + '"' + t + extra + "/>";
    if (shape === "triangle") return '<path d="M' + x + " " + (y - r * 1.3) + "L" + (x + r * 1.2) + " " + (y + r) + "L" + (x - r * 1.2) + " " + (y + r) + 'Z" fill="' + col + '"' + t + extra + "/>";
    return '<circle cx="' + x + '" cy="' + y + '" r="' + r + '" fill="' + col + '"' + t + extra + "/>";
  }
  function legend(items) {
    var d = el("div", { "class": "legend" });
    items.forEach(function (it) {
      var sw;
      if (it.shape) sw = '<svg width="14" height="12" style="vertical-align:middle;margin-right:5px" aria-hidden="true">' + shapeSvg(it.shape, 7, 6, 4.2, it.color) + "</svg>";
      else sw = '<i class="' + (it.dash ? "dash" : "") + '" style="background:' + it.color + ";color:" + it.color + '"></i>';
      d.appendChild(el("span", null, sw + esc(it.name)));
    });
    return d;
  }
  function luminance(rgb) { var m = /rgb\((\d+),(\d+),(\d+)\)/.exec(rgb); if (!m) return 0.5; return (0.299 * m[1] + 0.587 * m[2] + 0.114 * m[3]) / 255; }
  function mix(a, b, t) { return [0, 1, 2].map(function (i) { return Math.round(a[i] + (b[i] - a[i]) * t); }); }
  function hex(h) { return [parseInt(h.substr(1, 2), 16), parseInt(h.substr(3, 2), 16), parseInt(h.substr(5, 2), 16)]; }
  var VIR = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"].map(hex);
  function seqColor(t) { // 0..1 -> viridis
    if (!isNum(t)) return "#1a1f38";
    t = Math.max(0, Math.min(1, t)); var k = Math.min(3, Math.floor(t * 4)), f = t * 4 - k;
    return "rgb(" + mix(VIR[k], VIR[k + 1], f).join(",") + ")";
  }
  var NEG = hex("#0072B2"), MID = hex("#3a3f5c"), POS = hex("#E69F00");
  function divColor(t) { // -1..1 -> blue / grey / orange
    if (!isNum(t)) return "#1a1f38";
    t = Math.max(-1, Math.min(1, t));
    return "rgb(" + (t < 0 ? mix(MID, NEG, -t) : mix(MID, POS, t)).join(",") + ")";
  }
  function rampCss(kind) {
    if (kind === "div") return "linear-gradient(90deg,#0072B2,#3a3f5c,#E69F00)";
    return "linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725)";
  }
  function scaleLegend(kind, lo, hi, label) {
    return el("div", { "class": "scale" }, "<span>" + esc(lo) + '</span><span class="ramp" style="background:' + rampCss(kind) + '"></span><span>' + esc(hi) + "</span><span>" + esc(label || "") + "</span>");
  }
  function select(id, options, value, label) {
    var h = (label ? '<label for="' + id + '">' + esc(label) + "</label> " : "") + '<select id="' + id + '">';
    options.forEach(function (o) { var v = typeof o === "string" ? o : o.v, t = typeof o === "string" ? o : o.t; h += '<option value="' + esc(v) + '"' + (String(v) === String(value) ? " selected" : "") + ">" + esc(t) + "</option>"; });
    return h + "</select>";
  }
  function table(head, rows, opts) {
    opts = opts || {};
    var h = '<div class="scroll"><table class="t ' + (opts.cls || "") + '"><thead><tr>' + head.map(function (x) { return '<th scope="col">' + x + "</th>"; }).join("") + "</tr></thead><tbody>";
    rows.forEach(function (r) { var c = r.cls ? ' class="' + r.cls + '"' : ""; var cells = r.cells || r; h += "<tr" + c + ">" + cells.map(function (x) { return "<td>" + (x == null ? "–" : x) + "</td>"; }).join("") + "</tr>"; });
    if (!rows.length) h += '<tr><td colspan="' + head.length + '" class="l muted">Not available yet.</td></tr>';
    var d = el("div", null, h + "</tbody></table></div>");
    if (opts.caption) d.appendChild(el("p", { "class": "stats" }, opts.caption));
    return d;
  }
  function ciCell(v, lo, hi, d) { return isNum(v) ? num(v, d) + (isNum(lo) ? ' <span class="muted">[' + num(lo, d) + ", " + num(hi, d) + "]</span>" : "") : "–"; }
  function list(items) { return "<ul>" + (items || []).map(function (s) { return "<li>" + esc(s) + "</li>"; }).join("") + "</ul>"; }

  // ------------------------------------------------------------ tooltip
  var TIP = null;
  function tipShow(t, x, y) {
    TIP = TIP || $("tip");
    TIP.textContent = t; TIP.hidden = false;
    var w = TIP.offsetWidth, h = TIP.offsetHeight;
    var px = x + 14, py = y + 14;
    if (px + w > window.innerWidth - 6) px = x - w - 10;
    if (py + h > window.innerHeight - 6) py = y - h - 10;
    TIP.style.left = Math.max(4, px) + "px"; TIP.style.top = Math.max(4, py) + "px";
  }
  function tipHide() { if (TIP) TIP.hidden = true; }
  document.addEventListener("mousemove", function (e) {
    var t = e.target && e.target.closest ? e.target.closest("[data-tip]") : null;
    if (t) tipShow(t.getAttribute("data-tip"), e.clientX, e.clientY); else tipHide();
  });
  document.addEventListener("focusin", function (e) {
    var t = e.target && e.target.closest ? e.target.closest("[data-tip]") : null;
    if (t) { var r = t.getBoundingClientRect(); tipShow(t.getAttribute("data-tip"), r.right, r.top); } else tipHide();
  });
  document.addEventListener("scroll", tipHide, true);

  // =================================================================== charts
  function svgHead(W, H, label) { return '<svg class="chart" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' + esc(label || "chart") + '">'; }
  function axisLabels(s, W, H, m, xl, yl) {
    if (xl) s += '<text x="' + ((m.l + W - m.r) / 2) + '" y="' + (H - 4) + '" text-anchor="middle">' + esc(xl) + "</text>";
    if (yl) s += '<text transform="translate(12,' + ((m.t + H - m.b) / 2) + ') rotate(-90)" text-anchor="middle">' + esc(yl) + "</text>";
    return s;
  }

  // x/y chart: scatter, lines, CI whiskers, fitted lines, log x.  series: [{name,color,shape,line,dash,points:[{x,y,lo,hi,label,tip}],fit:{slope,intercept,log10x}}]
  function xyChart(o) {
    var W = o.width || 640, H = o.height || 290, m = { l: 58, r: o.labels ? 92 : 18, t: 12, b: 42 };
    var tx = o.logX ? function (v) { return Math.log10(v); } : function (v) { return v; };
    var pts = [], ys = [];
    o.series.forEach(function (se) { (se.points || []).forEach(function (p) { if (isNum(p.x) && isNum(p.y) && (!o.logX || p.x > 0)) { pts.push(p); ys.push(p.y); if (isNum(p.lo)) ys.push(p.lo); if (isNum(p.hi)) ys.push(p.hi); } }); });
    var wrap = el("div");
    if (!pts.length) { wrap.appendChild(na(o.emptyText || "No data points yet")); return wrap; }
    (o.refs || []).forEach(function (r) { if (isNum(r.y)) ys.push(r.y); });
    var xs = pts.map(function (p) { return tx(p.x); });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    if (o.xCats) { x0 = -0.5; x1 = o.xCats.length - 0.5; }
    else if (o.logX) { x0 = Math.floor(x0 - 0.05); x1 = Math.ceil(x1 + 0.05); }
    else { var px = (x1 - x0) * 0.06 || 0.5; x0 -= px; x1 += px; }
    if (o.xMin != null) x0 = tx(o.xMin); if (o.xMax != null) x1 = tx(o.xMax);
    var y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys), py = (y1 - y0) * 0.08 || Math.abs(y1) * 0.05 || 0.05;
    y0 = o.yMin != null ? o.yMin : y0 - py; y1 = o.yMax != null ? o.yMax : y1 + py;
    var X = function (v) { return m.l + (tx(v) - x0) / (x1 - x0) * (W - m.l - m.r); };
    var Xt = function (t) { return m.l + (t - x0) / (x1 - x0) * (W - m.l - m.r); };
    var Y = function (v) { return H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b); };
    var yf = o.yFmt || tickFmt, xf = o.xFmt || (o.logX ? params : tickFmt);
    var s = svgHead(W, H, o.title) + '<g class="grid">';
    niceTicks(y0, y1, 5).forEach(function (t) { s += '<line x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(t) + '" y2="' + Y(t) + '"/><text x="' + (m.l - 6) + '" y="' + (Y(t) + 4) + '" text-anchor="end">' + esc(yf(t)) + "</text>"; });
    if (o.xCats) o.xCats.forEach(function (c, i) { s += '<text x="' + Xt(i) + '" y="' + (H - m.b + 16) + '" text-anchor="middle">' + esc(c) + "</text>"; });
    else if (o.logX) for (var e = x0; e <= x1; e++) { s += '<line x1="' + Xt(e) + '" x2="' + Xt(e) + '" y1="' + m.t + '" y2="' + (H - m.b) + '"/><text x="' + Xt(e) + '" y="' + (H - m.b + 16) + '" text-anchor="middle">' + esc(xf(Math.pow(10, e))) + "</text>"; }
    else niceTicks(x0, x1, 7).forEach(function (t) { s += '<text x="' + Xt(t) + '" y="' + (H - m.b + 16) + '" text-anchor="middle">' + esc(xf(t)) + "</text>"; });
    s += '</g><g class="axis"><line x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + (H - m.b) + '" y2="' + (H - m.b) + '"/></g>';
    s = axisLabels(s, W, H, m, o.xLabel, o.yLabel);
    (o.refs || []).forEach(function (r, k) {
      if (!isNum(r.y) || r.y < y0 || r.y > y1) return;
      var col = r.color || C.grey, left = k % 2 === 1;
      s += '<g class="ref"><line x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(r.y) + '" y2="' + Y(r.y) + '" stroke="' + col + '"/>';
      s += '<text x="' + (left ? m.l + 4 : W - m.r - 2) + '" y="' + (Y(r.y) - 3) + '" text-anchor="' + (left ? "start" : "end") + '" style="fill:' + col + '">' + esc(r.label) + "</text></g>";
    });
    o.series.forEach(function (se, i) {
      var col = se.color || SEQ[i % SEQ.length], shape = se.shape || SHAPES[i % SHAPES.length];
      var p = (se.points || []).filter(function (q) { return isNum(q.x) && isNum(q.y) && (!o.logX || q.x > 0); });
      if (!p.length) return;
      if (se.fit && isNum(se.fit.slope) && isNum(se.fit.intercept)) {
        var fx = p.map(function (q) { return q.x; }), a = Math.min.apply(null, fx), b = Math.max.apply(null, fx);
        var f = function (x) { return se.fit.intercept + se.fit.slope * (se.fit.log10x ? Math.log10(x) : x); };
        s += '<line x1="' + X(a) + '" x2="' + X(b) + '" y1="' + Y(f(a)) + '" y2="' + Y(f(b)) + '" stroke="' + col + '" stroke-width="1.6" stroke-dasharray="7 4" opacity=".9"/>';
      }
      if (se.line) {
        var ps = p.slice().sort(function (u, v) { return u.x - v.x; });
        s += '<path d="' + ps.map(function (q, k) { return (k ? "L" : "M") + X(q.x).toFixed(1) + " " + Y(q.y).toFixed(1); }).join("") + '" fill="none" stroke="' + col + '" stroke-width="2"' + (se.dash ? ' stroke-dasharray="6 4"' : "") + "/>";
      }
      p.forEach(function (q) {
        if (isNum(q.lo) && isNum(q.hi)) {
          var xx = X(q.x);
          s += '<line x1="' + xx + '" x2="' + xx + '" y1="' + Y(q.lo) + '" y2="' + Y(q.hi) + '" stroke="' + col + '" stroke-width="1.4"/><line x1="' + (xx - 4) + '" x2="' + (xx + 4) + '" y1="' + Y(q.lo) + '" y2="' + Y(q.lo) + '" stroke="' + col + '"/><line x1="' + (xx - 4) + '" x2="' + (xx + 4) + '" y1="' + Y(q.hi) + '" y2="' + Y(q.hi) + '" stroke="' + col + '"/>';
        }
        var tip = q.tip || ((se.name ? se.name + "\n" : "") + (q.label ? q.label + "\n" : "") + (o.xName || "x") + ": " + (o.xCats ? o.xCats[q.x] : xf(q.x)) + "\n" + (o.yName || "y") + ": " + yf(q.y) + (isNum(q.lo) ? " [" + yf(q.lo) + ", " + yf(q.hi) + "]" : ""));
        s += shapeSvg(shape, X(q.x), Y(q.y), q.big ? 6.5 : 4.6, col, tip, q.big ? ' stroke="#fff" stroke-width="1.5"' : ' stroke="#0f1220" stroke-width=".8"');
        if (o.labels && q.label) s += '<text x="' + (X(q.x) + 8) + '" y="' + (Y(q.y) + 4) + '" style="fill:#cfd5ee;font-size:10px">' + esc(q.short || q.label) + "</text>";
      });
    });
    s += "</svg>";
    wrap.innerHTML = s;
    var items = o.series.filter(function (se) { return se.name && !se.noLegend; }).map(function (se) { var i = o.series.indexOf(se); return { name: se.name, color: se.color || SEQ[i % SEQ.length], shape: se.shape || SHAPES[i % SHAPES.length] }; });
    if (o.series.some(function (se) { return se.fit && isNum(se.fit.slope); })) items.push({ name: "fitted OLS line (dashed)", color: C.grey, dash: true });
    (o.refs || []).filter(function (r) { return r.legend; }).forEach(function (r) { items.push({ name: r.legend, color: r.color || C.grey, dash: true }); });
    if (items.length) wrap.appendChild(legend(items));
    return wrap;
  }

  // horizontal bars with optional CI whiskers. items: [{label,value,lo,hi,color,tip,bold}]
  function hbarsCI(items, o) {
    o = o || {};
    items = items.filter(function (it) { return isNum(it.value); });
    if (!items.length) return na(o.emptyText);
    var W = 640, rowH = o.rowH || 24, m = { l: o.labelW || 220, r: 74, t: 6, b: 34 }, H = m.t + m.b + rowH * items.length;
    var tx = o.logX ? function (v) { return Math.log10(Math.max(v, 1e-9)); } : function (v) { return v; };
    var vals = []; items.forEach(function (i) { vals.push(i.value); if (isNum(i.lo)) vals.push(i.lo); if (isNum(i.hi)) vals.push(i.hi); });
    if (o.ref && isNum(o.ref.v)) vals.push(o.ref.v);
    var lo = o.min != null ? o.min : Math.min.apply(null, vals), hi = o.max != null ? o.max : Math.max.apply(null, vals);
    if (o.base != null && !o.logX) { lo = Math.min(lo, o.base); hi = Math.max(hi, o.base); }
    lo = tx(lo); hi = tx(hi);
    if (o.logX) { lo = Math.floor(lo); hi = Math.ceil(hi); if (hi === lo) hi++; }
    else if (o.min == null || o.max == null) { var pad = (hi - lo) * 0.05 || 1; if (o.min == null) lo -= pad; if (o.max == null) hi += pad; }
    var X = function (v) { return m.l + (tx(v) - lo) / (hi - lo) * (W - m.l - m.r); };
    var Xt = function (t) { return m.l + (t - lo) / (hi - lo) * (W - m.l - m.r); };
    var f = o.fmt || function (v) { return num(v, 2); };
    var s = svgHead(W, H, o.title) + '<g class="grid">';
    if (o.logX) for (var e = lo; e <= hi; e++) s += '<line x1="' + Xt(e) + '" x2="' + Xt(e) + '" y1="' + m.t + '" y2="' + (H - m.b) + '"/><text x="' + Xt(e) + '" y="' + (H - m.b + 15) + '" text-anchor="middle">' + esc((o.tickFmt || f)(Math.pow(10, e))) + "</text>";
    else niceTicks(lo, hi, 6).forEach(function (t) { s += '<line x1="' + Xt(t) + '" x2="' + Xt(t) + '" y1="' + m.t + '" y2="' + (H - m.b) + '"/><text x="' + Xt(t) + '" y="' + (H - m.b + 15) + '" text-anchor="middle">' + esc((o.tickFmt || tickFmt)(t)) + "</text>"; });
    s += "</g>";
    var base = o.logX ? Math.pow(10, lo) : (o.base != null ? o.base : Math.pow(10, 0) * 0 + lo);
    if (o.ref && isNum(o.ref.v)) s += '<g class="ref"><line x1="' + X(o.ref.v) + '" x2="' + X(o.ref.v) + '" y1="' + m.t + '" y2="' + (H - m.b) + '" stroke="' + C.white + '"/><text x="' + (X(o.ref.v) + 3) + '" y="' + (m.t + 9) + '" style="fill:' + C.white + '">' + esc(o.ref.label || "") + "</text></g>";
    items.forEach(function (it, i) {
      var y = m.t + i * rowH + rowH / 2, col = it.color || C.blue;
      var tip = it.tip || (it.label + ": " + f(it.value) + (isNum(it.lo) ? " [" + f(it.lo) + ", " + f(it.hi) + "]" : ""));
      s += '<text x="' + (m.l - 8) + '" y="' + (y + 4) + '" text-anchor="end" style="fill:' + (it.bold ? C.gold : "#e9ecf8") + '">' + esc(it.label) + "</text>";
      var xa = X(base), xb = X(it.value);
      s += '<rect x="' + Math.min(xa, xb) + '" y="' + (y - 7) + '" width="' + Math.max(1, Math.abs(xb - xa)) + '" height="14" rx="2" fill="' + col + '" fill-opacity=".7" data-tip="' + esc(tip) + '"/>';
      if (isNum(it.lo) && isNum(it.hi)) s += '<line x1="' + X(it.lo) + '" x2="' + X(it.hi) + '" y1="' + y + '" y2="' + y + '" stroke="#e9ecf8" stroke-width="1.5"/><line x1="' + X(it.lo) + '" x2="' + X(it.lo) + '" y1="' + (y - 6) + '" y2="' + (y + 6) + '" stroke="#e9ecf8"/><line x1="' + X(it.hi) + '" x2="' + X(it.hi) + '" y1="' + (y - 6) + '" y2="' + (y + 6) + '" stroke="#e9ecf8"/>';
      s += '<text x="' + (W - m.r + 6) + '" y="' + (y + 4) + '" style="fill:#cfd5ee">' + esc(f(it.value)) + (it.n ? " (n=" + it.n + ")" : "") + "</text>";
    });
    if (o.xLabel) s += '<text x="' + ((m.l + W - m.r) / 2) + '" y="' + (H - 2) + '" text-anchor="middle">' + esc(o.xLabel) + "</text>";
    s += "</svg>";
    var d = el("div"); d.innerHTML = s;
    if (o.legend) d.appendChild(legend(o.legend));
    return d;
  }

  // vertical grouped bars, negative values allowed, optional CI. series: [{name,color,values,lo,hi,tips}]
  function groupBars(o) {
    var W = o.width || 640, H = o.height || 250, m = { l: 56, r: 12, t: 12, b: o.rotate ? 70 : 44 };
    var cats = o.categories, ser = o.series, vals = [];
    ser.forEach(function (se) { se.values.forEach(function (v, i) { if (isNum(v)) vals.push(v); if (se.lo && isNum(se.lo[i])) vals.push(se.lo[i]); if (se.hi && isNum(se.hi[i])) vals.push(se.hi[i]); }); });
    var wrap = el("div");
    if (!vals.length || !cats.length) { wrap.appendChild(na(o.emptyText)); return wrap; }
    (o.refs || []).forEach(function (r) { vals.push(r.y); });
    var y0 = o.yMin != null ? o.yMin : Math.min(0, Math.min.apply(null, vals)), y1 = o.yMax != null ? o.yMax : Math.max(0, Math.max.apply(null, vals));
    var pad = (y1 - y0) * 0.08 || 1; if (o.yMin == null && y0 < 0) y0 -= pad; if (o.yMax == null) y1 += pad;
    var Y = function (v) { return H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b); };
    var gw = (W - m.l - m.r) / cats.length, bw = Math.min(30, gw * 0.84 / ser.length);
    var yf = o.yFmt || tickFmt;
    var s = svgHead(W, H, o.title) + '<g class="grid">';
    niceTicks(y0, y1, 5).forEach(function (t) { s += '<line x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(t) + '" y2="' + Y(t) + '"/><text x="' + (m.l - 6) + '" y="' + (Y(t) + 4) + '" text-anchor="end">' + esc(yf(t)) + "</text>"; });
    s += '</g><g class="axis"><line x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(Math.max(y0, Math.min(0, y1))) + '" y2="' + Y(Math.max(y0, Math.min(0, y1))) + '"/></g>';
    (o.refs || []).forEach(function (r) { s += '<g class="ref"><line x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(r.y) + '" y2="' + Y(r.y) + '" stroke="' + (r.color || C.white) + '"/><text x="' + (W - m.r) + '" y="' + (Y(r.y) - 3) + '" text-anchor="end" style="fill:' + (r.color || C.white) + '">' + esc(r.label) + "</text></g>"; });
    var zero = Y(Math.max(y0, Math.min(0, y1)));
    cats.forEach(function (c, i) {
      var gx = m.l + i * gw + (gw - bw * ser.length) / 2;
      ser.forEach(function (se, j) {
        var v = se.values[i]; if (!isNum(v)) return;
        var col = se.color || SEQ[j % SEQ.length], yy = Y(v), x = gx + j * bw;
        var tip = (se.tips && se.tips[i]) || (se.name + " / " + c + ": " + yf(v) + (se.lo && isNum(se.lo[i]) ? " [" + yf(se.lo[i]) + ", " + yf(se.hi[i]) + "]" : ""));
        s += '<rect x="' + x + '" y="' + Math.min(yy, zero) + '" width="' + Math.max(1, bw - 2) + '" height="' + Math.max(1, Math.abs(zero - yy)) + '" rx="2" fill="' + col + '" data-tip="' + esc(tip) + '"/>';
        if (se.lo && isNum(se.lo[i]) && isNum(se.hi[i])) { var cx = x + (bw - 2) / 2; s += '<line x1="' + cx + '" x2="' + cx + '" y1="' + Y(se.lo[i]) + '" y2="' + Y(se.hi[i]) + '" stroke="#e9ecf8" stroke-width="1.3"/><line x1="' + (cx - 3) + '" x2="' + (cx + 3) + '" y1="' + Y(se.lo[i]) + '" y2="' + Y(se.lo[i]) + '" stroke="#e9ecf8"/><line x1="' + (cx - 3) + '" x2="' + (cx + 3) + '" y1="' + Y(se.hi[i]) + '" y2="' + Y(se.hi[i]) + '" stroke="#e9ecf8"/>'; }
      });
      var lx = m.l + i * gw + gw / 2;
      if (o.rotate) s += '<text transform="translate(' + lx + "," + (H - m.b + 12) + ') rotate(-35)" text-anchor="end">' + esc(c) + "</text>";
      else s += '<text x="' + lx + '" y="' + (H - m.b + 16) + '" text-anchor="middle">' + esc(c) + "</text>";
    });
    s = axisLabels(s, W, H, m, o.rotate ? null : o.xLabel, o.yLabel);
    s += "</svg>";
    wrap.innerHTML = s;
    wrap.appendChild(legend(ser.map(function (se, j) { return { name: se.name, color: se.color || SEQ[j % SEQ.length] }; })));
    return wrap;
  }

  // vertical stacked bars. series: [{name,color,values}]
  function stackBars(o) {
    var W = o.width || 640, H = o.height || 240, m = { l: 50, r: 12, t: 10, b: o.rotate ? 64 : 40 };
    var cats = o.categories, ser = o.series;
    var tot = cats.map(function (c, i) { return ser.reduce(function (a, se) { return a + (isNum(se.values[i]) ? se.values[i] : 0); }, 0); });
    var wrap = el("div");
    if (!cats.length || !Math.max.apply(null, tot.concat([0]))) { wrap.appendChild(na(o.emptyText)); return wrap; }
    var y1 = Math.max.apply(null, tot) * 1.05;
    var Y = function (v) { return H - m.b - v / y1 * (H - m.t - m.b); };
    var gw = (W - m.l - m.r) / cats.length, bw = Math.min(46, gw * 0.7);
    var s = svgHead(W, H, o.title) + '<g class="grid">';
    niceTicks(0, y1, 5).forEach(function (t) { s += '<line x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(t) + '" y2="' + Y(t) + '"/><text x="' + (m.l - 6) + '" y="' + (Y(t) + 4) + '" text-anchor="end">' + tickFmt(t) + "</text>"; });
    s += "</g>";
    cats.forEach(function (c, i) {
      var acc = 0, x = m.l + i * gw + (gw - bw) / 2;
      ser.forEach(function (se, j) {
        var v = se.values[i]; if (!isNum(v) || v <= 0) return;
        s += '<rect x="' + x + '" y="' + Y(acc + v) + '" width="' + bw + '" height="' + (Y(acc) - Y(acc + v)) + '" fill="' + (se.color || SEQ[j % SEQ.length]) + '" stroke="#0f1220" stroke-width=".8" data-tip="' + esc(c + "\n" + se.name + ": " + (o.fmt ? o.fmt(v) : v)) + '"/>';
        acc += v;
      });
      var lx = m.l + i * gw + gw / 2;
      if (o.rotate) s += '<text transform="translate(' + lx + "," + (H - m.b + 12) + ') rotate(-35)" text-anchor="end">' + esc(c) + "</text>";
      else s += '<text x="' + lx + '" y="' + (H - m.b + 16) + '" text-anchor="middle">' + esc(c) + "</text>";
    });
    s = axisLabels(s, W, H, m, o.rotate ? null : o.xLabel, o.yLabel);
    wrap.innerHTML = s + "</svg>";
    wrap.appendChild(legend(ser.map(function (se, j) { return { name: se.name, color: se.color || SEQ[j % SEQ.length] }; })));
    return wrap;
  }

  // heatmap. rows: [{label,group}], cols: [label], val(r,c), color(v) -> css, tip(r,c,v), text(v), mark(r,c) -> css stroke colour
  function heatmap(o) {
    var rows = o.rows, cols = o.cols, cw = o.cellW || 38, ch = o.cellH || 20, lw = o.labelW || 230, top = o.top || 90;
    var groups = 0, prev = null; rows.forEach(function (r) { if (r.group && r.group !== prev) { groups++; prev = r.group; } });
    var W = lw + cols.length * cw + 10, H = top + rows.length * ch + groups * 16 + 6;
    var s = svgHead(W, H, o.title).replace('class="chart"', 'class="chart hm"');
    cols.forEach(function (c, j) { var x = lw + j * cw + cw / 2; s += '<text transform="translate(' + (x + 3) + "," + (top - 6) + ') rotate(-55)" text-anchor="start">' + esc(c) + "</text>"; });
    var y = top; prev = null;
    rows.forEach(function (r, i) {
      if (r.group && r.group !== prev) { s += '<text class="grouplab" x="4" y="' + (y + 12) + '">' + esc(r.group) + "</text>"; y += 16; prev = r.group; }
      s += '<text class="rowlab" x="' + (lw - 6) + '" y="' + (y + ch / 2 + 4) + '" text-anchor="end"' + (o.onRow ? ' data-row="' + i + '" style="cursor:pointer' + (o.activeRow === i ? ";fill:" + C.gold + ";font-weight:700" : "") + '"' : "") + ">" + esc(r.label) + "</text>";
      cols.forEach(function (c, j) {
        var v = o.val(i, j), fill = o.color(v), mk = o.mark ? o.mark(i, j) : null;
        s += '<rect x="' + (lw + j * cw + 1) + '" y="' + (y + 1) + '" width="' + (cw - 2) + '" height="' + (ch - 2) + '" rx="2" fill="' + fill + '"' + (mk ? ' stroke="' + mk + '" stroke-width="2"' : "") + ' data-tip="' + esc(o.tip ? o.tip(i, j, v) : r.label + " / " + c + ": " + v) + '"' + (o.onRow ? ' data-row="' + i + '"' : "") + "/>";
        if (o.text && isNum(v)) s += '<text x="' + (lw + j * cw + cw / 2) + '" y="' + (y + ch / 2 + 4) + '" text-anchor="middle" pointer-events="none" style="fill:' + (luminance(fill) > 0.5 ? "#0b0e1c" : "#e9ecf8") + '">' + esc(o.text(v)) + "</text>";
      });
      y += ch;
    });
    s += "</svg>";
    var wrap = el("div", { "class": "scroll" }); wrap.innerHTML = s;
    var svg = wrap.firstChild; svg.style.maxWidth = W + "px";
    if (o.onRow) wrap.addEventListener("click", function (e) { var t = e.target.closest("[data-row]"); if (t) o.onRow(+t.getAttribute("data-row")); });
    return wrap;
  }

  // a 6x7 board heatmap (row 0 = TOP row, as in nn/encode.py and the play page grid)
  function boardMap(vals, o) {
    o = o || {};
    var cs = o.cell || 26, W = cs * 7 + 8, H = cs * 6 + (o.noCols ? 8 : 22);
    var flat = [], neg = false; (vals || []).forEach(function (r) { (r || []).forEach(function (v) { if (isNum(v)) { flat.push(Math.abs(v)); if (v < 0) neg = true; } }); });
    if (!flat.length) return na("No map");
    if (o.seq == null) o.seq = !neg;   // all values >= 0 (e.g. post-ReLU activations): sequential scale
    var mx = o.max || Math.max.apply(null, flat) || 1;
    var f = o.fmt || function (v) { return num(v, 3); };
    var s = '<svg class="chart" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' + esc(o.title || "board map") + '" style="max-width:' + (o.maxW || W * 1.6) + 'px">';
    s += '<rect x="0" y="0" width="' + W + '" height="' + (cs * 6 + 8) + '" rx="8" fill="#1d2a6b"/>';
    for (var r = 0; r < 6; r++) for (var c = 0; c < 7; c++) {
      var v = vals[r] ? vals[r][c] : null;
      var col = o.seq ? seqColor(isNum(v) ? v / mx : null) : divColor(isNum(v) ? v / mx : null);
      s += '<circle cx="' + (4 + c * cs + cs / 2) + '" cy="' + (4 + r * cs + cs / 2) + '" r="' + (cs * 0.42) + '" fill="' + col + '" data-tip="' + esc((o.title ? o.title + "\n" : "") + "row " + (6 - r) + " from bottom, column " + (c + 1) + ": " + f(v)) + '"/>';
    }
    if (!o.noCols) for (c = 0; c < 7; c++) s += '<text x="' + (4 + c * cs + cs / 2) + '" y="' + (H - 6) + '" text-anchor="middle"' + (o.hiCol === c ? ' style="fill:' + C.gold + ';font-weight:700"' : "") + ">" + (c + 1) + "</text>";
    var d = el("div"); d.innerHTML = s + "</svg>";
    if (o.scale !== false) d.appendChild(scaleLegend(o.seq ? "seq" : "div", o.seq ? "0" : "-" + num(mx, 3), num(mx, 3), o.scaleLabel || ""));
    return d;
  }

  // dumbbell: one row per item, several marked values (each with optional CI) on a shared x axis
  function dumbbell(rows, series, o) {
    o = o || {};
    var W = 640, rowH = 26, m = { l: o.labelW || 220, r: 18, t: 6, b: 34 }, H = m.t + m.b + rowH * rows.length;
    var vals = [];
    rows.forEach(function (r) { series.forEach(function (se) { var p = se.get(r); if (p && isNum(p.v)) { vals.push(p.v); if (isNum(p.lo)) vals.push(p.lo); if (isNum(p.hi)) vals.push(p.hi); } }); });
    if (!vals.length) return na(o.emptyText);
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals), pad = (hi - lo) * 0.04 || 10; lo -= pad; hi += pad;
    var X = function (v) { return m.l + (v - lo) / (hi - lo) * (W - m.l - m.r); };
    var s = svgHead(W, H, o.title) + '<g class="grid">';
    niceTicks(lo, hi, 7).forEach(function (t) { s += '<line x1="' + X(t) + '" x2="' + X(t) + '" y1="' + m.t + '" y2="' + (H - m.b) + '"/><text x="' + X(t) + '" y="' + (H - m.b + 15) + '" text-anchor="middle">' + tickFmt(t) + "</text>"; });
    s += "</g>";
    rows.forEach(function (r, i) {
      var y = m.t + i * rowH + rowH / 2, n = series.length;
      s += '<text x="' + (m.l - 8) + '" y="' + (y + 4) + '" text-anchor="end" style="fill:' + (r.bold ? C.gold : "#e9ecf8") + '">' + esc(r.label) + "</text>";
      series.forEach(function (se, j) {
        var p = se.get(r); if (!p || !isNum(p.v)) return;
        var yy = y + (j - (n - 1) / 2) * 6, col = se.color || SEQ[j];
        if (isNum(p.lo) && isNum(p.hi)) s += '<line x1="' + X(p.lo) + '" x2="' + X(p.hi) + '" y1="' + yy + '" y2="' + yy + '" stroke="' + col + '" stroke-width="1.5" opacity=".85"/>';
        s += shapeSvg(se.shape || SHAPES[j], X(p.v), yy, 4.3, col, r.label + "\n" + se.name + ": " + num(p.v, 1) + (isNum(p.lo) ? " [" + num(p.lo, 1) + ", " + num(p.hi, 1) + "]" : ""), ' stroke="#0f1220" stroke-width=".8"');
      });
    });
    if (o.xLabel) s += '<text x="' + ((m.l + W - m.r) / 2) + '" y="' + (H - 2) + '" text-anchor="middle">' + esc(o.xLabel) + "</text>";
    var d = el("div"); d.innerHTML = s + "</svg>";
    d.appendChild(legend(series.map(function (se, j) { return { name: se.name, color: se.color || SEQ[j], shape: se.shape || SHAPES[j] }; })));
    return d;
  }

  // =================================================================== data + selection
  var FILES = ["arena_analysis", "roster", "arena_top15", "sweep", "interp", "report"];   // interp_channels is loaded on demand
  var SEEN = {};
  function stamp(ent) { return ent && ent.exists ? (ent.mtime || "") + "|" + (ent.bytes || 0) : "none"; }
  function loadIndex() {
    return getJSON("/api/research/index").then(function (ix) {
      IDX = ix.datasets || {};
      var changed = {};
      var jobs = FILES.map(function (name) {
        var st = stamp(IDX[name]), first = !(name in SEEN);
        if (SEEN[name] === st) return null;
        if (!IDX[name] || !IDX[name].exists) { SEEN[name] = st; if (D[name] !== null || first) { D[name] = null; changed[name] = 1; } return null; }
        return getJSON("/api/research/data/" + name).then(function (d) { D[name] = d; SEEN[name] = st; changed[name] = 1; })
          .catch(function () { /* mid-write: try again next poll */ });
      }).filter(Boolean);
      if (stamp(IDX.interp_channels) !== SEEN.interp_channels) { D.interp_channels = null; SEEN.interp_channels = stamp(IDX.interp_channels); }
      return Promise.all(jobs).then(function () { return changed; });
    });
  }
  function loadSelection() {
    return getJSON("/api/research/selection").then(function (s) { SEL = { findings: s.findings || {}, roster: s.roster || {}, runs: s.runs || {} }; })
      .catch(function () { /* keep empty selection */ });
  }
  var saveTimer = null, saveSeq = 0;
  function saveSoon() {
    $("saveState").textContent = "Unsaved changes…";
    clearTimeout(saveTimer);
    saveTimer = setTimeout(saveNow, 600);
  }
  function saveNow() {
    var seq = ++saveSeq;
    $("saveState").textContent = "Saving…";
    fetch("/api/research/selection", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(SEL) })
      .then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error(j.error || r.status); return j; }); })
      .then(function (j) {
        if (seq !== saveSeq) return;
        $("saveState").textContent = "Saved " + when(j.updated).slice(11);
        renderOverviewCounts();
      })
      .catch(function (e) { $("saveState").textContent = "Could not save: " + e.message; });
  }
  function allFindings() {
    var out = [];
    [["arena_analysis", "Arena"], ["sweep", "Supervised sweep"], ["interp", "Interpretability"]].forEach(function (p) {
      var d = D[p[0]];
      ((d && d.findings) || []).forEach(function (f) { if (f && f.id) out.push({ f: f, file: p[0], section: p[1] }); });
    });
    return out;
  }
  function countSel(map) { var n = 0; for (var k in map) if (map[k] === true || (map[k] && map[k].include)) n++; return n; }

  // =================================================================== 1. overview
  var DESC = {
    arena_analysis: "The part3 round robin (20 bots, 20 openings x both colours) re-analysed: Bradley-Terry Elo with bootstrap CIs, head-to-head scores, a top-15 refit, solver-judged blunders, time per move, regressions of Elo on cheap predictors, and the honest answer to “why is our net the best?”.",
    roster: "The 15 strongest bots of part3, with label, family, search budget, size and both Elo fits, plus the five dropped bots and why.",
    arena_top15: "A clean re-run of the top-15 tournament with the fixed arena (same format as part3/summary.json, anchor ours_minimax_d4 = 1409).",
    sweep: "Supervised students of 8 sizes trained on the same 300k positions but labelled by different teachers (solver, heuristics, minimax depth 4, networks), each measured on 3,000 solver-scored positions and a gauntlet of games.",
    interp: "Linear probes for 21 game concepts at every layer, a census of what each channel responds to, block / channel-group knock-outs, policy-head weights, probe-direction patching and cross-model similarity.",
    interp_channels: "Per-channel 6x7 mean activation maps for the channel explorer (large; loaded only when you open a channel).",
    report: "The last arena (top8_o100_mcts_fixed) with every position solved: why the score nets out-rate Pons; linear probes, knock-outs and channel census for all nine arena nets; and every net in runs/ on one benchmark (policy, value and score heads)."
  };
  function renderOverview() {
    var body = $("overviewBody"); body.className = ""; body.innerHTML = "";
    var dg = el("div", { "class": "dsgrid" });
    Object.keys(DESC).forEach(function (k) {
      var e = (IDX && IDX[k]) || {};
      var ok = e.exists && e.valid !== false;
      dg.appendChild(el("div", { "class": "ds" + (ok ? " ok" : "") },
        '<span class="state" style="color:' + (ok ? C.blue : C.grey) + '">' + (ok ? "&#9679; available" : e.exists ? "&#9675; unreadable (being written?)" : "&#9675; not available yet") + "</span>" +
        "<h3>" + esc(e.label || k) + "</h3>" +
        '<dl class="kv">' + (e.exists ? "<dt>generated</dt><dd>" + esc(when(e.generated || e.mtime)) + "</dd><dt>file size</dt><dd>" + (e.bytes >= 1e6 ? (e.bytes / 1e6).toFixed(1) + " MB" : Math.round((e.bytes || 0) / 1e3) + " kB") + "</dd>" : "") +
        (isNum(e.n_findings) ? "<dt>findings</dt><dd>" + e.n_findings + "</dd>" : "") + "</dl><p>" + esc(DESC[k]) + "</p>"));
    });
    body.appendChild(dg);
    var g2 = el("div", { "class": "grid g2" });
    g2.appendChild(sweepStatusCard());
    var sc = card("Your selection", "What <code>python -m research.paper_outline --selected</code> puts in docs/PAPER_OUTLINE.md (change it on the Roster and Size & teachers tabs).", el("div", { id: "ovCounts" }));
    g2.appendChild(sc);
    body.appendChild(g2);
    renderOverviewCounts();
  }
  function renderOverviewCounts() {
    var box = $("ovCounts"); if (!box) return;
    var fs = allFindings(), per = {};
    fs.forEach(function (x) { per[x.section] = per[x.section] || [0, 0]; per[x.section][1]++; if (g(SEL.findings, x.f.id + ".include")) per[x.section][0]++; });
    var h = '<div class="tiles">';
    ["Arena", "Supervised sweep", "Interpretability"].forEach(function (s) { var p = per[s]; h += tile(s + " findings", p ? p[0] + " / " + p[1] : "–", p ? "selected / available" : "no findings yet"); });
    h += tile("Bots", countSel(SEL.roster), "selected on Roster") + tile("Runs", countSel(SEL.runs), "selected on Size & teachers") + "</div>";
    box.innerHTML = h;
  }
  function sweepStatusCard() {
    var S = D.sweep, st = S && S.status;
    if (!st) return card("Sweep progress", null, empty("sweep.json is not available yet. It appears after the first finished training run of <code>python -m research.sweep --run</code>."));
    var c = st.counts || {}, tot = st.total || ((c.done || 0) + (c.running || 0) + (c.queued || 0) + (c.failed || 0)) || 1;
    var seg = function (n, col, name) { return n ? '<span style="width:' + (100 * n / tot) + "%;background:" + col + '" data-tip="' + name + ": " + n + '"></span>' : ""; };
    var h = '<div class="progress" role="img" aria-label="' + (c.done || 0) + " of " + tot + ' runs done">' + seg(c.done, C.blue, "done") + seg(c.running, C.yellow, "running") + seg(c.failed, C.orange, "failed") + "</div>";
    h += '<p class="stats"><span class="st done">done ' + (c.done || 0) + '</span> <span class="st running">running ' + (c.running || 0) + '</span> <span class="st queued">queued ' + (c.queued || 0) + '</span> <span class="st failed">failed ' + (c.failed || 0) + "</span> of " + tot + "</p>";
    h += '<dl class="kv"><dt>state</dt><dd>' + esc(st.state || "–") + "</dd>" + (st.running ? "<dt>running now</dt><dd>" + esc(st.running) + "</dd>" : "") +
      "<dt>started</dt><dd>" + esc(when(st.started)) + "</dd><dt>last update</dt><dd>" + esc(when(st.updated || S.generated)) + "</dd>" +
      (st.eta_clock ? "<dt>estimated finish</dt><dd>" + esc(when(st.eta_clock)) + " (" + dur(st.eta_sec) + " left, estimate)</dd>" : "") + "</dl>";
    if (sweepActive()) h += '<p class="stats">This page re-checks the results files every 30 s while runs are queued or running.</p>';
    return card("Sweep progress", null, el("div", null, h));
  }
  function sweepActive() { var st = D.sweep && D.sweep.status; return !!(st && st.counts && ((st.counts.running || 0) + (st.counts.queued || 0) > 0) && st.state !== "done"); }

  // =================================================================== 2. roster
  function shortBot(n) { return String(n).replace(/^nn_/, "").replace("_mcts200", "+mcts").replace("_policy", "").replace("ntuple_td_ntuple", "ntuple").replace("_minimax", " mm").replace("ours mm", "ours_mm"); }
  function renderRoster() {
    var body = $("rosterBody"); body.innerHTML = "";
    var R = D.roster, A = D.arena_analysis, T = D.arena_top15;
    if (!R || !R.top15) { body.appendChild(empty("roster.json is not available yet (written by <code>python -m research.arena_analysis</code>).")); return; }
    var top = R.top15.slice().sort(function (a, b) { return (a.rank_top15 != null ? a.rank_top15 : 99) - (b.rank_top15 != null ? b.rank_top15 : 99); }), t15elo = T && T.elo ? T.elo : null;
    var bar = el("div", { "class": "bar" }, '<span class="hint">' + esc(R.anchor || "") + '</span><span class="spacer"></span><button type="button" class="btn" id="rsAll">Select all 15</button><button type="button" class="btn" id="rsNone">Select none</button>');
    body.appendChild(bar);
    // shared CI-bar domain
    var lo = Infinity, hi = -Infinity;
    top.forEach(function (r) { [r.top15_lo, r.top15_hi, r.part3_lo, r.part3_hi].forEach(function (v) { if (isNum(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }); });
    function ciBar(r) {
      if (!isNum(r.top15_elo) || !isFinite(lo)) return "";
      var w = 150, X = function (v) { return 4 + (v - lo) / (hi - lo) * (w - 8); };
      return '<svg class="cibar" width="' + w + '" height="14" aria-hidden="true"><line x1="4" x2="' + (w - 4) + '" y1="7" y2="7" stroke="#252c4d"/>' +
        (isNum(r.top15_lo) ? '<line x1="' + X(r.top15_lo) + '" x2="' + X(r.top15_hi) + '" y1="7" y2="7" stroke="#e9ecf8" stroke-width="1.5"/>' : "") +
        '<circle cx="' + X(r.top15_elo) + '" cy="7" r="4" fill="' + (r.group === "ours" ? C.gold : C.blue) + '"/></svg>';
    }
    var head = ["Use", "Rank", "Bot", "Family", "Group", "Top-15 Elo [95% CI]", "", "Part3 Elo [CI]"];
    if (t15elo) head.push("Re-run Elo [CI]");
    head = head.concat(["Score", "ms / move (avg)", "Params", "Search budget"]);
    var rows = top.map(function (r) {
      var cells = ['<input type="checkbox" data-bot="' + esc(r.name) + '" aria-label="use ' + esc(r.name) + ' in the paper"' + (SEL.roster[r.name] ? " checked" : "") + ">",
        r.rank_top15 != null ? r.rank_top15 : r.rank_part3,
        '<b>' + esc(r.label || r.name) + '</b><br><span class="muted sm">' + esc(r.name) + "</span>",
        '<span class="sm">' + esc(r.family || "") + "</span>",
        '<span class="badge ' + (r.group === "ours" ? "ours" : r.group === "open source" ? "os" : "base") + '">' + esc(r.group || "") + "</span>",
        ciCell(r.top15_elo, r.top15_lo, r.top15_hi, 0), ciBar(r), ciCell(r.part3_elo, r.part3_lo, r.part3_hi, 0)];
      if (t15elo) { var e = t15elo[r.name]; cells.push(e ? ciCell(e.elo, e.lo, e.hi, 0) : "–"); }
      cells = cells.concat([num(r.part3_score, 3), ms(r.part3_avg_ms) + (isNum(r.part3_max_ms) ? '<br><span class="muted sm">max ' + ms(r.part3_max_ms) + "</span>" : ""), params(r.params), '<span class="sm">' + esc(r.search_budget || "–") + "</span>"]);
      return { cls: (r.group === "ours" ? "ours " : "") + (SEL.roster[r.name] ? "sel" : ""), cells: cells };
    });
    var tbl = table(head, rows, { cls: "lt", caption: "Score and ms / move are from part3 (all 20 bots). Elo: Bradley-Terry with bootstrap 95% CI. The CI bar shows the top-15 refit on a shared scale (" + num(lo, 0) + " to " + num(hi, 0) + ")." });
    tbl.querySelectorAll("td").forEach(function (td) { td.classList.add("l"); });
    body.appendChild(card("Top 15", "Tick the bots you want to report (saved to selection.json).", tbl, true));
    tbl.addEventListener("change", function (e) { var b = e.target.getAttribute("data-bot"); if (!b) return; SEL.roster[b] = e.target.checked; e.target.closest("tr").classList.toggle("sel", e.target.checked); saveSoon(); });
    $("rsAll").onclick = function () { top.forEach(function (r) { SEL.roster[r.name] = true; }); saveSoon(); renderRoster(); };
    $("rsNone").onclick = function () { top.forEach(function (r) { SEL.roster[r.name] = false; }); saveSoon(); renderRoster(); };

    var series = [{ name: "part3 (all 20 bots)", color: C.grey, shape: "diamond", get: function (r) { return { v: r.part3_elo, lo: r.part3_lo, hi: r.part3_hi }; } },
                  { name: "top-15 refit (games among the 15)", color: C.blue, shape: "circle", get: function (r) { return { v: r.top15_elo, lo: r.top15_lo, hi: r.top15_hi }; } }];
    if (t15elo) series.push({ name: "top-15 re-run (fixed arena)", color: C.orange, shape: "square", get: function (r) { var e = t15elo[r.name]; return e ? { v: e.elo, lo: e.lo, hi: e.hi } : null; } });
    var dmRows = top.map(function (r) { return { label: shortBot(r.name), bold: r.name === BEST, top15_elo: r.top15_elo, top15_lo: r.top15_lo, top15_hi: r.top15_hi, part3_elo: r.part3_elo, part3_lo: r.part3_lo, part3_hi: r.part3_hi, name: r.name }; });
    var shifts = (A && A.elo_top15) || [];
    var shiftTbl = table(["Bot", "Rank (top-15)", "Rank (full pool)", "Change", "Shift in Elo"], shifts.map(function (r) { return [esc(r.name), r.rank, r.rank_full_pool, sgn(r.rank_change), sgn(r.shift_vs_full_pool, 1)]; }));
    var meta = A && A.elo_top15_meta;
    body.appendChild(grid("g2", [
      card("Full pool vs top-15 refit", "Same anchor value (ours_minimax_d4 = 1409 in both fits). Bars are 95% bootstrap CIs." + (meta ? " Top-15 refit uses " + fmtInt(meta.games) + " games." : ""), dumbbell(dmRows, series, { xLabel: "Elo", labelW: 150 })),
      card("Rank changes when the five weakest bots are removed", null, shiftTbl)]));

    var dr = R.dropped || [];
    body.appendChild(card("The five dropped bots", null, table(["Bot", "Part3 rank", "Part3 Elo", "Score", "Family", "Reason"], dr.map(function (r) { return [esc(r.name), r.rank_part3, num(r.part3_elo, 1), num(r.part3_score, 3), esc(r.family || ""), '<span class="sm">' + esc(r.reason || "") + "</span>"]; })), true));

    var M = A && A.head_to_head && A.head_to_head.top15_score_matrix;
    if (M && M.bots && M.matrix) {
      var hm = heatmap({ rows: M.bots.map(function (b) { return { label: shortBot(b) }; }), cols: M.bots.map(shortBot), cellW: 36, cellH: 22, labelW: 150, top: 110,
        val: function (i, j) { return M.matrix[i] ? M.matrix[i][j] : null; }, color: function (v) { return isNum(v) ? divColor((v - 0.5) * 2) : "#141832"; },
        text: function (v) { return v.toFixed(2); }, tip: function (i, j, v) { return M.bots[i] + " vs " + M.bots[j] + "\nmean score of row bot: " + (isNum(v) ? v.toFixed(4) : "–"); }, title: "head-to-head score matrix" });
      body.appendChild(card("Head-to-head scores (top 15)", esc(M.note || "Row bot's mean score against column bot."), [hm, scaleLegend("div", "0 (row bot loses all)", "1 (row bot wins all)", "0.5 = even")], true));
    }
    body.appendChild(top15Health(T));
  }
  function top15Health(T) {
    if (!T) return card("Top-15 re-run (fixed arena)", null, empty("experiments/results/arena/top15/summary.json is not there yet. When the clean top-15 tournament finishes, its Elo appears in the table and chart above and its health here."));
    var mt = T.meta || {}, pb = T.per_bot || {};
    var miss = Array.isArray(mt.missing) ? mt.missing.length : mt.missing, tos = Array.isArray(mt.timeouts) ? mt.timeouts.length : mt.timeouts;
    var fz = Array.isArray(T.forfeits) ? T.forfeits.length : T.forfeits;
    var line = "Games scheduled " + (mt.scheduled != null ? fmtInt(mt.scheduled) : "–") + ", recorded " + (mt.recorded != null ? fmtInt(mt.recorded) : fmtInt(T.games)) +
      ", missing " + (miss != null ? miss : "–") + ", timeouts " + (tos != null ? tos : "–") + ", forfeits " + (fz != null ? fz : "–") + (mt.created ? " (created " + when(mt.created) + ")" : "") + ". Elo anchor: " + esc(T.elo_anchor || "–") + ".";
    var bots = (T.bots || Object.keys(pb));
    var rows = bots.map(function (b) { var p = pb[b] || {}, e = (T.elo || {})[b] || {}; return [esc(b), fmtInt(p.games), num(p.score, 3), ciCell(e.elo, e.lo, e.hi, 0), ms(p.avg_ms), ms(p.max_ms), p.forfeits != null ? p.forfeits : "–"]; });
    return card("Top-15 re-run (fixed arena): tournament health", '<b>' + esc(line) + "</b>", table(["Bot", "Games", "Score", "Elo [CI]", "avg ms / move", "max ms / move", "Forfeits"], rows), true);
  }

  // =================================================================== 3. why ours is the best net
  function botColor(name, i) { if (name === BEST) return C.gold; return SEQ[i % SEQ.length]; }
  function renderBest() {
    var body = $("bestBody"); body.innerHTML = "";
    var A = D.arena_analysis;
    if (!A) { body.appendChild(empty("arena_analysis.json is not available yet.")); return; }
    var B = A.best_nn || {}, dec = B.decomposition || {}, ans = B.answer || {};
    // --- the honest answer
    var ah = '<div class="answer">' + (B.sense ? '<p class="note">' + esc(B.sense) + "</p>" : "") +
      '<div class="sup"><h4>Supported</h4>' + (ans.supported && ans.supported.length ? list(ans.supported) : '<p class="muted">–</p>') + "</div>" +
      '<div class="sug"><h4>Suggestive only (not established)</h4>' + (ans.suggestive && ans.suggestive.length ? list(ans.suggestive) : '<p class="muted">–</p>') + "</div>" +
      '<div class="need"><h4>Experiments still needed</h4>' + (ans.needed_experiments && ans.needed_experiments.length ? list(ans.needed_experiments) : '<p class="muted">–</p>') + "</div></div>";
    body.appendChild(card("The honest answer", "Copied from arena_analysis.json → best_nn.answer.", el("div", null, ah), true));

    // --- pairwise
    var pw = B.pairwise || [];
    var items = pw.map(function (p, i) { return { label: "vs " + shortBot(p.b), value: p.score, lo: p.lo, hi: p.hi, n: p.games, color: C.blue,
      tip: BEST + " vs " + p.b + "\nscore " + num(p.score, 3) + " [" + num(p.lo, 3) + ", " + num(p.hi, 3) + "] bootstrap\nWilson [" + num(p.wilson_lo, 3) + ", " + num(p.wilson_hi, 3) + "]\nW-D-L " + p.w + "-" + p.d + "-" + p.l + " in " + p.games + " games" }; });
    var pwt = table(["Opponent", "Games", "W-D-L", "Score [bootstrap CI]", "Wilson CI", "Sign test p", "Elo from score", "Elo diff [CI] (BT)", "P(ours better)", "Ours first / second"],
      pw.map(function (p) { var e = p.elo_diff || {}; return [esc(p.b), p.games, p.w + "-" + p.d + "-" + p.l, ciCell(p.score, p.lo, p.hi, 3), "[" + num(p.wilson_lo, 3) + ", " + num(p.wilson_hi, 3) + "]", num(p.p_sign_test, 4), sgn(p.elo_diff_from_score, 1), isNum(e.diff) ? sgn(e.diff, 1) + ' <span class="muted">[' + sgn(e.lo, 1) + ", " + sgn(e.hi, 1) + "]</span>" : "–", num(e.p_a_better, 3), num(g(p, "first.score"), 3) + " / " + num(g(p, "second.score"), 3)]; }));
    body.appendChild(grid("g2", [
      card("Head-to-head: " + BEST + " vs each rival", "Mean score per game (win 1, draw 0.5) with 95% bootstrap CI; n = games. The dashed line is an even match.", hbarsCI(items, { min: 0, max: 1, base: 0, ref: { v: 0.5, label: "0.5" }, fmt: function (v) { return num(v, 3); }, xLabel: "score of " + BEST + " (0 to 1)", labelW: 170 })),
      card("Details", null, pwt)]));

    // --- Elo differences, incl. sensitivity
    var rowsE = [];
    pw.forEach(function (p) { var e = p.elo_diff; if (e) rowsE.push({ an: "full pool (20 bots)", e: e }); });
    ((A.elo_top15_meta || {}).differences_vs_best || []).forEach(function (e) { rowsE.push({ an: "top-15 refit", e: e }); });
    var sens = A.elo_sensitivity_no_trivial_openings || {};
    (sens.differences || []).forEach(function (e) { rowsE.push({ an: "without openings " + (sens.excluded_openings || []).join(", "), e: e }); });
    var et = table(["Analysis", "A", "B", "A − B", "95% CI", "P(A better)", "Bootstrap reps"], rowsE.map(function (r) { var e = r.e; return { cls: e.b === "nn_az_6x64_mcts200" ? "ours" : "", cells: [esc(r.an), esc(e.a), esc(e.b), sgn(e.diff, 1), "[" + sgn(e.lo, 1) + ", " + sgn(e.hi, 1) + "]", num(e.p_a_better, 4), e.reps]}; }));
    var h2 = sens.h2h_sup_vs_az;
    var supaz = rowsE.filter(function (r) { return r.e.b === "nn_az_6x64_mcts200"; }).map(function (r) { return { label: r.an, value: r.e.diff, lo: r.e.lo, hi: r.e.hi, color: C.blue }; });
    body.appendChild(grid("g2", [
      card("Elo differences (Bradley-Terry bootstrap)", "Rows highlighted: our supervised net vs our AlphaZero net, the only rival whose interval touches zero in some analysis." + (h2 ? " Head-to-head without the excluded openings: " + h2.w + "-" + h2.d + "-" + h2.l + " in " + h2.games + " games, score " + num(h2.score, 3) + " [" + num(h2.lo, 3) + ", " + num(h2.hi, 3) + "], sign test p = " + num(h2.p_sign_test, 3) + "." : ""), et),
      card("sup − az across the three analyses", "If an interval crosses 0 the ranking of the two nets is not established by that analysis.", hbarsCI(supaz, { base: 0, ref: { v: 0, label: "0" }, fmt: function (v) { return sgn(v, 1); }, tickFmt: function (v) { return sgn(v, 0); }, xLabel: "Elo difference (sup − az)", labelW: 200 }))]));

    // --- blunders
    body.appendChild(blunderCard(dec.f_blunders));
    // --- points vs perfect play
    body.appendChild(perfectPlayCard(dec.vs_game_theoretic_expectation));
    // --- policy vs search
    body.appendChild(policySearchCard(dec.a_policy_quality, dec.b_search_gain));
    // --- time confound
    body.appendChild(timeCard(dec.g_time_confound, A.health));
    // --- training signal / size
    var c = dec.c_training_signal_and_cost || [];
    body.appendChild(card("Training signal and cost", "Arena Elo (part3) against what each net was trained on and what that took. Blank = not metered.", table(["Bot", "Signal", "Arena Elo", "Elo above minimax d4", "Wall h", "Wall h incl. labels / teachers", "Energy kWh", "Elo above d4 per hour", "Params", "Cost source"],
      c.map(function (r) { return { cls: r.bot === BEST ? "ours" : "", cells: [esc(r.bot), '<span class="sm">' + esc(r.signal || "") + "</span>", num(r.arena_elo, 1), sgn(r.elo_above_minimax_d4, 1), num(r.wall_hours, 2), num(isNum(r.wall_hours_incl_labels) ? r.wall_hours_incl_labels : r.wall_hours_incl_teachers, 2), num(r.energy_kwh, 2), num(r.elo_above_d4_per_hour, 1), params(r.params), '<span class="sm">' + esc(r.cost_source || "") + "</span>"] }; })), true));
    // --- regressions + predictors
    body.appendChild(regressionCard(A.regressions || [], A.predictor_ranking || []));
    // --- openings
    var e = dec.e_where_decided || {};
    var op = e.openings || [];
    if (op.length) {
      var cols = Object.keys(op[0]).filter(function (k) { return k.indexOf("score_") === 0; });
      body.appendChild(card("Where games are decided: the 20 openings", "First player's overall score " + num(e.first_player_score_overall, 4) + ". value = game-theoretic result for the first player after the opening (solver).", table(["#", "Moves", "Value (1st player)", "Games", "Observed 1st-player score", "Median plies", "Share ≤ 9 plies"].concat(cols.map(function (k) { return shortBot(k.slice(6)); })),
        op.map(function (o) { return [o.opening_id, esc(o.moves), esc(o.value_first_player), o.games, num(o.observed_score_first_player, 3), num(o.plies_median, 0), pct(o.share_games_le_9_plies)].concat(cols.map(function (k) { return num(o[k], 3); })); })), true));
    }
  }

  function blunderCard(F) {
    var per = F && F.per_bot;
    if (!per) return card("Blunders by game phase", null, empty("No blunder data in arena_analysis.json."));
    var bots = Object.keys(per);
    UI.blBots = UI.blBots || bots.filter(function (b) { return [BEST, "nn_az_6x64_mcts200", "bruneton_az", "nn_ppo_from_sup_policy", "pons_perfect"].indexOf(b) >= 0; });
    UI.blMetric = UI.blMetric || "error_rate";
    var wrap = el("div");
    var ctrl = el("div", { "class": "ctrl" }, select("blMetric", [{ v: "error_rate", t: "error rate (any move that worsens the result)" }, { v: "threw_win_rate", t: "threw a won position" }], UI.blMetric, "Metric"));
    var chips = el("div", { "class": "chips", role: "group", "aria-label": "bots to plot" });
    bots.forEach(function (b) { chips.innerHTML += '<label><input type="checkbox" value="' + esc(b) + '"' + (UI.blBots.indexOf(b) >= 0 ? " checked" : "") + "> " + esc(shortBot(b)) + "</label>"; });
    var chartBox = el("div");
    function draw() {
      chartBox.innerHTML = "";
      var cats = null, ser = [], k0 = 0;
      bots.forEach(function (b) {
        if (UI.blBots.indexOf(b) < 0) return;
        var bp = per[b].by_ply || [];
        if (!cats) cats = bp.map(function (x) { return x.bucket; });
        var col = b === BEST ? C.gold : [C.blue, C.orange, C.green, C.purple, C.sky, C.red, C.grey, C.white, C.yellow][k0++ % 9];
        ser.push({ name: shortBot(b), color: col, line: true, points: bp.map(function (x, k) {
          var nm = UI.blMetric === "error_rate" ? x.moves : x.moves_in_won;
          return { x: k, y: x[UI.blMetric], tip: b + "\nplies " + x.bucket + "\n" + UI.blMetric + ": " + pct(x[UI.blMetric], 2) + "\nn = " + nm + (UI.blMetric === "error_rate" ? " moves (" + x.errors + " errors)" : " moves in won positions (" + x.threw_win + " thrown)") }; }) });
      });
      chartBox.appendChild(xyChart({ series: ser, xCats: cats || [], yMin: 0, yFmt: function (v) { return pct(v, 1); }, xLabel: "ply of the move (bucket)", yLabel: UI.blMetric === "error_rate" ? "share of own moves that are errors" : "share of won positions thrown", xName: "plies", yName: UI.blMetric, emptyText: "Pick at least one bot" }));
    }
    ctrl.querySelector("select").addEventListener("change", function () { UI.blMetric = this.value; draw(); });
    chips.addEventListener("change", function () { UI.blBots = [].slice.call(chips.querySelectorAll("input:checked")).map(function (x) { return x.value; }); draw(); });
    wrap.appendChild(ctrl); wrap.appendChild(chips); wrap.appendChild(chartBox); draw();
    var rows = bots.slice().sort(function (a, b) { return (per[a].error_rate || 0) - (per[b].error_rate || 0); }).map(function (b) { var p = per[b]; return { cls: b === BEST ? "ours" : "", cells: [esc(b), fmtInt(p.moves), pct(p.error_rate, 2), pct(p.threw_win_rate, 2), pct(p.draw_to_loss_rate, 2), num(p.points_lost_per_100_moves, 2), pct(p.games_with_error_share, 1), num(p.first_error_ply_median, 0), fmtInt(p.games)] }; });
    var meta = F.meta || {};
    return card("Blunders by game phase (solver-judged)", "Every move of part3 judged by Pons' solver (" + esc(meta.mode || "") + "; " + fmtInt(meta.positions_total_moves) + " moves). Each bot reaches different positions, so this is descriptive, not a controlled comparison.",
      [wrap, table(["Bot", "Moves", "Error rate", "Threw a win", "Draw → loss", "Points lost / 100 moves", "Games with an error", "Median ply of 1st error", "Games"], rows)], true);
  }

  function perfectPlayCard(V) {
    if (!V) return card("Points lost against perfect play", null, empty("Not in arena_analysis.json."));
    var bots = Object.keys(V);
    var items = bots.filter(function (b) { return V[b].vs_pons; }).map(function (b) { var v = V[b].vs_pons; return { label: shortBot(b), value: v.points_lost, n: v.games, color: b === BEST ? C.gold : C.blue, bold: b === BEST, tip: b + " vs pons_perfect\npoints lost " + v.points_lost + " in " + v.games + " games\nscore " + num(v.score, 3) + ", expected from opening values " + num(v.expected, 3) }; })
      .sort(function (a, b) { return a.value - b.value; });
    var rows = bots.map(function (b) { var a = V[b].all || {}, p = V[b].vs_pons || {}; return { cls: b === BEST ? "ours" : "", cells: [esc(b), a.games, num(a.score, 3), num(a.expected, 3), sgn(a.delta_per_game, 3), num(a.points_lost, 1), num(a.points_gained, 1), p.games || "–", num(p.score, 3), num(p.expected, 3), num(p.points_lost, 1)] }; });
    return card("Points lost against perfect play", "Expected score = what perfect play would get from each opening's game-theoretic value. Points lost = points given away relative to that (vs everybody, and vs the Pons solver only).",
      [hbarsCI(items, { base: 0, min: 0, fmt: function (v) { return num(v, 1); }, xLabel: "points given away vs pons_perfect (fewer is better)", labelW: 170 }),
       table(["Bot", "Games", "Score", "Expected", "Δ per game", "Points lost", "Points gained", "Games vs Pons", "Score vs Pons", "Expected vs Pons", "Lost vs Pons"], rows)], true);
  }

  function policySearchCard(P, S) {
    var kids = [];
    var rows = (P && P.rows) || [];
    var runs = []; rows.forEach(function (r) { if (runs.indexOf(r.run) < 0) runs.push(r.run); });
    var searches = []; rows.forEach(function (r) { if (searches.indexOf(r.search) < 0) searches.push(r.search); });
    var find = function (run, s) { for (var i = 0; i < rows.length; i++) if (rows[i].run === run && rows[i].search === s) return rows[i]; return null; };
    UI.psMetric = UI.psMetric || "optimal";
    var box = el("div");
    function draw() {
      box.innerHTML = "";
      box.appendChild(groupBars({ categories: runs, series: searches.map(function (s, j) { return { name: s === "none" ? "policy head only" : s, color: SEQ[j], values: runs.map(function (r) { var x = find(r, s); return x ? x[UI.psMetric] : null; }) }; }),
        yMin: UI.psMetric === "ladder_elo" ? null : 0, yMax: UI.psMetric === "ladder_elo" ? null : 1, yFmt: UI.psMetric === "ladder_elo" ? tickFmt : function (v) { return pct(v, 0); }, rotate: runs.length > 5, yLabel: { optimal: "optimal-move rate", keeps_nl: "keeps a non-lost result", ladder_elo: "minimax-ladder Elo" }[UI.psMetric], emptyText: "No methods-table rows" }));
    }
    var ctrl = el("div", { "class": "ctrl" }, select("psMetric", [{ v: "optimal", t: "optimal-move rate" }, { v: "keeps_nl", t: "keeps won/drawn result" }, { v: "ladder_elo", t: "minimax-ladder Elo" }], UI.psMetric, "Metric"));
    ctrl.querySelector("select").addEventListener("change", function () { UI.psMetric = this.value; draw(); });
    kids.push(ctrl, box); draw();
    kids.push(table(["Run", "Search", "Optimal", "Keeps W/D", "Ladder Elo", "ms / move", "Params"], rows.map(function (r) { return [esc(r.run), esc(r.search), num(r.optimal, 3), num(r.keeps_nl, 4), num(r.ladder_elo, 1), num(r.ms_per_move, 2), params(r.params)]; }),
      { caption: P && P.note ? esc(P.note) : "" }));
    var left = card("Policy head alone vs + MCTS", "Methods table: " + (P && P.positions ? P.positions + " solver-scored positions." : ""), kids);
    // by ply for sup
    var E = P && P.sup_eval_by_ply, right;
    if (E) {
      var keys = ["nn_policy", "nn_mcts50", "nn_mcts200", "nn_mcts800"].filter(function (k) { return E[k] && E[k].by_bucket; });
      var cats = keys.length ? Object.keys(E[keys[0]].by_bucket) : [];
      var mm = E.minimax_reference || {};
      var ser = keys.map(function (k, i) { return { name: k.replace("nn_", "sup_6x64 ").replace("policy", "policy only") + " (all: " + pct(E[k].optimal) + ")", color: SEQ[i], line: true, points: cats.map(function (c, j) { var b = E[k].by_bucket[c]; return { x: j, y: b.optimal, tip: k + "\nplies " + c + ": optimal " + pct(b.optimal) + "\nn = " + b.n }; }) }; });
      var refs = ["d1", "d4", "d8"].filter(function (d) { return mm[d]; }).map(function (d, i) { return { y: mm[d].optimal, label: "minimax " + d + " (all plies)", color: [C.grey, C.purple, C.white][i] }; });
      right = card("sup_6x64 by game phase", "Optimal-move rate on the 3,000 held-out positions by ply bucket; dashed lines are minimax references over all plies.", xyChart({ series: ser, xCats: cats, refs: refs, yFmt: function (v) { return pct(v, 0); }, xLabel: "plies already played", yLabel: "optimal-move rate", xName: "plies", yName: "optimal" }));
    } else right = card("sup_6x64 by game phase", null, na());
    var gain = S || {};
    var gt = table(["Run", "Policy ladder Elo", "MCTS200 ladder Elo", "Ladder gain", "Policy optimal", "MCTS200 optimal", "Optimal gain"], (gain.methods_table || []).map(function (r) { return [esc(r.run), num(r.policy_ladder_elo, 1), num(r.mcts200_ladder_elo, 1), sgn(r.ladder_gain, 1), num(r.policy_optimal, 3), num(r.mcts200_optimal, 3), sgn(r.optimal_gain, 3)]; }),
      { caption: (gain.arena ? "In the arena (only " + esc(gain.arena.run) + " played with and without search): " + num(gain.arena.policy_arena_elo, 1) + " → " + num(gain.arena.mcts200_arena_elo, 1) + " Elo, gain " + sgn(gain.arena.arena_gain, 1) + " [" + (gain.arena.arena_gain_ci || []).map(function (v) { return sgn(v, 1); }).join(", ") + "]. " : "") + esc(gain.note || "") });
    return grid("g2", [left, right, card("What search adds", null, gt, true)]);
  }

  function timeCard(T, H) {
    var kids = [];
    if (T && T.arena_avg_ms) {
      var items = Object.keys(T.arena_avg_ms).map(function (b) { return { label: shortBot(b), value: T.arena_avg_ms[b], color: b === BEST ? C.gold : C.blue, bold: b === BEST, tip: b + "\navg " + ms(T.arena_avg_ms[b]) + " per move" + (T.arena_max_ms && T.arena_max_ms[b] ? "\nmax " + ms(T.arena_max_ms[b]) : "") }; }).sort(function (a, b) { return b.value - a.value; });
      kids.push(hbarsCI(items, { logX: true, fmt: ms, tickFmt: ms, xLabel: "average time per move in the arena (log scale)", labelW: 170 }));
      var mn = T.measured_now_single_thread_ms || {};
      var mrows = Object.keys(mn).filter(function (k) { return k !== "note"; }).map(function (k) { return [esc(k), ms(mn[k]), T.methods_table_ms && isNum(T.methods_table_ms[k]) ? ms(T.methods_table_ms[k]) : "–"]; });
      kids.push(table(["Configuration", "Measured single-thread", "Methods table"], mrows, { caption: esc(mn.note || "") + (isNum(T.slowdown_factor_policy_6x64) ? " Arena slowdown vs single-thread: policy 6x64 ×" + T.slowdown_factor_policy_6x64 + ", MCTS200 ×" + T.slowdown_factor_mcts200 + "." : "") }));
      ["explanation", "affects_strength", "equal_budget"].forEach(function (k) { if (T[k]) kids.push(el("p", { "class": "note" + (k === "affects_strength" ? " warn" : "") }, "<b>" + { explanation: "Why so slow", affects_strength: "Does it affect strength?", equal_budget: "Equal budget?" }[k] + "</b> " + esc(T[k]))); });
    } else kids.push(na("No timing data"));
    var left = card("The time-per-move confound", null, kids);
    var hk = [];
    if (H && H.per_bot) {
      var pb = H.per_bot.slice().sort(function (a, b) { return (b.think_hours || 0) - (a.think_hours || 0); });
      hk.push(el("p", { "class": "stats" }, "Wall time <b>" + num(H.wall_hours, 1) + " h</b> on " + H.workers + " workers; " + fmtInt(H.games) + " of " + fmtInt(H.expected_games) + " games recorded, " + H.forfeits + " forfeits, " + H.draws + " draws." + (H.wall_note ? " " + esc(H.wall_note) : "")));
      hk.push(hbarsCI(pb.map(function (r) { return { label: shortBot(r.name), value: r.think_hours, color: r.name === BEST ? C.gold : C.blue, bold: r.name === BEST, tip: r.name + "\nthinking " + num(r.think_hours, 1) + " h (" + pct(r.share) + " of all bot thinking)\navg " + ms(r.avg_ms) + ", max " + ms(r.max_ms) + "\n" + fmtInt(r.moves) + " moves" }; }), { base: 0, min: 0, fmt: function (v) { return num(v, 1) + " h"; }, xLabel: "bot thinking time, hours (summed over parallel workers)", labelW: 170 }));
      hk.push(table(["Bot", "Share of thinking", "Hours", "avg / move", "max / move"], pb.map(function (r) { return [esc(r.name), pct(r.share), num(r.think_hours, 1), ms(r.avg_ms), ms(r.max_ms)]; }), { caption: esc(H.note || "") }));
      if (H.missing && H.missing.length) hk.push(el("p", { "class": "stats" }, "Missing games: " + H.missing.map(function (m) { return esc((m.pair || []).join(" vs ") + " (opening " + m.opening_id + ", " + m.first_player + " first)"); }).join("; ")));
    } else hk.push(na("No tournament health data"));
    return grid("g2", [left, card("Where the tournament time went", null, hk)]);
  }

  function regressionCard(R, P) {
    if (!R.length) return card("Regressions", null, empty("No regressions in arena_analysis.json."));
    if (UI.regIdx == null || !R[UI.regIdx]) UI.regIdx = 0;
    var opts = R.map(function (r, i) { return { v: i, t: r.id.replace(/_/g, " ") + " — " + r.series + " (n=" + r.n + ")" }; });
    var ctrl = el("div", { "class": "ctrl" }, select("regSel", opts, UI.regIdx, "Regression"));
    var box = el("div");
    function draw() {
      var r = R[UI.regIdx]; box.innerHTML = "";
      box.appendChild(xyChart({ series: [{ name: r.series, color: C.blue, points: (r.points || []).map(function (p) { return { x: p.x, y: p.y, label: p.name, short: shortBot(p.name), big: p.name === BEST }; }), fit: { slope: r.slope, intercept: r.intercept } }],
        labels: true, xLabel: r.x, yLabel: r.y, xName: r.x, yName: r.y, height: 320 }));
      box.appendChild(el("p", { "class": "stats" }, "n = <b>" + r.n + "</b>" + (r.n < 10 ? " (small sample)" : "") + " · slope <b>" + num(r.slope, 3) + "</b> [" + (r.slope_ci || []).map(function (v) { return num(v, 3); }).join(", ") + "] (95% bootstrap CI) · intercept " + num(r.intercept, 2) + " · R² <b>" + num(r.r2, 3) + "</b> · Pearson r " + num(r.pearson_r, 3) + " · Spearman ρ " + num(r.spearman_rho, 3) +
        (r.slope_ci && r.slope_ci[0] < 0 && r.slope_ci[1] > 0 ? ' · <span style="color:' + C.orange + '">slope CI includes 0</span>' : "")));
    }
    ctrl.querySelector("select").addEventListener("change", function () { UI.regIdx = +this.value; draw(); });
    draw();
    var pr = hbarsCI(P.map(function (p) { return { label: p.id.replace("elo_vs_", "") + " (" + p.series + ")", value: p.r2, n: p.n, color: C.green, tip: p.id + " / " + p.series + "\nR² " + num(p.r2, 4) + ", Spearman ρ " + num(p.spearman_rho, 4) + "\nn = " + p.n }; }), { min: 0, max: 1, base: 0, fmt: function (v) { return num(v, 2); }, xLabel: "R² of a linear fit of arena Elo", labelW: 200 });
    return grid("g2", [card("Regressions: what predicts arena Elo?", "Pick any of the " + R.length + " fits in arena_analysis.json. Points are labelled by bot; ours is ringed.", [ctrl, box]),
      card("Predictor ranking", "Which cheap measurement best predicts arena Elo (same bots).", pr)]);
  }

  // =================================================================== 4. sweep
  var SMETRIC = {
    optimal: { t: "benchmark optimal-move rate", get: function (r) { return g(r, "bench.optimal"); }, pct: true, reg: "optimal_vs_log10_params", tq: "bench_optimal" },
    keeps_nl: { t: "benchmark keeps won/drawn result", get: function (r) { return g(r, "bench.keeps_nl"); }, pct: true, tq: "bench_keeps_nl" },
    value_acc: { t: "value-head accuracy (W/D/L)", get: function (r) { return g(r, "bench.value_acc"); }, pct: true },
    gauntlet: { t: "gauntlet mean score", get: function (r) { return g(r, "gauntlet.mean_score"); }, ci: function (r) { return g(r, "gauntlet.ci95"); }, reg: "gauntlet_vs_log10_params" },
    teacher_agreement: { t: "agreement with its teacher", get: function (r) { return r.teacher_agreement; }, pct: true },
    ms_per_pos: { t: "ms per position (policy head)", get: function (r) { return r.ms_per_pos; } }
  };
  function stBadge(s) { s = s || "unknown"; return '<span class="st ' + esc(s) + '">' + esc(s) + "</span>"; }
  function sizeOf(r) { return r.blocks + "x" + r.channels; }
  function teacherLabel(id) { var t = ((D.sweep && D.sweep.teachers) || []).filter(function (x) { return x.id === id; })[0]; return t ? t.label : id; }
  function renderSweep() {
    var body = $("sweepBody"); body.innerHTML = "";
    var S = D.sweep;
    if (!S) { body.appendChild(empty("sweep.json is not available yet. Start the sweep with <code>python -m research.sweep --run</code>; this tab fills in as runs finish (the page re-checks every 30 s).")); return; }
    var runs = S.runs || [], done = runs.filter(function (r) { return r.status === "done" && r.bench; }), an = S.analysis || {};
    var st = S.status || {}, c = st.counts || {};
    body.appendChild(el("div", { "class": "bar" }, "<span>" + stBadge(st.state === "done" ? "done" : st.state === "running" ? "running" : "queued") + " " + (c.done || 0) + " of " + (st.total || runs.length) + " runs done" + (c.failed ? ", " + c.failed + " failed" : "") + (st.running ? " · running <b>" + esc(st.running) + "</b>" : "") + (st.eta_clock && sweepActive() ? " · estimated finish " + esc(when(st.eta_clock)) : "") + '</span><span class="spacer"></span><span class="hint">generated ' + esc(when(S.generated)) + "</span>"));
    var notes = S.notes || [];
    var sizeNote = notes.filter(function (n) { return /billion/i.test(n); })[0];
    var sizes = S.sizes || [];
    var nk = [];
    nk.push(el("p", { "class": "note" }, "<b>The size ladder.</b> Parameters grow roughly with blocks × channels² (each residual block has two 3×3 convolutions of channels × channels weights), so doubling the width quadruples the size. " + (sizeNote ? esc(sizeNote) : "")));
    if (sizes.length) nk.push(table(["Size", "Blocks", "Channels", "Params"], sizes.map(function (z) { return [z.blocks + "x" + z.channels, z.blocks, z.channels, fmtInt(z.params)]; })));
    var det = el("details", { "class": "notes" }, "<summary>Method notes from sweep.json (" + notes.length + ")</summary>" + list(notes));
    nk.push(det);
    body.appendChild(card("What was trained", null, nk, true));
    body.appendChild(sizeChartCard(S, done, an));
    body.appendChild(teacherCard(S, an));
    body.appendChild(grid("g2", [evoCard(an.evo_curve), searchCard(an.search_vs_nosearch)]));
    body.appendChild(grid("g2", [distillCard(an.distill_vs_direct), paretoCard(done, an)]));
    body.appendChild(runsCard(runs));
  }

  function sizeChartCard(S, done, an) {
    UI.swMetric = UI.swMetric || "optimal"; UI.swBy = UI.swBy || "teacher"; if (UI.swRefs == null) UI.swRefs = false;
    var ctrl = el("div", { "class": "ctrl" }, select("swMetric", Object.keys(SMETRIC).map(function (k) { return { v: k, t: SMETRIC[k].t }; }), UI.swMetric, "Y axis") + " " +
      select("swBy", [{ v: "teacher", t: "one series per teacher" }, { v: "group", t: "one series per group (A/B/C)" }], UI.swBy, "Series") +
      ' <label><input type="checkbox" id="swRefs"' + (UI.swRefs ? " checked" : "") + "> teacher's own score as reference lines</label>");
    var box = el("div"), side = el("div"), chips = el("div", { "class": "chips", role: "group", "aria-label": "series to show" });
    UI.swHide = UI.swHide || {};
    function drawChips(keys) {
      chips.innerHTML = keys.map(function (k) { return '<label><input type="checkbox" value="' + esc(k) + '"' + (UI.swHide[k] ? "" : " checked") + "> " + esc(k) + "</label>"; }).join("");
    }
    chips.addEventListener("change", function (e) { UI.swHide[e.target.value] = !e.target.checked; draw(true); });
    function draw(keepChips) {
      var M = SMETRIC[UI.swMetric]; box.innerHTML = "";
      var keys = [], by = {};
      done.forEach(function (r) { var k = UI.swBy === "teacher" ? r.teacher : "group " + r.group; if (!by[k]) { by[k] = []; keys.push(k); } by[k].push(r); });
      if (!keepChips) drawChips(keys);
      var allKeys = keys.slice();
      keys = keys.filter(function (k) { return !UI.swHide[k]; });
      var ser = keys.map(function (k) {
        var i = allKeys.indexOf(k);   // stable colour / marker per series when others are hidden
        var regs = UI.swBy === "teacher" && M.reg ? g(an, "size_regression." + k + "." + M.reg) : null;
        var nm = (UI.swBy === "teacher" ? teacherLabel(k) : k) + (regs && isNum(regs.slope) ? " (" + (M.pct ? sgn(100 * regs.slope, 1) + " pts" : sgn(regs.slope, 3)) + " per 10× params, R² " + num(regs.r2, 2) + ", n=" + regs.n + (regs.small_n ? ", small n" : "") + ")" : "");
        return { name: nm, color: SEQ[i % SEQ.length], shape: SHAPES[i % SHAPES.length], line: !regs || !isNum(regs.slope), fit: regs && isNum(regs.slope) ? { slope: regs.slope, intercept: regs.intercept, log10x: true } : null,
          points: by[k].map(function (r) { var v = M.get(r), ci = M.ci ? M.ci(r) : null; return { x: r.params, y: v, lo: ci ? ci[0] : null, hi: ci ? ci[1] : null,
            tip: r.name + "\nteacher " + r.teacher + ", " + sizeOf(r) + ", seed " + r.seed + "\n" + fmtInt(r.params) + " params\n" + M.t + ": " + (M.pct ? pct(v, 2) : num(v, 4)) + (ci ? " [" + num(ci[0], 3) + ", " + num(ci[1], 3) + "]" : "") + (r.gauntlet && UI.swMetric === "gauntlet" ? "\n" + r.gauntlet.games + " games" : "") }; }) };
      });
      var refs = [];
      if (UI.swRefs && M.tq && UI.swBy === "teacher") keys.forEach(function (k) { var i = allKeys.indexOf(k); var t = (S.teachers || []).filter(function (x) { return x.id === k; })[0]; if (t && isNum(t[M.tq])) refs.push({ y: t[M.tq], label: "teacher " + k, color: SEQ[i % SEQ.length] }); });
      box.appendChild(xyChart({ series: ser, logX: true, refs: refs, height: 330, xLabel: "parameters (log scale)", yLabel: M.t, yFmt: M.pct ? function (v) { return pct(v, 0); } : tickFmt, xName: "params", yName: M.t, emptyText: "No finished runs yet" }));
      if (!M.reg) box.appendChild(el("p", { "class": "stats" }, "No fitted line for this metric (sweep.json only has regressions for optimal-rate and gauntlet score); points are joined in size order."));
    }
    ctrl.addEventListener("change", function (e) { if (e.target.id === "swMetric") UI.swMetric = e.target.value; if (e.target.id === "swBy") UI.swBy = e.target.value; if (e.target.id === "swRefs") UI.swRefs = e.target.checked; draw(); });
    draw();
    // regression table + seed noise side by side
    var SR = an.size_regression || {}, rr = [];
    Object.keys(SR).forEach(function (t) { ["optimal_vs_log10_params", "gauntlet_vs_log10_params"].forEach(function (k) { var r = SR[t][k]; if (r) rr.push([esc(t), k.replace("_vs_log10_params", ""), r.n, isNum(r.slope) ? sgn(r.slope, 4) : "–", r.slope_ci95 ? "[" + r.slope_ci95.map(function (v) { return num(v, 4); }).join(", ") + "]" : "–", num(r.r2, 3), r.small_n ? '<span class="st failed">small n</span>' : "", r.slope_ci95 && r.slope_ci95[0] < 0 && r.slope_ci95[1] > 0 ? "CI includes 0" : ""]); }); });
    side.appendChild(table(["Teacher", "Y", "n", "Slope per 10× params", "95% CI (bootstrap)", "R²", "", ""], rr, { caption: "OLS of the metric on log10(params) for each teacher, from analysis.size_regression." }));
    var sn = an.seed_noise || [];
    side.appendChild(el("h3", { style: "margin:14px 0 4px;font-size:.95rem" }, "Seed noise (same teacher and size, seed 0 vs seed 1)"));
    side.appendChild(sn.length ? table(["Size", "Teacher", "Optimal s0", "s1", "|diff|", "Gauntlet s0", "s1", "|diff|"], sn.map(function (r) { return [r.size, esc(r.teacher), num(r.optimal_s0, 4), num(r.optimal_s1, 4), "<b>" + num(r.abs_diff_optimal, 4) + "</b>", num(r.gauntlet_s0, 3), num(r.gauntlet_s1, 3), "<b>" + num(r.abs_diff_gauntlet, 3) + "</b>"]; }),
      { caption: "Differences between runs smaller than these seed-to-seed differences should not be read as real." }) : el("p", { "class": "muted" }, "No seed-1 twins finished yet, so seed noise is unknown: treat small differences between single runs with caution (benchmark SE ≈ 0.0065 per the notes)."));
    return grid("", [card("Benchmark quality vs network size", "Each point is one finished student. Dashed = OLS fit on log10(params) from sweep.json; whiskers on gauntlet points are 95% CIs. Untick series to declutter.", [ctrl, chips, box], true), card("Fits and seed noise", null, side, true)]);
  }

  function teacherCard(S, an) {
    var T = S.teachers || [], tvs = an.teacher_vs_student || {};
    var sizes = Object.keys(tvs);
    var stu = function (tid, size) { var e = tvs[size]; if (!e) return null; return (e.rows || []).filter(function (r) { return r.teacher === tid; })[0] || null; };
    var head = ["Teacher", "Kind", "Search depth", "Gen", "Own optimal", "Own keeps W/D", "Agrees with Pons", "Labelling time"];
    sizes.forEach(function (s) { head.push(s + " student", "Student − teacher"); });
    var rows = T.map(function (t) {
      var cells = ['<b>' + esc(t.id) + '</b><br><span class="muted sm">' + esc(t.label || "") + "</span>", esc(t.kind), t.search_depth == null ? "–" : t.search_depth, t.gen == null ? "–" : t.gen, pct(t.bench_optimal), pct(t.bench_keeps_nl), pct(t.agree_pons), isNum(t.label_sec) ? dur(t.label_sec) : "–"];
      sizes.forEach(function (s) { var r = stu(t.id, s); cells.push(r ? pct(r.student_optimal) : "–", r ? '<b style="color:' + (r.delta_optimal > 0 ? C.blue : C.orange) + '">' + (r.delta_optimal > 0 ? "▲ " : "▼ ") + sgn(100 * r.delta_optimal, 1) + " pts</b>" : "–"); });
      return cells;
    });
    var kids = [table(head, rows, { caption: "Own = the teacher's own move on the same 3,000 benchmark positions; student = policy head of the student trained on that teacher's labels (no search). ▲ student better than its teacher, ▼ worse." + (sizes.length ? "" : " No size has two finished teachers yet, so the student columns are empty.") })];
    if (sizes.length) {
      var stx = [];
      var ser = sizes.map(function (s, i) {
        var e = tvs[s], r = e.student_vs_teacher || {};
        stx.push("<b>" + s + "</b>: " + (e.rows || []).length + " teachers, slope " + num(r.slope, 3) + (r.slope_ci95 ? " [" + r.slope_ci95.map(function (v) { return num(v, 3); }).join(", ") + "]" : "") + ", R² " + num(r.r2, 3) + (isNum(e.pearson_r) ? ", Pearson r " + num(e.pearson_r, 3) : "") + (r.small_n ? " <span class=\"st failed\">small n</span>" : ""));
        return { name: s + " students (n=" + (e.rows || []).length + (r.small_n ? ", small n" : "") + ")", color: SEQ[i % SEQ.length],
          points: (e.rows || []).map(function (q) { return { x: q.teacher_optimal, y: q.student_optimal, tip: s + " student of " + q.teacher + "\nteacher optimal " + pct(q.teacher_optimal, 2) + "\nstudent optimal " + pct(q.student_optimal, 2) + " (" + sgn(100 * q.delta_optimal, 1) + " pts)" + (isNum(q.student_gauntlet) ? "\nstudent gauntlet " + num(q.student_gauntlet, 3) : "") }; }),
          fit: isNum(r.slope) && !r.small_n ? { slope: r.slope, intercept: r.intercept } : null };
      });
      kids.push(xyChart({ series: ser, height: 320, refs: [], xLabel: "teacher's own optimal-move rate", yLabel: "student optimal-move rate (no search)", xName: "teacher", yName: "student", yFmt: function (v) { return pct(v, 1); }, xFmt: function (v) { return pct(v, 0); } }));
      kids.push(el("p", { "class": "stats" }, "Student vs teacher (OLS, from analysis.teacher_vs_student; fitted lines drawn only when n is not small): " + stx.join(" · ")));
    }
    return card("Teachers: own quality vs the students trained from them", null, kids, true);
  }

  function evoCard(E) {
    var rows = (E && E.rows) || [];
    var evo = rows.filter(function (r) { return r.search_depth === 0 || r.search_depth == null && /heur/.test(r.teacher); });
    var mm = rows.filter(function (r) { return /^mm_/.test(r.teacher); });
    var pick = function (k, name, col, dash) { return { name: name, color: col, line: true, dash: dash, points: evo.map(function (r) { return { x: r.gen, y: r[k], tip: r.teacher + " (gen " + r.gen + ")\n" + name + ": " + pct(r[k], 2) }; }) }; };
    var ser = [pick("teacher_optimal", "teacher (heuristic, no search)", C.grey), pick("student_2x32_optimal", "2x32 student", C.blue), pick("student_6x64_optimal", "6x64 student", C.orange)];
    var refs = mm.filter(function (r) { return isNum(r.teacher_optimal); }).map(function (r, i) { return { y: r.teacher_optimal, label: r.teacher + " teacher (depth 4)", color: [C.purple, C.green][i % 2], legend: r.teacher + " teacher (minimax depth 4)" }; });
    var regs = (E && E.regressions) || {};
    var txt = Object.keys(regs).map(function (k) { var r = regs[k]; return k.replace(/_/g, " ") + ": " + (isNum(r.slope) ? sgn(r.slope, 5) + " per generation, R² " + num(r.r2, 3) + ", n=" + r.n + (r.small_n ? " (small n)" : "") : "not enough points"); }).join(" · ");
    var t = table(["Teacher", "Gen", "Depth", "Teacher optimal", "2x32 student", "2x32 gauntlet", "6x64 student", "6x64 gauntlet"], rows.map(function (r) { return [esc(r.teacher), r.gen, r.search_depth == null ? "–" : r.search_depth, pct(r.teacher_optimal), pct(r.student_2x32_optimal), num(r.student_2x32_gauntlet, 3), pct(r.student_6x64_optimal), num(r.student_6x64_gauntlet, 3)]; }));
    return card("Evolved heuristic: generation → teacher and student quality", "Heuristic weights from evolution generation 0 (hand-made) to 60, used as no-search teachers. Horizontal lines: the depth-4 minimax teachers.", [xyChart({ series: ser, refs: refs, xLabel: "evolution generation of the heuristic", yLabel: "benchmark optimal-move rate", yFmt: function (v) { return pct(v, 0); }, xName: "generation", yName: "optimal", height: 260 }), el("p", { "class": "stats" }, esc(txt)), t]);
  }

  function searchCard(rows) {
    rows = rows || [];
    var cats = rows.map(function (r) { return r.heuristic; });
    var ser = [
      { name: "teacher, no search", color: C.grey, values: rows.map(function (r) { return r.no_search_teacher_optimal; }) },
      { name: "teacher, depth-4 search", color: C.purple, values: rows.map(function (r) { return r.search_d4_teacher_optimal; }) },
      { name: "2x32 student of no-search teacher", color: C.sky, values: rows.map(function (r) { return r.no_search_student_2x32_optimal; }) },
      { name: "2x32 student of depth-4 teacher", color: C.blue, values: rows.map(function (r) { return r.search_d4_student_2x32_optimal; }) },
      { name: "6x64 student of no-search teacher", color: C.red, values: rows.map(function (r) { return r.no_search_student_6x64_optimal; }) },
      { name: "6x64 student of depth-4 teacher", color: C.orange, values: rows.map(function (r) { return r.search_d4_student_6x64_optimal; }) }];
    return card("Does a searching teacher make a better student?", "Same heuristic with and without depth-4 minimax as the labeller. Missing bars = run not finished.", groupBars({ categories: cats, series: ser, yMin: 0, yMax: 1, yFmt: function (v) { return pct(v, 0); }, yLabel: "benchmark optimal-move rate", emptyText: "No data yet" }));
  }

  function distillCard(Dd) {
    var rows = (Dd && Dd.rows) || [], T = (Dd && Dd.teachers) || {};
    UI.dvMetric = UI.dvMetric || "optimal";
    var ctrl = el("div", { "class": "ctrl" }, select("dvMetric", [{ v: "optimal", t: "optimal-move rate" }, { v: "keeps_nl", t: "keeps won/drawn" }, { v: "gauntlet", t: "gauntlet score" }], UI.dvMetric, "Metric"));
    var box = el("div");
    function draw() {
      var k = UI.dvMetric; box.innerHTML = "";
      var refs = [];
      if (k === "optimal") Object.keys(T).forEach(function (t, i) { if (isNum(T[t].bench_optimal)) refs.push({ y: T[t].bench_optimal, label: t + " teacher (" + params(T[t].params) + ")", color: [C.white, C.yellow][i % 2] }); });
      box.appendChild(groupBars({ categories: rows.map(function (r) { return r.size + (r.params ? " (" + params(r.params) + ")" : ""); }), refs: refs,
        series: [{ name: "direct (solver labels)", color: C.blue, values: rows.map(function (r) { return r["direct_" + k]; }) }, { name: "distilled from 6x64", color: C.orange, values: rows.map(function (r) { return r["from_6x64_" + k]; }) }, { name: "distilled from 8x128", color: C.green, values: rows.map(function (r) { return r["from_8x128_" + k]; }) }],
        yMin: 0, yMax: k === "gauntlet" ? null : 1, yFmt: k === "gauntlet" ? tickFmt : function (v) { return pct(v, 0); }, yLabel: k, xLabel: "student size" }));
    }
    ctrl.querySelector("select").addEventListener("change", function () { UI.dvMetric = this.value; draw(); });
    draw();
    var t = table(["Size", "Direct", "From 6x64", "Retained", "Compression", "From 8x128", "Retained", "Compression"], rows.map(function (r) { return [r.size, pct(r.direct_optimal), pct(r.from_6x64_optimal), pct(r.from_6x64_retained), r.from_6x64_compression ? "×" + r.from_6x64_compression : "–", pct(r.from_8x128_optimal), pct(r.from_8x128_retained), r.from_8x128_compression ? "×" + r.from_8x128_compression : "–"]; }), { caption: "Optimal-move rate. Retained = student / teacher optimal-rate; compression = teacher params / student params." });
    return card("Distillation: direct vs distilled from a bigger net", null, [ctrl, box, t]);
  }

  function paretoCard(done, an) {
    UI.prX = UI.prX || "params";
    var ctrl = el("div", { "class": "ctrl" }, select("prX", [{ v: "params", t: "parameters" }, { v: "ms_per_pos", t: "ms per position" }], UI.prX, "Cost axis"));
    var box = el("div");
    function draw() {
      box.innerHTML = "";
      var xk = UI.prX, front = (xk === "params" ? an.pareto_params : an.pareto_ms) || [];
      var onF = {}; front.forEach(function (p) { onF[p.name] = 1; });
      box.appendChild(xyChart({ series: [
        { name: "all finished runs", color: C.grey, points: done.filter(function (r) { return !onF[r.name]; }).map(function (r) { return { x: r[xk], y: g(r, "bench.optimal"), label: r.name }; }) },
        { name: "Pareto front (best optimal-rate at that cost or less)", color: C.orange, line: true, shape: "diamond", points: front.map(function (p) { return { x: p[xk], y: p.optimal, label: p.name }; }) }],
        logX: true, xFmt: xk === "params" ? params : function (v) { return v + " ms"; }, xLabel: xk === "params" ? "parameters (log)" : "ms per position (log)", yLabel: "benchmark optimal-move rate", yFmt: function (v) { return pct(v, 0); }, xName: xk, yName: "optimal", height: 270 }));
    }
    ctrl.querySelector("select").addEventListener("change", function () { UI.prX = this.value; draw(); });
    draw();
    return card("Pareto front: quality vs cost", null, [ctrl, box]);
  }

  function runsCard(runs) {
    UI.rs = UI.rs || { key: "name", dir: 1, group: "", status: "", q: "" };
    var cols = [
      { k: "use", t: "Use" }, { k: "name", t: "Run", v: function (r) { return r.name; } }, { k: "group", t: "Group", v: function (r) { return r.group; } }, { k: "teacher", t: "Teacher", v: function (r) { return r.teacher; } },
      { k: "params", t: "Params", v: function (r) { return r.params; } }, { k: "seed", t: "Seed", v: function (r) { return r.seed; } }, { k: "status", t: "Status", v: function (r) { return r.status; } },
      { k: "optimal", t: "Optimal", v: function (r) { return g(r, "bench.optimal"); } }, { k: "keeps_nl", t: "Keeps W/D", v: function (r) { return g(r, "bench.keeps_nl"); } },
      { k: "gauntlet", t: "Gauntlet [CI]", v: function (r) { return g(r, "gauntlet.mean_score"); } }, { k: "agree", t: "Teacher agree", v: function (r) { return r.teacher_agreement; } },
      { k: "ms", t: "ms / pos", v: function (r) { return r.ms_per_pos; } }, { k: "train", t: "Train time", v: function (r) { return r.train_sec; } }];
    var wrap = el("div");
    var groups = []; runs.forEach(function (r) { if (groups.indexOf(r.group) < 0) groups.push(r.group); });
    var ctrl = el("div", { "class": "ctrl" }, select("rsGroup", [{ v: "", t: "all groups" }].concat(groups.map(function (x) { return { v: x, t: "group " + x }; })), UI.rs.group, "Show") + " " +
      select("rsStatus", [{ v: "", t: "any status" }, "done", "running", "queued", "failed"], UI.rs.status) +
      ' <input type="search" id="rsQ" placeholder="filter by name" aria-label="filter runs by name" value="' + esc(UI.rs.q) + '"> <button type="button" class="btn" id="rsDone">Select all finished</button> <button type="button" class="btn" id="rsClear">Clear selection</button>');
    var tbox = el("div", { "class": "scroll" });
    function draw() {
      var rs = runs.filter(function (r) { return (!UI.rs.group || r.group === UI.rs.group) && (!UI.rs.status || r.status === UI.rs.status) && (!UI.rs.q || r.name.toLowerCase().indexOf(UI.rs.q.toLowerCase()) >= 0); });
      var col = cols.filter(function (c) { return c.k === UI.rs.key; })[0] || cols[1];
      rs.sort(function (a, b) { var x = col.v(a), y = col.v(b); if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1; return (x < y ? -1 : x > y ? 1 : 0) * UI.rs.dir; });
      var h = '<table class="t sortable"><thead><tr>' + cols.map(function (c) { return '<th scope="col"' + (c.k === UI.rs.key ? ' aria-sort="' + (UI.rs.dir > 0 ? "ascending" : "descending") + '"' : "") + ">" + (c.v ? '<button type="button" data-k="' + c.k + '">' + c.t + "</button>" : c.t) + "</th>"; }).join("") + "</tr></thead><tbody>";
      rs.forEach(function (r) {
        var gci = g(r, "gauntlet.ci95");
        h += '<tr class="' + (SEL.runs[r.name] ? "sel" : "") + '"><td><input type="checkbox" data-run="' + esc(r.name) + '" aria-label="use ' + esc(r.name) + '"' + (SEL.runs[r.name] ? " checked" : "") + "></td><td>" + esc(r.name) + (r.optional ? ' <span class="muted sm">optional</span>' : "") + "</td><td>" + esc(r.group) + "</td><td>" + esc(r.teacher) + "</td><td>" + params(r.params) + " <span class=\"muted sm\">" + sizeOf(r) + "</span></td><td>" + r.seed + "</td><td>" + stBadge(r.status) + (r.error ? '<br><span class="sm" title="' + esc(r.error) + '">' + esc(String(r.error).slice(0, 60)) + "</span>" : "") + "</td><td>" + pct(g(r, "bench.optimal")) + "</td><td>" + pct(g(r, "bench.keeps_nl")) + "</td><td>" + ciCell(g(r, "gauntlet.mean_score"), gci && gci[0], gci && gci[1], 3) + "</td><td>" + pct(r.teacher_agreement) + "</td><td>" + num(r.ms_per_pos, 3) + "</td><td>" + dur(r.train_sec) + "</td></tr>";
      });
      if (!rs.length) h += '<tr><td colspan="' + cols.length + '" class="l muted">No runs match.</td></tr>';
      tbox.innerHTML = h + "</tbody></table>";
    }
    ctrl.addEventListener("change", function (e) { if (e.target.id === "rsGroup") UI.rs.group = e.target.value; if (e.target.id === "rsStatus") UI.rs.status = e.target.value; draw(); });
    ctrl.addEventListener("input", function (e) { if (e.target.id === "rsQ") { UI.rs.q = e.target.value; draw(); } });
    tbox.addEventListener("click", function (e) { var b = e.target.closest("button[data-k]"); if (!b) return; var k = b.getAttribute("data-k"); if (UI.rs.key === k) UI.rs.dir = -UI.rs.dir; else { UI.rs.key = k; UI.rs.dir = 1; } draw(); var nb = tbox.querySelector('button[data-k="' + k + '"]'); if (nb) nb.focus(); });
    tbox.addEventListener("change", function (e) { var n = e.target.getAttribute("data-run"); if (!n) return; SEL.runs[n] = e.target.checked; e.target.closest("tr").classList.toggle("sel", e.target.checked); saveSoon(); });
    ctrl.querySelector("#rsDone").onclick = function () { runs.forEach(function (r) { if (r.status === "done") SEL.runs[r.name] = true; }); saveSoon(); draw(); };
    ctrl.querySelector("#rsClear").onclick = function () { SEL.runs = {}; saveSoon(); draw(); };
    draw();
    wrap.appendChild(ctrl); wrap.appendChild(tbox);
    return card("All runs (" + runs.length + ")", "Click a column header to sort. Tick runs to list them in the paper outline. Status is shown as text (done / running / queued / failed).", wrap, true);
  }

  // =================================================================== 5. interpretability
  var GROUP_ORDER = ["geometry", "tactics", "own_move", "opp_move", "phase"];
  var GROUP_NAME = { geometry: "geometry", tactics: "tactics", own_move: "own move", opp_move: "opponent move", phase: "phase" };
  function renderInterp() {
    var body = $("interpBody"); body.innerHTML = "";
    var I = D.interp;
    if (!I) {
      body.appendChild(empty("interp.json is not available yet (written by <code>python -m research.interp</code>). When it appears this tab shows: the layer × concept probe matrix, a channel explorer with 6×7 activation maps, block and channel-group knock-outs, policy-head weight maps, probe-direction patching and a cross-model comparison. The page checks for it every 30 s."));
      return;
    }
    var models = I.models || [];
    if (!models.length) { body.appendChild(empty("interp.json has no models yet.")); return; }
    if (!UI.imodel || !models.some(function (m) { return m.id === UI.imodel; })) UI.imodel = models[0].id;
    var M = models.filter(function (m) { return m.id === UI.imodel; })[0];
    var chk = (I.checks || {})[M.id] || {};
    var bar = el("div", { "class": "bar" }, select("imodel", models.map(function (m) { return { v: m.id, t: m.id + " (" + m.blocks + "x" + m.channels + ", " + params(m.params) + ")" }; }), UI.imodel, "Model") +
      '<span class="hint">run ' + esc(M.run || "") + (isNum(chk.bench_agreement) ? " · agreement with solver on its benchmark " + pct(chk.bench_agreement) + (chk.bench_n ? " (n=" + chk.bench_n + ")" : "") : "") + (isNum(chk.forward_max_abs_diff_vs_numpy) ? " · torch vs numpy max |diff| " + chk.forward_max_abs_diff_vs_numpy : "") + "</span>");
    bar.querySelector("select").addEventListener("change", function () { UI.imodel = this.value; UI.iconcept = null; UI.ich = null; renderInterp(); });
    body.appendChild(bar);
    if (I.notes && I.notes.length) body.appendChild(el("details", { "class": "notes" }, "<summary>Method notes from interp.json (" + I.notes.length + ")</summary>" + list(I.notes)));
    body.appendChild(probeCard(I, M));
    body.appendChild(channelCard(I, M));
    body.appendChild(ablationCard(I, M));
    body.appendChild(headCard(I, M));
    body.appendChild(grid("g2", [patchCard(I, M), crossCard(I, M)]));
  }

  function probeCard(I, M) {
    var P = (I.probes || {})[M.id], PS = (I.probe_summary || {})[M.id] || {};
    if (!P) return card("Which layer knows what (linear probes)", null, empty("No probes for this model."));
    var concepts = (I.concepts || []).filter(function (c) { return P[c.id]; });
    concepts.sort(function (a, b) { return GROUP_ORDER.indexOf(a.group) - GROUP_ORDER.indexOf(b.group); });
    var layers = (M.layers || []).filter(function (l) { return concepts.some(function (c) { return P[c.id][l]; }); });
    UI.pmode = UI.pmode || "flat";
    var ctrl = el("div", { "class": "ctrl" }, select("pmode", [{ v: "flat", t: "flat probe (all activations)" }, { v: "pooled", t: "pooled probe (per-channel spatial mean)" }, { v: "delta", t: "flat minus input-layer probe" }, { v: "shuffled", t: "shuffled-label control" }], UI.pmode, "Show") +
      (concepts.length ? " " + select("pconcept", concepts.map(function (c) { return { v: c.id, t: c.label }; }), UI.iconcept || concepts[0].id, "Concept") : ""));
    var box = el("div"), line = el("div");
    function val(ci, li) {
      var c = concepts[ci], p = P[c.id][layers[li]]; if (!p) return null;
      if (UI.pmode === "delta") { var b = P[c.id].input; return b && isNum(p.flat) && isNum(b.flat) ? p.flat - b.flat : null; }
      return p[UI.pmode];
    }
    function norm(ci, v) { var ch = g(PS, concepts[ci].id + ".majority"); if (!isNum(ch)) ch = concepts[ci].metric === "auc" ? 0.5 : 0; return (v - ch) / (1 - ch || 1); }
    function draw() {
      box.innerHTML = "";
      var activeRow = concepts.map(function (c) { return c.id; }).indexOf(UI.iconcept);
      box.appendChild(heatmap({ rows: concepts.map(function (c) { return { label: c.id + " (" + c.metric + ")", group: GROUP_NAME[c.group] || c.group }; }), cols: layers, cellW: 40, cellH: 20, labelW: 170, top: 80, activeRow: activeRow,
        val: val, text: function (v) { return v.toFixed(2); },
        color: function (v) { return !isNum(v) ? "#141832" : UI.pmode === "delta" ? divColor(v / 0.5) : seqColor(Math.max(0, v)); },
        mark: function (ci, li) { var s = PS[concepts[ci].id]; if (!s || UI.pmode === "delta") return null; if (s.best_layer === layers[li]) return C.gold; if (s.first_layer === layers[li]) return "#ffffff"; return null; },
        tip: function (ci, li, v) { var c = concepts[ci], s = PS[c.id] || {}, p = P[c.id][layers[li]] || {}; return c.label + "\nlayer " + layers[li] + " (" + c.metric + ")\nflat " + num(p.flat, 4) + " · pooled " + num(p.pooled, 4) + " · shuffled " + num(p.shuffled, 4) + "\nchance (majority) " + num(s.majority, 4) + (isNum(norm(ci, p.flat)) ? "\nnormalised skill (flat) " + num(norm(ci, p.flat), 3) : "") + (s.best_layer === layers[li] ? "\n★ best layer" : "") + (s.first_layer === layers[li] ? "\nfirst layer reaching 90% of best" : ""); },
        onRow: function (ci) { UI.iconcept = concepts[ci].id; var sel = $("pconcept"); if (sel) sel.value = UI.iconcept; draw(); }, title: "probe matrix" }));
      box.appendChild(UI.pmode === "delta" ? scaleLegend("div", "−0.5", "+0.5", "gain over the raw input (same metric)") : scaleLegend("seq", "0", "1", "probe score (AUC / accuracy / R²); gold outline = best layer, white = first layer reaching 90% of best"));
      drawLine();
    }
    function drawLine() {
      line.innerHTML = "";
      var c = concepts.filter(function (x) { return x.id === UI.iconcept; })[0] || concepts[0]; if (!c) return;
      UI.iconcept = c.id;
      var s = PS[c.id] || {};
      var mk = function (k, name, col, dash) { return { name: name, color: col, line: true, dash: dash, points: layers.map(function (l, i) { var p = P[c.id][l]; return { x: i, y: p ? p[k] : null }; }) }; };
      line.appendChild(el("h3", { style: "font-size:.95rem;margin:12px 0 2px" }, esc(c.label)));
      line.appendChild(xyChart({ series: [mk("flat", "flat probe", C.blue), mk("pooled", "pooled probe", C.orange), mk("shuffled", "shuffled-label control", C.grey, true)], xCats: layers, height: 230,
        refs: [{ y: s.majority, label: "chance / majority", color: C.white, legend: "chance (majority)" }], xLabel: "layer (input → stem → blocks → heads)", yLabel: c.metric, xName: "layer", yName: c.metric }));
      line.appendChild(el("p", { "class": "stats" }, "best <b>" + esc(s.best_layer || "–") + "</b> (" + num(s.best, 4) + ") · first layer reaching 90% of best: <b>" + esc(s.first_layer || "–") + "</b> · input baseline " + num(s.input_baseline, 4) + " · normalised best skill " + num(s.norm_best, 3) + " · survives pooling: " + (s.survives_pooling ? "yes" : "no") + " · max shuffled " + num(s.shuffled_max, 4)));
    }
    ctrl.addEventListener("change", function (e) { if (e.target.id === "pmode") { UI.pmode = e.target.value; draw(); } if (e.target.id === "pconcept") { UI.iconcept = e.target.value; draw(); } });
    draw();
    return card("Which layer knows what (linear probes)", "Rows: concepts grouped by kind; columns: layers. Click a concept (or use the Concept menu) to see its curve across layers.", [ctrl, box, line], true);
  }

  // census labels: fixed colours (and a short text on every tile, so colour is never the only cue)
  var LABEL_COL = { "mixed": "#8e96b5", "own-threat": C.yellow, "opponent-threat": C.purple, "phase: early": C.blue, "phase: late": C.orange,
                    "occupancy copy": C.green, "column detector": C.sky, "row/height detector": "#e9ecf8" };
  var LABEL_ABBR = { "mixed": "mix", "own-threat": "own-T", "opponent-threat": "opp-T", "phase: early": "early", "phase: late": "late",
                     "occupancy copy": "occ", "column detector": "col", "row/height detector": "row", "dead": "dead" };
  function labelColor(lab) {
    if (lab === "dead") return "#2a2f4d";
    if (!LABEL_COL[lab]) { var n = Object.keys(LABEL_COL).length; LABEL_COL[lab] = ["#a6d854", "#fdb462", C.red, "#bcbddc"][n % 4]; }
    return LABEL_COL[lab];
  }
  function labelAbbr(lab) { return LABEL_ABBR[lab] || String(lab || "").replace(/[^A-Za-z]/g, "").slice(0, 5); }
  function channelCard(I, M) {
    var CH = (I.channels || {})[M.id], CC = (I.channel_counts || {})[M.id] || {};
    if (!CH) return card("Channel explorer", null, empty("No channel census for this model."));
    var layers = Object.keys(CH);
    if (!UI.ilayer || layers.indexOf(UI.ilayer) < 0) UI.ilayer = layers[0];
    var allLabels = []; layers.forEach(function (l) { Object.keys(CC[l] || {}).forEach(function (k) { if (allLabels.indexOf(k) < 0) allLabels.push(k); }); });
    allLabels.sort(); allLabels.forEach(labelColor);
    var ctrl = el("div", { "class": "ctrl" }, select("ilayer", layers, UI.ilayer, "Layer"));
    var tiles = el("div", { "class": "chgrid", role: "listbox", "aria-label": "channels" }), detail = el("div");
    function drawTiles() {
      var rows = CH[UI.ilayer] || [];
      tiles.innerHTML = rows.map(function (r, i) { return '<button type="button" class="chtile' + (r.dead ? " dead" : "") + (UI.ich === i ? " on" : "") + '" data-i="' + i + '" style="background:' + labelColor(r.label) + '" data-tip="' + esc("channel " + r.ch + "\nlabel: " + r.label + "\ntop concept: " + (r.top_concept || "–") + " (r = " + num(r.top_corr, 3) + ")") + '">' + r.ch + "<small>" + esc(labelAbbr(r.label)) + "</small></button>"; }).join("");
      drawDetail();
    }
    function drawDetail() {
      detail.innerHTML = "";
      var rows = CH[UI.ilayer] || [], r = rows[UI.ich];
      if (!r) { detail.appendChild(el("p", { "class": "muted" }, "Click a channel tile to see its mean activation map.")); return; }
      var left = el("div"), right = el("div");
      left.appendChild(el("h3", { style: "font-size:.95rem;margin:4px 0" }, esc(UI.ilayer + " / channel " + r.ch + " — " + r.label)));
      var mapBox = el("div", null, '<p class="muted">Loading the activation map…</p>');
      left.appendChild(mapBox);
      loadChannels().then(function (H) {
        var m = g(H, "channels." + M.id + "." + UI.ilayer + "." + UI.ich);
        mapBox.innerHTML = "";
        if (!m || !m.map) { mapBox.appendChild(na("No map for this channel in interp_channels.json")); return; }
        mapBox.appendChild(boardMap(m.map, { title: UI.ilayer + " ch " + r.ch + " mean activation", cell: 30, maxW: 320, scaleLabel: "mean activation (row 0 of the data = top row)" }));
        mapBox.appendChild(grid("g2", [
          groupBars({ categories: ["top", "5", "4", "3", "2", "bottom"], series: [{ name: "row mean", color: C.sky, values: m.row || [] }], height: 170, xLabel: "board row" }),
          groupBars({ categories: ["1", "2", "3", "4", "5", "6", "7"], series: [{ name: "column mean", color: C.orange, values: m.col || [] }], height: 170, xLabel: "board column" })]));
      }).catch(function () { mapBox.innerHTML = ""; mapBox.appendChild(na("interp_channels.json is not available yet")); });
      right.appendChild(groupBars({ categories: ["early", "mid", "late"], series: [{ name: "mean pooled activation by phase", color: C.purple, values: r.phase || [] }], height: 170, xLabel: "game phase" }));
      var cont = r.content || [];
      right.appendChild(table(["Property", "Value"], [["label", esc(r.label)], ["dead", r.dead ? "yes" : "no"], ["top concept", esc(r.top_concept || "–") + " (r = " + num(r.top_corr, 3) + ")"], ["active (share > 0)", pct(r.active)], ["row selectivity", num(r.row_sel, 3)], ["column selectivity", num(r.col_sel, 3)], ["positional share of variance", num(r.pos_frac, 3)], ["phase index (late−early)", num(r.phase_idx, 3)],
        ["content means: empty playable / empty other / mine / theirs", cont.map(function (v) { return num(v, 3); }).join(" / ")], ["content η²", num(r.content_eta2, 3)]]));
      if (r.corr) right.appendChild(table(["Correlation with", "r"], Object.keys(r.corr).sort(function (a, b) { return Math.abs(r.corr[b] || 0) - Math.abs(r.corr[a] || 0); }).map(function (k) { return [esc(k), num(r.corr[k], 3)]; })));
      detail.appendChild(el("div", { "class": "chdetail" })).appendChild(left);
      detail.firstChild.appendChild(right);
    }
    ctrl.querySelector("select").addEventListener("change", function () { UI.ilayer = this.value; UI.ich = null; drawTiles(); });
    tiles.addEventListener("click", function (e) { var b = e.target.closest(".chtile"); if (!b) return; UI.ich = +b.getAttribute("data-i"); tiles.querySelectorAll(".chtile").forEach(function (t) { t.classList.toggle("on", t === b); }); drawDetail(); });
    drawTiles();
    var counts = stackBars({ categories: layers, series: allLabels.map(function (lab) { return { name: lab, color: labelColor(lab), values: layers.map(function (l) { return (CC[l] || {})[lab] || 0; }) }; }), yLabel: "channels", rotate: layers.length > 8, xLabel: "layer" });
    return grid("g2", [card("Channel explorer", "Tiles are coloured by the census label and also show it as text (striped = dead). Board maps follow the play-page orientation: top row at the top.", [ctrl, legend(allLabels.map(function (l) { return { name: l + " (" + labelAbbr(l) + ")", color: labelColor(l) }; })), tiles, detail], true),
      card("Channel labels per layer", "Rule-based labels from correlations / mean maps (thresholds in the notes).", counts, true)]);
  }
  var CH_PROMISE = null;
  function loadChannels() {
    if (D.interp_channels) return Promise.resolve(D.interp_channels);
    if (!CH_PROMISE) CH_PROMISE = getJSON("/api/research/data/interp_channels").then(function (d) { D.interp_channels = d; CH_PROMISE = null; return d; }, function (e) { CH_PROMISE = null; throw e; });
    return CH_PROMISE;
  }

  var AMETRIC = { agree: "Δ agreement with solver", value_acc: "Δ value accuracy", kl: "KL(intact ‖ ablated)", dv: "mean |Δ value|", missed_win: "Δ missed wins", missed_block: "Δ missed blocks" };
  function ablationCard(I, M) {
    var AB = I.ablation || {}, blocks = (AB.blocks || {})[M.id] || [], groups = (AB.groups || {})[M.id] || [], intact = (AB.intact || {})[M.id], top = (AB.top_channels || {})[M.id];
    var modes = []; blocks.forEach(function (b) { if (modes.indexOf(b.mode) < 0) modes.push(b.mode); });
    UI.amode = modes.indexOf(UI.amode) >= 0 ? UI.amode : (modes.indexOf("mean_branch") >= 0 ? "mean_branch" : modes[0]);
    UI.ametric = UI.ametric || "agree";
    var gmodes = []; groups.forEach(function (x) { if (gmodes.indexOf(x.mode) < 0) gmodes.push(x.mode); });
    UI.gmode = gmodes.indexOf(UI.gmode) >= 0 ? UI.gmode : (gmodes.indexOf("mean") >= 0 ? "mean" : gmodes[0]);
    var ctrl = el("div", { "class": "ctrl" }, (modes.length ? select("amode", modes, UI.amode, "Block knock-out") + " " : "") + select("ametric", Object.keys(AMETRIC).map(function (k) { return { v: k, t: AMETRIC[k] }; }), UI.ametric, "Metric") +
      (gmodes.length ? " " + select("gmode", gmodes, UI.gmode, "Channel-group ablation") : ""));
    var box = el("div");
    function draw() {
      box.innerHTML = "";
      var bs = blocks.filter(function (b) { return b.mode === UI.amode; }), k = UI.ametric, cats = bs.map(function (b) { return b.layer; });
      var f = function (v) { return num(v, 4); };
      box.appendChild(grid("g2", [
        el("div", null, "<h3 style=\"font-size:.92rem;margin:4px 0\">By game phase</h3>"),
        el("div", null, "<h3 style=\"font-size:.92rem;margin:4px 0\">By position type</h3>")]));
      var gg = box.firstChild;
      gg.children[0].appendChild(groupBars({ categories: cats, series: [{ name: "all", color: C.grey, values: bs.map(function (b) { return g(b, "overall." + k); }) }].concat(["early", "mid", "late"].map(function (p, i) { return { name: p, color: [C.blue, C.orange, C.green][i], values: bs.map(function (b) { return g(b, "by_phase." + p + "." + k); }) }; })), yFmt: f, yLabel: AMETRIC[k] + " (ablated − intact)", height: 230, emptyText: "No block ablations" }));
      gg.children[1].appendChild(groupBars({ categories: cats, series: ["own_win", "must_block", "quiet"].map(function (p, i) { return { name: { own_win: "own immediate win", must_block: "must block", quiet: "quiet" }[p], color: [C.sky, C.red, C.purple][i], values: bs.map(function (b) { return g(b, "by_type." + p + "." + k); }) }; }), yFmt: f, yLabel: AMETRIC[k], height: 230, emptyText: "No block ablations" }));
      if (intact) box.appendChild(el("p", { "class": "stats" }, "Intact model: " + Object.keys(intact.all || intact).slice(0, 8).map(function (q) { var v = (intact.all || intact)[q]; return isNum(v) ? q + " " + (q === "n" ? fmtInt(v) : num(v, 4)) : ""; }).filter(Boolean).join(" · ")));
      // groups
      var gs = groups.filter(function (x) { return x.mode === UI.gmode; });
      var items = gs.filter(function (x) { return x.specificity; }).map(function (x) { var s = x.specificity; return { label: x.layer + " / " + x.group + " (" + x.n_channels + ")", value: s.block_minus_win, lo: s.null_mean - 1.96 * s.null_std, hi: s.null_mean + 1.96 * s.null_std, color: s.pct >= 0.95 ? C.orange : s.pct <= 0.05 ? C.sky : C.grey,
        tip: x.layer + " / " + x.group + " (" + x.n_channels + " channels, " + x.mode + " ablation)\nmissed blocks − missed wins: " + num(s.block_minus_win, 4) + "\nrandom-channel null: mean " + num(s.null_mean, 4) + ", sd " + num(s.null_std, 4) + "\npercentile vs null: " + pct(s.pct, 0) }; });
      box.appendChild(el("h3", { style: "font-size:.92rem;margin:14px 0 2px" }, "Channel-group knock-out: specificity vs random channels (" + (gs[0] ? gs[0].mode : "–") + " ablation)"));
      box.appendChild(hbarsCI(items, { fmt: function (v) { return num(v, 3); }, ref: { v: 0, label: "0" }, xLabel: "Δ missed blocks − Δ missed wins (bar = the group; whisker = null mean ± 1.96 sd of equal-size random sets)", labelW: 230, emptyText: "No channel groups with a specificity score",
        legend: [{ name: "above 95% of the null (block-specific)", color: C.orange }, { name: "below 5% of the null (win-specific)", color: C.sky }, { name: "within the null", color: C.grey }] }));
      box.appendChild(table(["Layer", "Group", "Channels", "Mode", "Δ agree", "Δ missed win", "Δ missed block", "pct vs null (agree)", "Specificity pct"], gs.map(function (x) { return [esc(x.layer), esc(x.group), x.n_channels, esc(x.mode), num(g(x, "effect.agree"), 4), num(g(x, "effect.missed_win"), 4), num(g(x, "effect.missed_block"), 4), pct(g(x, "null.agree.pct"), 0), pct(g(x, "specificity.pct"), 0)]; })));
    }
    ctrl.addEventListener("change", function (e) { if (e.target.id === "amode") UI.amode = e.target.value; if (e.target.id === "ametric") UI.ametric = e.target.value; if (e.target.id === "gmode") UI.gmode = e.target.value; draw(); });
    draw();
    var kids = [ctrl, box];
    if (top && (top.policy || top.value)) kids.push(grid("g2", [
      el("div", null, "<h3 style=\"font-size:.92rem\">Single channels that move the policy most</h3>").appendChild(table(["Layer", "Ch", "Label", "KL", "Δ agree"], (top.policy || []).map(function (r) { return [esc(r.layer), r.ch, esc(r.label), num(r.kl, 4), num(r.d_agree, 4)]; }))).parentNode,
      el("div", null, "<h3 style=\"font-size:.92rem\">Single channels that move the value most</h3>").appendChild(table(["Layer", "Ch", "Label", "|Δ value|", "Δ value acc"], (top.value || []).map(function (r) { return [esc(r.layer), r.ch, esc(r.label), num(r.dv, 4), num(r.d_value_acc, 4)]; }))).parentNode]));
    return card("Knock-outs: which blocks and channels play depends on", "Δ = ablated minus intact on the test positions.", kids, true);
  }

  function headCard(I, M) {
    var H = (I.heads || {})[M.id];
    if (!H || !H.policy_fc) return card("Policy head weights", null, empty("No head weights for this model."));
    var pol = H.policy_fc, pc = pol[0] ? pol[0].length : 0;
    UI.hch = UI.hch == null ? "abs" : UI.hch;
    var opts = [{ v: "abs", t: "sum of |w| over the " + pc + " policy channels" }, { v: "sum", t: "signed sum over channels" }];
    for (var i = 0; i < pc; i++) opts.push({ v: String(i), t: "policy channel " + i + " only" });
    var ctrl = el("div", { "class": "ctrl" }, select("hch", opts, UI.hch, "Weights"));
    var box = el("div");
    function mapFor(j) {
      var out = [];
      for (var r = 0; r < 6; r++) { out.push([]); for (var c = 0; c < 7; c++) { var v = 0; if (UI.hch === "abs" || UI.hch === "sum") { for (var k = 0; k < pc; k++) { var w = pol[j][k][r][c]; v += UI.hch === "abs" ? Math.abs(w) : w; } } else v = pol[j][+UI.hch][r][c]; out[r].push(v); } }
      return out;
    }
    function draw() {
      box.innerHTML = "";
      var maps = [], mx = 0; for (var j = 0; j < 7; j++) { maps.push(mapFor(j)); maps[j].forEach(function (row) { row.forEach(function (v) { mx = Math.max(mx, Math.abs(v)); }); }); }
      var gridEl = el("div", { "class": "boards7" });
      maps.forEach(function (m, j) {
        var f = el("figure"); f.appendChild(boardMap(m, { cell: 14, max: mx, seq: UI.hch === "abs", scale: false, noCols: false, hiCol: j, title: "logit for column " + (j + 1), maxW: 140, fmt: function (v) { return num(v, 4); } }));
        f.appendChild(el("figcaption", null, "column " + (j + 1) + (H.policy_locality ? " · locality " + num(H.policy_locality[j], 2) : ""))); gridEl.appendChild(f);
      });
      box.appendChild(gridEl);
      box.appendChild(scaleLegend(UI.hch === "abs" ? "seq" : "div", UI.hch === "abs" ? "0" : "−" + num(mx, 3), num(mx, 3), "weight (shared scale across the 7 boards)"));
      if (H.policy_own_col_frac) box.appendChild(el("p", { "class": "stats" }, "Share of |w| on the logit's own board column: " + H.policy_own_col_frac.map(function (v, j) { return (j + 1) + ": " + num(v, 2); }).join(" · ") + " (1/7 ≈ 0.14 = no locality)."));
    }
    ctrl.querySelector("select").addEventListener("change", function () { UI.hch = this.value; draw(); });
    draw();
    var kids = [ctrl, box];
    if (H.value_fc && H.value_fc.length) {
      var vm = []; for (var r = 0; r < 6; r++) { vm.push([]); for (var c = 0; c < 7; c++) { var s = 0; H.value_fc.forEach(function (ch) { s += ch[r][c]; }); vm[r].push(s); } }
      kids.push(el("h3", { style: "font-size:.92rem;margin:14px 0 4px" }, "Value head: win-minus-loss read-out, summed over value channels"));
      kids.push(boardMap(vm, { cell: 26, maxW: 260, title: "value read-out", scaleLabel: "orange pushes toward a win, blue toward a loss" }));
      if (H.value_note) kids.push(el("p", { "class": "stats" }, esc(H.value_note)));
    }
    return card("Head weights per output column", "Each small board shows which board cells the final policy layer reads for that column's logit (highlighted column number).", kids, true);
  }

  function patchCard(I, M) {
    var P = (I.patching || {})[M.id];
    if (!P || !P.per_column) return card("Probe-direction patching", null, empty("No patching results for this model."));
    var pc = P.per_column;
    var chart = groupBars({ categories: pc.map(function (r) { return "col " + (r.col + 1); }), series: [{ name: "add direction (quiet positions)", color: C.orange, values: pc.map(function (r) { return g(r, "add.z"); }) }, { name: "subtract (must-block positions)", color: C.sky, values: pc.map(function (r) { return g(r, "subtract.z"); }) }], yLabel: "z vs " + P.random_draws + " random directions", height: 220 });
    var t = table(["Col", "Step", "n add", "Add: Δ logit (rel)", "Δ P(col)", "z", "n sub", "Sub: Δ logit (rel)", "Δ P(col)", "z"], pc.map(function (r) { return [r.col + 1, num(r.alpha, 3), r.n_add, num(g(r, "add.d_logit_rel"), 3), num(g(r, "add.d_prob"), 3), num(g(r, "add.z"), 1), r.n_sub, num(g(r, "subtract.d_logit_rel"), 3), num(g(r, "subtract.d_prob"), 3), num(g(r, "subtract.z"), 1)]; }));
    return card("Probe-direction patching", "Adding the “" + esc(P.concept) + "” probe direction at " + esc(P.layer) + " and measuring the change in that column's logit.", [chart, t]);
  }

  function crossCard(I, M) {
    var X = I.cross_model || {}, cka = X.cka || {}, others = Object.keys(cka);
    if (!others.length) return card("Cross-model comparison", null, empty("No cross-model results (needs two or more models)."));
    if (!UI.xmodel || others.indexOf(UI.xmodel) < 0) UI.xmodel = others[0];
    var primary = ((I.models || []).filter(function (m) { return others.indexOf(m.id) < 0; })[0] || {}).id || "primary";
    var ctrl = el("div", { "class": "ctrl" }, select("xmodel", others, UI.xmodel, "Compare with"));
    var box = el("div");
    function draw() {
      box.innerHTML = "";
      var c = cka[UI.xmodel];
      box.appendChild(heatmap({ rows: c.rows.map(function (r) { return { label: r }; }), cols: c.cols, cellW: 34, cellH: 18, labelW: 90, top: 70, val: function (i, j) { return c.matrix[i] ? c.matrix[i][j] : null; }, color: function (v) { return seqColor(v); }, text: function (v) { return v.toFixed(2); }, tip: function (i, j, v) { return "rows: " + primary + ", columns: " + UI.xmodel + "\n" + c.rows[i] + " vs " + c.cols[j] + ": CKA " + num(v, 3); } }));
      box.appendChild(scaleLegend("seq", "0", "1", "linear CKA (rows = " + primary + ", columns = " + UI.xmodel + ")"));
      var cd = (X.concept_diffs || {})[UI.xmodel] || [];
      box.appendChild(table(["Concept", primary + " best", UI.xmodel + " best", "Δ normalised skill (" + primary + " − other)"], cd.slice(0, 12).map(function (d) { return [esc(d.concept), num(d.primary_best, 3), num(d.other_best, 3), sgn(d.diff_norm, 3)]; })));
    }
    ctrl.querySelector("select").addEventListener("change", function () { UI.xmodel = this.value; draw(); });
    draw();
    return card("Cross-model comparison", "Layer-by-layer similarity (linear CKA) and concepts one model encodes better than the other.", [ctrl, box]);
  }

  // =================================================================== 6-8. report.json (research/report_data.py)
  // Nets vs Pons · inside all arena nets · every net benchmarked.  Same tables as the report artifact.
  function RT(name) { return (D.report && D.report.tables && D.report.tables[name]) || []; }
  function rFind(name, key, val) { return RT(name).filter(function (r) { return r[key] === val; })[0] || {}; }
  function reportMissing(body) {
    body.appendChild(empty("report.json is not available yet. Run <code>python -m research.arena_vs_pons</code>, <code>python -m research._next_turn_check</code>, <code>python -m research.bench_all</code>, <code>python -m research.interp_bruneton</code> and <code>python -m research.interp --models sup_6x64 az_6x64 ppo_from_sup ppo_6x64 distill_az_2x32 score_8x128_s0 score_8x128_s1 score_8x128_s2 --out experiments/results/research/interp_arena.json</code>, then <code>python -m research.report_data</code>."));
  }
  var ENS = "nn_ens3_score_8x128_mcts200", ENSN = "3-net score ensemble + MCTS200", PONS = "pons_perfect";

  // ---------------------------------------------------------------- 6. nets vs Pons
  function renderVsPons() {
    var body = $("vsponsBody"); body.innerHTML = "";
    if (!D.report) { reportMissing(body); return; }
    var rec = RT("records"), E = rFind("records", "bot", ENS), P = rFind("records", "bot", PONS);
    var m = function (net, st) { return rFind("matched", "key", net + "|" + st); };
    var mAll = m(ENSN, "all"), mLost = m(ENSN, "lost"), mWon = m(ENSN, "won"), pLost = m("3-net score ensemble (policy)", "lost");
    var nt = rFind("next_turn", "key", ENSN + "|lost");
    body.appendChild(el("p", { "class": "note" },
      "The 3-net score ensemble with MCTS rates <b>" + num(E.elo, 0) + "</b> against Pons at <b>" + num(P.elo, 0) + "</b>, but loses to Pons head to head (" +
      num(E.h2h_w) + "–" + num(E.h2h_d) + "–" + num(E.h2h_l) + ", score " + num(E.h2h_score, 3) + ", sign test p = " + num(E.h2h_p, 2) + "). Its rating comes from beating the rest of the field more often than Pons does (" +
      num(E.score_field, 3) + " vs " + num(P.score_field, 3) + " against the same 7 opponents), not from being stronger than Pons. Arena <code>" + esc(D.report.arena_run) + "</code>, " +
      fmtInt(D.report.games) + " games, all " + fmtInt(D.report.positions_solved) + " positions solved."));
    body.appendChild(el("div", { "class": "tiles" },
      tile("Arena Elo, ensemble + MCTS vs Pons", num(E.elo, 0) + " vs " + num(P.elo, 0), "95% intervals [" + num(E.elo_lo, 0) + ", " + num(E.elo_hi, 0) + "] and [" + num(P.elo_lo, 0) + ", " + num(P.elo_hi, 0) + "] overlap") +
      tile("Head to head vs Pons", num(E.h2h_w) + "–" + num(E.h2h_d) + "–" + num(E.h2h_l), "Elo predicts " + num(E.elo_pred_vs_pons, 3) + ", actual " + num(E.h2h_score, 3)) +
      tile("Score vs the 7 common opponents", num(E.score_field, 3) + " vs " + num(P.score_field, 3), "ensemble + MCTS vs Pons") +
      tile("Matched games vs Pons", num(mAll.net_better) + " / " + num(mAll.net_worse), "net line better / worse, p " + (mAll.sign_p < 0.001 ? "< 0.001" : num(mAll.sign_p, 3)))));

    var sorted = rec.slice().sort(function (a, b) { return b.elo - a.elo; });
    body.appendChild(card("Ratings measured four ways",
      "Elo gives each bot one number, which only works if results are transitive. Here they are not: Pons beats both score nets head to head, but they score more than Pons against the weaker bots. <b>Elo from Pons games only</b> converts each bot's score against Pons into Elo; <b>top-group Elo</b> is refit on games among the six strongest bots only; <b>strength-weighted score</b> weights each opponent by its own overall score.",
      table(["Bot", "Arena Elo [95% CI]", "Elo from Pons games only", "Top-group Elo", "Score vs Pons", "Elo-predicted vs Pons", "Strength-weighted score", "Raw score"],
        sorted.map(function (r) { return { cls: r.bot === PONS ? "sel" : "", cells: ["<b>" + esc(r.name) + "</b>", ciCell(r.elo, r.elo_lo, r.elo_hi, 0), num(r.elo_from_pons_h2h, 0), num(r.elo_top_group, 0), num(r.h2h_score, 3), num(r.elo_pred_vs_pons, 3), num(r.strength_weighted, 3), num(r.score_all, 3)] }; }),
        { cls: "lt" }), true));

    var byField = rec.slice().sort(function (a, b) { return b.score_field - a.score_field; });
    body.appendChild(card("Where each bot's points come from (vs the 7 common opponents)",
      "Each game is scored against <b>par</b>, its opening's result under perfect play (0.500: the openings are balanced). <b>score = par + gifts − own losses</b>. Gifts are points the opponent threw away with result-changing moves; own losses are points the bot threw away. Pons never errs, so its whole margin is gifts. The split adds up exactly for every one of the games.",
      [decompChart(byField), table(["Bot", "Par", "Gifts", "Own losses", "Score", "Own error rate"],
        byField.map(function (r) { return ["<b>" + esc(r.name) + "</b>", num(r.par_field, 3), "+" + num(r.gifts_field, 3), "−" + num(r.own_losses_field, 3), num(r.score_field, 3), pct(r.own_error_rate, 2)]; }), { cls: "lt" })], true));

    var byErr = rec.slice().sort(function (a, b) { return b.opp_error_rate_field - a.opp_error_rate_field; });
    var col = function (r) { return r.bot === PONS ? C.grey : (r.bot.indexOf("score") >= 0 ? C.gold : C.blue); };
    body.appendChild(grid("g2", [
      card("Opponents make more mistakes against the nets", "Share of the opponent's moves that changed the game's theoretical result, counted only when the opponent still had something to lose.",
        hbarsCI(byErr.map(function (r) { return { label: r.name, value: r.opp_error_rate_field, color: col(r), bold: r.bot === PONS }; }), { min: 0, fmt: function (v) { return pct(v); }, tickFmt: function (v) { return pct(v, 0); }, labelW: 200 })),
      card("…and face more only-move positions", "Share of those turns where the opponent had exactly one move that kept its result.",
        hbarsCI(byErr.map(function (r) { return { label: r.name, value: r.opp_only_move_field, color: col(r), bold: r.bot === PONS }; }), { min: 0, fmt: function (v) { return pct(v); }, tickFmt: function (v) { return pct(v, 0); }, labelW: 200 }))]));

    var mt = RT("matched");
    body.appendChild(card("Same opponent, same opening: where the net and Pons first choose differently",
      "Every player is deterministic, so Pons vs X and a net vs X from the same opening and colours are the same game until Pons and the net first choose differently. That position is identical, so the difference in result traces back to that choice. Rows are grouped by the position's theoretical value at the split.",
      table(["Net", "Position", "Games", "Net better", "Same", "Net worse", "Mean score Δ", "Opp. error, Pons line", "Opp. error, net line", "Only-move, Pons line", "Only-move, net line", "Sign test p"],
        mt.map(function (r) { var all = r.state === "all"; return { cls: all ? "sel" : "", cells: [all ? "<b>" + esc(r.net) + "</b>" : esc(r.net), all ? "<b>all</b>" : esc(r.state), fmtInt(r.n), num(r.net_better), num(r.same), num(r.net_worse), isNum(r.mean_score_diff) ? sgn(r.mean_score_diff, 3) : "", pct(r.opp_err_after_pons), pct(r.opp_err_after_net), pct(r.only_move_after_pons), pct(r.only_move_after_net), isNum(r.sign_p) ? (r.sign_p < 0.001 ? "&lt; 0.001" : num(r.sign_p, 3)) : ""] }; }),
        { cls: "lt" }), true));

    body.appendChild(card("The very next move after the divergence",
      "Only the opponent's reply right after the split, in positions where Pons and the net were not winning. In the drawn rows some net moves were themselves mistakes, so the opponent's safe replies there include winning moves; the lost rows are the cleaner comparison.",
      table(["Net", "Position", "Games", "Opp. safe replies after Pons", "…after net", "Only-move after Pons", "…after net", "Opp. error next move after Pons", "…after net"],
        RT("next_turn").map(function (r) { return [esc(r.net), esc(r.state), fmtInt(r.n), num(r.mean_safe_pons, 2), num(r.mean_safe_net, 2), pct(r.only_move_pons), pct(r.only_move_net), pct(r.next_move_err_pons), pct(r.next_move_err_net)]; }),
        { cls: "lt" }), true));

    body.appendChild(card("Why", null, el("div", { "class": "answer" },
      '<div class="sup"><h4>What the data shows</h4><ul>' +
      "<li>In lost positions the ensemble + MCTS line did better than Pons's line in " + num(mLost.net_better) + " games and worse in " + num(mLost.net_worse) + ". Opponents erred on " + pct(mLost.opp_err_after_net) + " of moves on the net's line vs " + pct(mLost.opp_err_after_pons) + " on Pons's line.</li>" +
      "<li>On the very next move after the split (lost positions), the opponent had " + num(nt.mean_safe_net, 2) + " safe replies after the net's move vs " + num(nt.mean_safe_pons, 2) + " after Pons's, and erred " + pct(nt.next_move_err_net) + " vs " + pct(nt.next_move_err_pons) + ".</li>" +
      "<li>In won positions the net can only lose ground; it gave back " + num(mWon.net_worse) + " games.</li>" +
      "<li>The same ensemble without MCTS sets no traps: in lost positions it did better " + num(pLost.net_better) + " times and worse " + num(pLost.net_worse) + " times. sup and az + MCTS do set some traps but throw away more won and drawn games, so they finish behind Pons on matched games.</li></ul></div>" +
      '<div class="sug"><h4>Likely mechanism</h4><ul><li>Pons plays for the theoretical value only: the fastest win, the most central drawing move, the slowest loss. Nothing in that rule asks which move is hardest for the opponent to answer.</li>' +
      "<li>MCTS values a move by averaging over the opponent's replies it explored, so a move that leaves only one good answer scores better than one where every reply works. That makes it prefer trappy moves without being designed to.</li>" +
      "<li>Against Pons this earns nothing, because Pons never errs: the ensemble's few mistakes against Pons are its whole deficit.</li></ul></div>" +
      '<div class="need"><h4>Next test</h4><ul><li>Re-run the matched comparison with Pons choosing, among its equal-value moves, the one that leaves the opponent the fewest safe replies. If the mechanism is right, that Pons should out-score the nets against the field too.</li></ul></div>')));

    var pairNets = ["3-net score ensemble + MCTS200", "score 8x128 s0 + MCTS200", "3-net score ensemble (policy)", "sup 6x64 + MCTS200", "az 6x64 + MCTS200"];
    if (!UI.pairBot) UI.pairBot = pairNets[0];
    var pwrap = el("div");
    var pctrl = el("div", { "class": "ctrl" }, select("pairBot", pairNets, UI.pairBot, "Compare Pons with"));
    var pbox = el("div");
    function drawPairs() {
      var ps = RT("pairs"), opps = [];
      ps.forEach(function (r) { if (r.bot === UI.pairBot && opps.indexOf(r.opponent) < 0) opps.push(r.opponent); });
      var rows = [];
      opps.forEach(function (o) { ["Pons solver", UI.pairBot].forEach(function (b) { var r = ps.filter(function (x) { return x.bot === b && x.opponent === o; })[0]; if (r) rows.push({ cls: b === "Pons solver" ? "" : "sel", cells: [esc(o), esc(b), num(r.score, 3), "+" + num(r.gifts, 3), "−" + num(r.own_losses, 3), pct(r.opp_error_rate), pct(r.opp_only_move_share)] }); }); });
      pbox.innerHTML = ""; pbox.appendChild(table(["Opponent", "Bot", "Score", "Gifts", "Own losses", "Opp. error rate", "Opp. only-move share"], rows, { cls: "lt" }));
    }
    pctrl.querySelector("select").addEventListener("change", function () { UI.pairBot = this.value; drawPairs(); });
    drawPairs(); pwrap.appendChild(pctrl); pwrap.appendChild(pbox);
    body.appendChild(card("Per opponent: Pons vs one net", null, pwrap, true));

    body.appendChild(card("Example divergences (ensemble + MCTS vs Pons)", "Moves are 1-indexed columns, as the solver writes them. Position = the shared game up to the split.",
      table(["Opponent", "Position", "Value", "Pons played", "Net played", "Pons result", "Net result", "Net's game"],
        RT("examples").map(function (r) { return [esc(r.opponent), '<span class="sm">' + esc(r.position) + "</span>", esc(r.state), r.pons_move, r.net_move, num(r.pons_score, 1), num(r.net_score, 1), '<span class="sm">' + esc(r.net_game) + "</span>"]; }),
        { cls: "lt" }), true));
  }
  function decompChart(rows) {
    var W = 640, rowH = 26, m = { l: 210, r: 60, t: 22, b: 30 }, H = m.t + m.b + rowH * rows.length;
    var lo = Math.min(0.45, Math.min.apply(null, rows.map(function (r) { return r.score_field; })) - 0.03);
    var hi = Math.max.apply(null, rows.map(function (r) { return 0.5 + r.gifts_field; })) + 0.02;
    var X = function (v) { return m.l + (v - lo) / (hi - lo) * (W - m.l - m.r); };
    var s = svgHead(W, H, "Score split into par, gifts and own losses") + '<g class="grid">';
    niceTicks(lo, hi, 6).forEach(function (t) { s += '<line x1="' + X(t) + '" x2="' + X(t) + '" y1="' + m.t + '" y2="' + (H - m.b) + '"/><text x="' + X(t) + '" y="' + (H - m.b + 15) + '" text-anchor="middle">' + tickFmt(t) + "</text>"; });
    s += '</g><g class="ref"><line x1="' + X(0.5) + '" x2="' + X(0.5) + '" y1="' + (m.t - 8) + '" y2="' + (H - m.b) + '" stroke="' + C.white + '"/><text x="' + (X(0.5) + 3) + '" y="' + (m.t - 10) + '" style="fill:' + C.white + '">par</text></g>';
    rows.forEach(function (r, i) {
      var y = m.t + i * rowH + rowH / 2, g1 = 0.5 + r.gifts_field;
      s += '<text x="' + (m.l - 8) + '" y="' + (y + 4) + '" text-anchor="end" style="fill:' + (r.bot === PONS ? C.gold : "#e9ecf8") + '">' + esc(r.name) + "</text>";
      s += '<rect x="' + X(0.5) + '" y="' + (y - 8) + '" width="' + Math.max(1, X(g1) - X(0.5)) + '" height="16" rx="2" fill="' + C.blue + '" fill-opacity=".75" data-tip="' + esc(r.name + "\ngifts +" + num(r.gifts_field, 3)) + '"/>';
      if (r.own_losses_field > 0) s += '<rect x="' + X(g1 - r.own_losses_field) + '" y="' + (y - 4) + '" width="' + Math.max(1, X(g1) - X(g1 - r.own_losses_field)) + '" height="8" fill="' + C.orange + '" data-tip="' + esc(r.name + "\nown losses −" + num(r.own_losses_field, 3)) + '"/>';
      s += shapeSvg("circle", X(r.score_field), y, 4.6, C.white, r.name + "\nscore " + num(r.score_field, 3) + " = 0.500 + " + num(r.gifts_field, 3) + " − " + num(r.own_losses_field, 3), ' stroke="#0f1220" stroke-width="1"');
      s += '<text x="' + (Math.max(X(g1), X(r.score_field)) + 8) + '" y="' + (y + 4) + '" style="fill:#cfd5ee">' + num(r.score_field, 3) + "</text>";
    });
    var d = el("div"); d.innerHTML = s + "</svg>";
    d.appendChild(legend([{ name: "gifts (from par)", color: C.blue }, { name: "own losses (taken off the gifts)", color: C.orange }, { name: "final score", color: C.white, shape: "circle" }]));
    return d;
  }

  // ---------------------------------------------------------------- 7. inside all arena nets
  var CONCEPT_ORDER = ["heights", "landing_row", "ply", "first_to_move", "win_now", "opp_win_now", "own_threats", "opp_threats", "own_double", "opp_double", "own_odd_thr", "own_even_thr", "opp_odd_thr", "opp_even_thr", "opt_set", "net_move", "outcome", "score", "opp_reply", "opp_net_move", "phase"];
  function shortLayer(l) { return String(l).replace("block", "b"); }
  function renderInside9() {
    var body = $("inside9Body"); body.innerHTML = "";
    if (!D.report) { reportMissing(body); return; }
    var models = RT("models"), PS = RT("probe_summary"), KO = RT("knockouts");
    if (!models.length) { body.appendChild(empty("report.json has no interpretability tables yet (needs interp_arena.json).")); return; }
    var psv = function (mm, c) { return (PS.filter(function (r) { return r.key === mm + "|" + c; })[0] || {}); };
    var kov = function (mm, b) { return (KO.filter(function (r) { return r.key === mm + "|" + b; })[0] || {}).d_agree; };
    var sm = function (mm) { return rFind("models", "model", mm); };
    body.appendChild(el("p", { "class": "note" }, "All nine nets in the last arena: the 8 of ours (the three score nets are the ensemble's members) plus the open-source Bruneton AlphaZero net. For every layer a linear probe reads a concept from the activations (trained on 35,000 positions, scored on 10,000 held out). <b>Skill</b> = (score − chance) / (1 − chance). A probe shows the information is there, not that the net uses it; the block knock-outs are the causal check. The earlier 4-net study with the channel explorer and patching is on the <a href=\"#interp\">Inside the net</a> tab."));
    body.appendChild(card("What the nine nets share, and where they differ", null, el("div", { "class": "answer" }, '<div class="sup"><h4>Findings</h4><ul>' +
      "<li><b>Immediate tactics come first, in every net.</b> By block 1 (the stem in Bruneton's net) every net reads its own immediate wins and the opponent's must-block columns. Geometry and phase are readable from the raw input, so the nets keep that information rather than compute it.</li>" +
      "<li><b>Only nets trained on solver labels represent the opponent's best reply.</b> sup_6x64 reaches AUC " + num(psv("sup_6x64", "opp_reply").best, 2) + " (" + esc(psv("sup_6x64", "opp_reply").best_layer || "") + "), score_8x128_s0 " + num(psv("score_8x128_s0", "opp_reply").best, 2) + ". In az_6x64 the best layer is the raw input (" + num(psv("az_6x64", "opp_reply").best, 2) + "), and the same holds for ppo_6x64 and distill_az_2x32.</li>" +
      "<li><b>The score nets encode exact solver scores:</b> R² " + num(psv("score_8x128_s0", "score").best, 2) + " vs " + num(psv("sup_6x64", "score").best, 2) + " for sup_6x64 and " + num(psv("az_6x64", "score").best, 2) + " for az_6x64.</li>" +
      "<li><b>Block 1 matters most in every deep net:</b> skipping it changes sup_6x64's optimal-move rate by " + sgn(100 * kov("sup_6x64", "block1"), 1) + " points.</li>" +
      "<li><b>Same recipe, different insides.</b> The three score nets differ only in seed, yet skipping block 5 costs s1 " + sgn(100 * kov("score_8x128_s1", "block5"), 1) + " points vs " + sgn(100 * kov("score_8x128_s0", "block5"), 1) + " (s0) and " + sgn(100 * kov("score_8x128_s2", "block5"), 1) + " (s2). \"The block that does X\" is a fact about one net, not about the recipe.</li>" +
      "<li><b>Bruneton's net is one wide block with a one-filter value head.</b> Its raw policy is optimal on " + pct(sm("bruneton_az").intact_agree) + " of probe positions vs " + pct(sm("sup_6x64").intact_agree) + " for sup_6x64.</li></ul></div>")));

    if (!UI.i9 || !models.some(function (x) { return x.model === UI.i9; })) UI.i9 = models[0].model;
    var bar = el("div", { "class": "bar" }, select("i9model", models.map(function (x) { return { v: x.model, t: x.model + " (" + x.blocks + "x" + x.channels + ", " + params(x.params) + ")" }; }), UI.i9, "Net"));
    bar.querySelector("select").addEventListener("change", function () { UI.i9 = this.value; renderInside9(); });
    body.appendChild(bar);
    var M = sm(UI.i9);
    body.appendChild(el("div", { "class": "tiles" }, tile("Arena bot", '<span style="font-size:.9rem">' + esc(M.arena_bot || "–") + "</span>") + tile("Size", M.blocks + "x" + M.channels, params(M.params) + " params") +
      tile("Policy optimal (probe set)", pct(M.intact_agree)) + tile("Value accuracy (probe set)", pct(M.intact_value_acc)) +
      tile("Patching", M.patch_layer ? esc(M.patch_layer) : "–", isNum(M.patch_mean_abs_z) ? "mean |z| " + num(M.patch_mean_abs_z, 1) + " vs random directions" : "not run")));

    var pr = RT("probes").filter(function (r) { return r.model === UI.i9; });
    var layers = []; pr.slice().sort(function (a, b) { return a.layer_index - b.layer_index; }).forEach(function (r) { if (layers.indexOf(r.layer) < 0) layers.push(r.layer); });
    var concepts = CONCEPT_ORDER.filter(function (c) { return pr.some(function (r) { return r.concept === c; }); });
    var cell = {}; pr.forEach(function (r) { cell[r.concept + "|" + r.layer] = r; });
    var rows = concepts.map(function (c) { var r0 = pr.filter(function (r) { return r.concept === c; })[0]; return { label: c, group: GROUP_NAME[r0.group] || r0.group, c: c }; });
    var hm = heatmap({ rows: rows, cols: layers.map(shortLayer), cellW: 40, labelW: 130, top: 70, title: "probe skill per layer",
      val: function (i, j) { var r = cell[rows[i].c + "|" + layers[j]]; return r ? r.skill : null; },
      color: function (v) { return seqColor(v); }, text: function (v) { return num(v, 2); },
      tip: function (i, j, v) { var r = cell[rows[i].c + "|" + layers[j]]; return r ? r.concept_label + "\n" + r.layer + ": " + r.metric + " " + num(r.flat, 3) + "\nskill " + num(v, 2) + (isNum(r.shuffled) ? "\nshuffled-label control " + num(r.shuffled, 3) : "") : "not probed"; } });
    body.appendChild(card("Which layer knows what: " + UI.i9, "Rows grouped by concept type; columns run from the input through the stem and every block to the heads. Hover a cell for the raw probe score and the shuffled-label control.", [hm, scaleLegend("seq", "0", "1", "skill")], true));

    var kos = KO.filter(function (r) { return r.model === UI.i9; });
    var hasType = kos.length && isNum(kos[0].d_agree_own_win);
    var koSeries = [{ name: "all positions", color: C.blue, values: kos.map(function (r) { return r.d_agree; }) }];
    if (hasType) koSeries.push({ name: "own win available", color: C.green, values: kos.map(function (r) { return r.d_agree_own_win; }) },
      { name: "must block", color: C.orange, values: kos.map(function (r) { return r.d_agree_must_block; }) },
      { name: "quiet", color: C.purple, values: kos.map(function (r) { return r.d_agree_quiet; }) });
    var ch = RT("channels").filter(function (r) { return r.model === UI.i9; });
    var chLayers = [], chLabels = ["own-threat", "opponent-threat", "occupancy copy", "column detector", "row/height detector", "phase: early", "phase: late", "dead", "mixed"];
    ch.forEach(function (r) { if (chLayers.indexOf(r.layer) < 0) chLayers.push(r.layer); if (chLabels.indexOf(r.label) < 0) chLabels.push(r.label); });
    chLabels = chLabels.filter(function (l) { return ch.some(function (r) { return r.label === l; }); });
    var chCols = { "own-threat": C.green, "opponent-threat": C.orange, "occupancy copy": C.yellow, "column detector": C.sky, "row/height detector": C.blue, "phase: early": C.purple, "phase: late": C.red, dead: "#39406a", mixed: C.grey };
    body.appendChild(grid("g2", [
      card("Knock out one block at a time", "Each residual block is skipped in turn; the bar is the change in how often the net's move is solver-optimal. A causal test, not a probe.",
        groupBars({ categories: kos.map(function (r) { return "skip " + shortLayer(r.block); }), series: koSeries, yFmt: function (v) { return sgn(100 * v, 0); }, yLabel: "points of optimal moves" })),
      card("Channel census", ch.length ? "One label per channel per layer from its activation pattern (rules in research/interp.py)." : null,
        ch.length ? stackBars({ categories: chLayers.map(shortLayer), series: chLabels.map(function (l) { return { name: l, color: chCols[l] || C.white, values: chLayers.map(function (L) { var r = ch.filter(function (x) { return x.layer === L && x.label === l; })[0]; return r ? r.count : 0; }) }; }), yLabel: "channels" }) : empty("No channel census for this net (Bruneton was probed only)."))]));

    // across nets: first layer reaching 90% of best skill, as relative depth
    var showC = ["win_now", "opp_win_now", "own_threats", "opp_threats", "own_double", "opp_double", "opt_set", "net_move", "outcome", "score", "opp_reply"];
    var mids = models.map(function (x) { return x.model; });
    var depth = {};
    mids.forEach(function (mm) { var ls = RT("probes").filter(function (r) { return r.model === mm; }), mx = Math.max.apply(null, ls.map(function (r) { return r.layer_index; })); ls.forEach(function (r) { depth[mm + "|" + r.layer] = r.layer_index / (mx || 1); }); });
    var hm2 = heatmap({ rows: showC.map(function (c) { return { label: c }; }), cols: mids, cellW: 54, labelW: 110, top: 110, title: "first layer at 90% of best skill",
      val: function (i, j) { var r = psv(mids[j], showC[i]); return r.first_layer ? depth[mids[j] + "|" + r.first_layer] : null; },
      color: function (v) { return seqColor(isNum(v) ? 1 - v : null); }, text: function (v) { return pct(v, 0); },
      tip: function (i, j) { var r = psv(mids[j], showC[i]); return r.first_layer ? mids[j] + " · " + showC[i] + "\nfirst at " + r.first_layer + ", best at " + r.best_layer + " (" + num(r.best, 3) + ")" : "not probed"; } });
    body.appendChild(card("How early each net knows each concept", "Cell = the first layer that reaches 90% of that net's best probe skill, as a share of the net's depth (0% = input, 100% = heads). Brighter = earlier. A 0% cell means no layer beats the raw input, i.e. the net never builds that concept itself (see opp_reply in the self-play nets). Hover for the layer names and best score.", [hm2], true));
  }

  // ---------------------------------------------------------------- 8. every net benchmarked
  var FAM_ORDER = ["perfect labels + score head (Pons, 3M)", "perfect labels (Pons)", "net teacher (distillation)", "search teacher (minimax d4)", "heuristic teacher (no search)", "RL-net teacher", "self-play (AlphaZero)", "self-play RL (PPO / DQN)"];
  function renderAllNets() {
    var body = $("allnetsBody"); body.innerHTML = "";
    if (!D.report) { reportMissing(body); return; }
    var nets = RT("nets");
    body.appendChild(el("p", { "class": "note" }, "Every net in <code>runs/</code> (" + nets.length + ") on the same 3,000 held-out positions (<code>data/accuracy_positions.csv</code>, plies 8–30), so the numbers compare across nets. <b>Policy optimal</b>: the raw policy head's top move is solver-optimal. <b>Keeps result</b>: among positions not already lost, that move keeps the win or the draw. <b>Value acc.</b>: the W/D/L head's top class matches the solver; W / D / L split it by the true result. <b>Brier</b>: error of the W/D/L probabilities (lower is better). <b>Train-val</b> columns are the last line of each run's metrics.jsonl: its own split and its own labels (for sweep-B nets, the teacher's labels), so they do not compare across runs."));
    body.appendChild(card("Grouped by label source", null, table(["Label source", "Nets", "Best net", "Best policy optimal", "Mean policy optimal", "Best value acc."],
      RT("families").map(function (r) { return ["<b>" + esc(r.family) + "</b>", r.nets, esc(r.best_run), num(r.best_policy_optimal, 3), num(r.mean_policy_optimal, 3), num(r.best_value_acc, 3)]; }), { cls: "lt" }), true));
    var fams = FAM_ORDER.filter(function (f) { return nets.some(function (r) { return r.family === f; }); });
    body.appendChild(card("Policy vs value head, every net", "Each point is one net, coloured by the source of its training labels. Hover for the run.",
      xyChart({ series: fams.map(function (f, i) { return { name: f, color: SEQ[i % SEQ.length], shape: SHAPES[i % SHAPES.length], points: nets.filter(function (r) { return r.family === f; }).map(function (r) { return { x: r.policy_optimal, y: r.value_acc, tip: r.run + " (" + r.size + ")\n" + r.teacher_label + "\npolicy optimal " + pct(r.policy_optimal) + "\nvalue acc. " + pct(r.value_acc) }; }) }; }),
        xLabel: "policy optimal", yLabel: "value accuracy", xName: "policy", yName: "value", xFmt: function (v) { return pct(v, 0); }, yFmt: function (v) { return pct(v, 0); }, height: 330 }), true));

    if (!UI.an) UI.an = { fam: "", group: true, key: "policy_optimal", dir: -1 };
    var A = UI.an, wrap = el("div");
    var ctrl = el("div", { "class": "ctrl" }, select("anFam", [{ v: "", t: "All label sources" }].concat(fams.map(function (f) { return { v: f, t: f }; })), A.fam, "Label source") +
      '<label><input type="checkbox" id="anGroup"' + (A.group ? " checked" : "") + "> Group by label source</label>");
    var tbox = el("div");
    var cols = [["run", "Run", "s"], ["teacher_label", "Labels", "s"], ["params", "Size", "n"], ["policy_optimal", "Policy optimal", "n"], ["policy_keeps", "Keeps result", "n"], ["played_optimal", "As played", "n"],
      ["value_acc", "Value acc.", "n"], ["value_acc_W", "W", "n"], ["value_acc_D", "D", "n"], ["value_acc_L", "L", "n"], ["value_brier", "Brier", "n"], ["score_mae", "Score MAE", "n"],
      ["opt_8_15", "Opt. ply 8–15", "n"], ["opt_24_30", "Opt. ply 24–30", "n"], ["train_val_policy_acc", "Train-val policy", "n"], ["train_val_value_acc", "Train-val value", "n"]];
    function draw() {
      var rs = nets.filter(function (r) { return !A.fam || r.family === A.fam; }), text = cols.filter(function (c) { return c[0] === A.key; })[0][2] === "s";
      rs.sort(function (a, b) {
        if (A.group) { var fa = FAM_ORDER.indexOf(a.family), fb = FAM_ORDER.indexOf(b.family); if (fa !== fb) return fa - fb; }
        var x = a[A.key], y = b[A.key];
        if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1;
        return (text ? String(x).localeCompare(String(y)) : x - y) * A.dir;
      });
      var h = '<div class="scroll"><table class="t sortable lt"><thead><tr>' + cols.map(function (c) { return '<th scope="col"' + (A.key === c[0] ? ' aria-sort="' + (A.dir > 0 ? "ascending" : "descending") + '"' : "") + '><button type="button" data-k="' + c[0] + '">' + esc(c[1]) + "</button></th>"; }).join("") + "</tr></thead><tbody>";
      var last = null;
      rs.forEach(function (r) {
        if (A.group && r.family !== last) { h += '<tr class="sel"><td colspan="' + cols.length + '"><b>' + esc(r.family) + "</b></td></tr>"; last = r.family; }
        h += "<tr><td><b>" + esc(r.run) + '</b></td><td class="sm">' + esc(r.teacher_label) + "</td><td>" + esc(r.size) + ' <span class="muted sm">' + params(r.params) + "</span></td>" +
          ["policy_optimal", "policy_keeps", "played_optimal", "value_acc", "value_acc_W", "value_acc_D", "value_acc_L"].map(function (k) { return "<td>" + num(r[k], 3) + "</td>"; }).join("") +
          "<td>" + num(r.value_brier, 3) + "</td><td>" + (isNum(r.score_mae) ? num(r.score_mae, 2) : "") + "</td><td>" + num(r.opt_8_15, 3) + "</td><td>" + num(r.opt_24_30, 3) + "</td><td>" + (isNum(r.train_val_policy_acc) ? num(r.train_val_policy_acc, 3) : "") + "</td><td>" + (isNum(r.train_val_value_acc) ? num(r.train_val_value_acc, 3) : "") + "</td></tr>";
      });
      tbox.innerHTML = h + "</tbody></table></div>";
    }
    ctrl.addEventListener("change", function (e) { if (e.target.id === "anFam") A.fam = e.target.value; if (e.target.id === "anGroup") A.group = e.target.checked; draw(); });
    tbox.addEventListener("click", function (e) { var b = e.target.closest("button[data-k]"); if (!b) return; var k = b.getAttribute("data-k"); if (A.key === k) A.dir = -A.dir; else { A.key = k; A.dir = ["run", "teacher_label", "value_brier", "score_mae"].indexOf(k) >= 0 ? 1 : -1; } draw(); var nb = tbox.querySelector('button[data-k="' + k + '"]'); if (nb) nb.focus(); });
    draw(); wrap.appendChild(ctrl); wrap.appendChild(tbox);
    body.appendChild(card("All nets", "Click a column header to sort. \"As played\" is the move the net actually plays with its own settings (score head, mirror averaging). Score MAE: score-head error in Pons score units (score-head nets only).", wrap, true));
  }

  // =================================================================== tabs, refresh, start
  var RENDER = { overview: renderOverview, roster: renderRoster, best: renderBest, sweep: renderSweep, interp: renderInterp,
                 vspons: renderVsPons, inside9: renderInside9, allnets: renderAllNets };
  var DIRTY = {};
  function safe(name) {
    try { RENDER[name](); DIRTY[name] = false; }
    catch (e) { var b = $(name + "Body"); if (b) { b.innerHTML = ""; b.appendChild(empty("This section could not be drawn from the current data (" + esc(e.message) + "). The file may be mid-update; it will retry on the next refresh.")); } if (window.console) console.error(e); }
  }
  function markDirty(changed) {
    var dep = { overview: ["arena_analysis", "roster", "arena_top15", "sweep", "interp", "report"], roster: ["roster", "arena_analysis", "arena_top15"], best: ["arena_analysis"], sweep: ["sweep"], interp: ["interp"],
                vspons: ["report"], inside9: ["report"], allnets: ["report"] };
    Object.keys(dep).forEach(function (t) { if (dep[t].some(function (n) { return changed[n]; })) DIRTY[t] = true; });
  }
  function renderVisible() {
    // draw the visible tab now; others are drawn when opened (keeps the page light)
    var t = UI.tab;
    if (DIRTY.overview && t !== "overview") safe("overview");
    if (!DIRTY[t]) return;
    safe(t);
  }
  function tabs() {
    var btns = [].slice.call(document.querySelectorAll(".tabs button"));
    function show(name, focus) {
      UI.tab = name;
      btns.forEach(function (b) { var on = b.getAttribute("data-tab") === name; b.classList.toggle("on", on); b.setAttribute("aria-selected", on); b.tabIndex = on ? 0 : -1; if (on && focus) b.focus(); });
      document.querySelectorAll(".tab").forEach(function (t) { t.classList.toggle("on", t.id === "tab-" + name); });
      if (history.replaceState) history.replaceState(null, "", "#" + name);
      tipHide();
      renderVisible();
    }
    btns.forEach(function (b, i) {
      b.addEventListener("click", function () { show(b.getAttribute("data-tab")); });
      b.addEventListener("keydown", function (e) {
        var k = e.key, j = null;
        if (k === "ArrowRight") j = (i + 1) % btns.length; else if (k === "ArrowLeft") j = (i - 1 + btns.length) % btns.length; else if (k === "Home") j = 0; else if (k === "End") j = btns.length - 1;
        if (j != null) { e.preventDefault(); show(btns[j].getAttribute("data-tab"), true); }
      });
    });
    function fromHash() {
      // deep links, e.g. #interp&model=sup_6x64&layer=block3&ch=5&concept=opp_win_now
      var parts = location.hash.replace("#", "").split("&"), h = parts[0];
      parts.slice(1).forEach(function (p) {
        var kv = p.split("="), v = decodeURIComponent(kv[1] || "");
        if (kv[0] === "model") UI.imodel = v; else if (kv[0] === "layer") UI.ilayer = v; else if (kv[0] === "ch") UI.ich = +v; else if (kv[0] === "concept") UI.iconcept = v;
        if (kv[0] && DIRTY.interp === false) DIRTY.interp = true;
      });
      if (h && RENDER[h]) show(h); else show(UI.tab);
    }
    window.addEventListener("hashchange", fromHash);
    fromHash();
  }
  function needsPolling() { return sweepActive() || !D.sweep || !D.interp || !D.arena_top15 || !D.arena_analysis || !D.report; }
  function poll() {
    if (document.hidden || !needsPolling()) return;
    loadIndex().then(function (changed) { markDirty(changed); renderVisible(); }).catch(function () { });
  }
  Object.keys(RENDER).forEach(function (k) { DIRTY[k] = true; });
  Promise.all([loadIndex().catch(function (e) { $("overviewBody").innerHTML = ""; $("overviewBody").appendChild(empty("Could not reach the server: " + esc(e.message))); return {}; }), loadSelection()])
    .then(function () { tabs(); });
  setInterval(poll, 30000);
})();
