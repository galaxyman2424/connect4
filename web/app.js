(function () {
  "use strict";
  var ROWS = 6, COLS = 7;
  // Level table (statement section 4.2): [depth label, random %]
  var LEVELS = {
    1: ["1", 50], 2: ["1", 20], 3: ["2", 10], 4: ["3", 0], 5: ["4", 0],
    6: ["5", 0], 7: ["6", 0], 8: ["7", 0], 9: ["8", 0], 10: ["9+ (about 2 s limit)", 0]
  };
  // Encoding: +1 = whoever moves first, -1 = second. Rendered p1 = red, p2 = yellow.
  var grid, human, bot, turn, over, busy, gameId = 0, humanFirst = true;
  var useNN = false;   // opponent: false = minimax (slider level), true = trained network (Lab)

  var $ = function (id) { return document.getElementById(id); };
  var boardEl = $("board"), statusEl = $("status"), reasonEl = $("reason"),
      ghostRow = $("ghostRow"), colBtns = $("colButtons"), levelEl = $("level");
  var cells = [], ghosts = [], buttons = [];

  function cls(p) { return p === 1 ? "p1" : "p2"; }

  function build() {
    var r, c, el;
    for (r = 0; r < ROWS; r++) {
      cells.push([]);
      for (c = 0; c < COLS; c++) {
        el = document.createElement("div");
        el.className = "cell";
        boardEl.appendChild(el);
        cells[r].push(el);
      }
    }
    for (c = 0; c < COLS; c++) {
      el = document.createElement("div");
      el.className = "ghost";
      ghostRow.appendChild(el);
      ghosts.push(el);
      var b = document.createElement("button");
      b.type = "button";
      b.setAttribute("aria-label", "Column " + (c + 1));
      (function (col) {
        b.addEventListener("click", function () { humanMove(col); });
        b.addEventListener("mouseenter", function () { hover(col); });
        b.addEventListener("mouseleave", function () { hover(-1); });
        b.addEventListener("focus", function () { hover(col); });
        b.addEventListener("blur", function () { hover(-1); });
      })(c);
      colBtns.appendChild(b);
      buttons.push(b);
    }
  }

  function emptyGrid() {
    var g = [];
    for (var r = 0; r < ROWS; r++) { g.push([0, 0, 0, 0, 0, 0, 0]); }
    return g;
  }
  function colFull(c) { return grid[0][c] !== 0; }
  function dropRow(c) {
    for (var r = ROWS - 1; r >= 0; r--) if (grid[r][c] === 0) return r;
    return -1;
  }
  function findWin(g) {
    var dirs = [[0, 1], [1, 0], [1, 1], [1, -1]];
    for (var r = 0; r < ROWS; r++) for (var c = 0; c < COLS; c++) {
      var p = g[r][c];
      if (!p) continue;
      for (var d = 0; d < 4; d++) {
        var cellsW = [[r, c]], k;
        for (k = 1; k < 4; k++) {
          var rr = r + dirs[d][0] * k, cc = c + dirs[d][1] * k;
          if (rr < 0 || rr >= ROWS || cc < 0 || cc >= COLS || g[rr][cc] !== p) break;
          cellsW.push([rr, cc]);
        }
        if (cellsW.length === 4) return { winner: p, cells: cellsW };
      }
    }
    return null;
  }
  function isFull() { for (var c = 0; c < COLS; c++) if (!colFull(c)) return false; return true; }

  // ------------------------------------------------------------ evaluation bar
  // Rated from RED's side (+1 = first player). History holds one entry per position,
  // index = number of moves played, so the graph and the per-move change line up.
  var moves = "", evalHist = [];
  function evalSource() { return $("evalSrc").value || ""; }
  function showBar(e) {
    var bar = $("evalBar"), red = $("evalRed"), lab = $("evalLabel");
    var frac = Math.max(0, Math.min(1, e.bar));
    var humanRed = human === 1;
    bar.classList.toggle("flip", !humanRed);              // human's colour at the bottom, as in chess
    red.style.height = (100 * frac).toFixed(1) + "%";
    lab.textContent = e.label;
    // label sits at the end of the side that is ahead (like lichess / chess.com)
    var redAhead = frac >= 0.5;
    lab.classList.toggle("top", redAhead !== humanRed);
    lab.style.color = redAhead ? "#fff" : "#0b0e1c";
    bar.setAttribute("aria-valuenow", Math.round(100 * frac));
    bar.setAttribute("aria-valuetext", e.text || e.label);
    bar.title = e.text || "";
    bar.classList.remove("stale");
  }
  function grade(prev, cur, col, mover) {
    // returns [css class, word, detail] for the move that led from prev to cur
    if (prev.columns && prev.columns[col] != null) {        // solver: exact
      var sc = prev.columns, best = -1e9, k;
      for (k = 0; k < sc.length; k++) if (sc[k] != null && sc[k] > best) best = sc[k];
      var got = sc[col], sgn = function (x) { return x > 0 ? 1 : x < 0 ? -1 : 0; };
      if (got === best) return ["best", "Best", "a perfect-play move"];
      if (sgn(got) < sgn(best)) return ["blunder", "Blunder", best > 0 ? (got === 0 ? "threw away a forced win (now a draw)" : "threw away a forced win (now lost)") : "turned a draw into a loss"];
      return ["inacc", "Inaccuracy", best > 0 ? "still winning, but slower than the best move" : "still lost, but faster than necessary"];
    }
    var d = (cur.value - prev.value) * mover;               // change for the side that moved
    if (d >= -0.03) return ["best", "Good", "kept or improved the evaluation"];
    if (d >= -0.10) return ["good", "OK", "small drop"];
    if (d >= -0.25) return ["inacc", "Inaccuracy", "evaluation dropped"];
    if (d >= -0.5) return ["mistake", "Mistake", "evaluation dropped a lot"];
    return ["blunder", "Blunder", "evaluation collapsed"];
  }
  function showMove() {
    var n = moves.length, cur = evalHist[n], prev = evalHist[n - 1], box = $("evalMove");
    if (!n || !cur || !prev) { box.innerHTML = "&nbsp;"; return; }
    var mover = n % 2 === 1 ? 1 : -1, col = parseInt(moves.charAt(n - 1), 10) - 1;
    var who = mover === human ? "Your" : "Bot's";
    var g = grade(prev, cur, col, mover);
    var dv = cur.value - prev.value, dtxt = cur.columns ? (prev.label + " → " + cur.label) :
      ((dv >= 0 ? "+" : "") + dv.toFixed(2) + " for red (" + prev.label + " → " + cur.label + ")");
    box.innerHTML = '<span class="tag ' + g[0] + '">' + g[1] + "</span>" + who + " move " + n + " (column " + (col + 1) + "): " +
      g[2] + ". <span style=\"color:var(--muted)\">" + dtxt + "</span>";
  }
  function drawGraph() {
    var svg = $("evalGraph"), W = 420, H = 80, n = Math.max(moves.length, 8), pts = [], i;
    for (i = 0; i < evalHist.length; i++) if (evalHist[i]) pts.push([i * W / n, H * (1 - evalHist[i].bar)]);
    var h = '<line class="mid" x1="0" y1="' + H / 2 + '" x2="' + W + '" y2="' + H / 2 + '"/>';
    if (pts.length) {
      var line = pts.map(function (p) { return p[0].toFixed(1) + "," + p[1].toFixed(1); }).join(" ");
      var last = pts[pts.length - 1], first = pts[0];
      h += '<clipPath id="cR"><rect x="0" y="0" width="' + W + '" height="' + H / 2 + '"/></clipPath>' +
           '<clipPath id="cY"><rect x="0" y="' + H / 2 + '" width="' + W + '" height="' + H / 2 + '"/></clipPath>';
      var area = '<polygon points="' + first[0].toFixed(1) + "," + H / 2 + " " + line + " " + last[0].toFixed(1) + "," + H / 2 + '"';
      h += area + ' class="area-r" clip-path="url(#cR)"/>' + area + ' class="area-y" clip-path="url(#cY)"/>';
      h += '<polyline class="line" points="' + line + '"/><circle r="3" cx="' + last[0].toFixed(1) + '" cy="' + last[1].toFixed(1) + '"/>';
    }
    svg.innerHTML = h;
    svg.setAttribute("aria-label", "Evaluation after each move, red above the line: " +
      evalHist.map(function (e, k) { return e ? k + ": " + e.label : null; }).filter(Boolean).join(", "));
  }
  function evaluate() {
    var id = gameId, src = evalSource();
    if (!src) return;
    $("evalBar").classList.add("stale");
    fetch("/api/eval", { method: "POST", headers: { "Content-Type": "application/json" },
                         body: JSON.stringify({ moves: moves, source: src }) })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })
      .then(function (o) {
        if (id !== gameId || !o.ok) { if (!o.ok && id === gameId) $("evalText").textContent = "Evaluation unavailable: " + (o.j.error || "server error"); return; }
        evalHist[o.j.ply] = o.j;
        if (o.j.ply !== moves.length) { drawGraph(); return; }   // an older position answered late
        showBar(o.j);
        $("evalText").textContent = (o.j.note ? o.j.note + ". " : "") + o.j.text;
        showMove(); drawGraph();
      }).catch(function () { });
  }
  function resetEval() {
    moves = ""; evalHist = [];
    showBar({ bar: 0.5, label: "…", text: "" });
    $("evalText").innerHTML = "&nbsp;"; $("evalMove").innerHTML = "&nbsp;"; drawGraph();
    evaluate();
  }
  function reevaluateAll() {
    // new evaluator: rate every position of the game so far, then the current one
    var keep = moves, id = gameId, chain = Promise.resolve();
    evalHist = [];
    for (var k = 0; k <= keep.length; k++) (function (k) {
      chain = chain.then(function () {
        if (id !== gameId || moves !== keep) return;
        return fetch("/api/eval", { method: "POST", headers: { "Content-Type": "application/json" },
                                    body: JSON.stringify({ moves: keep.slice(0, k), source: evalSource() }) })
          .then(function (r) { return r.ok ? r.json() : null; }).then(function (j) { if (j) evalHist[k] = j; });
      });
    })(k);
    chain.then(function () {
      if (id !== gameId || moves !== keep) return;
      var cur = evalHist[keep.length];
      if (cur) { showBar(cur); $("evalText").textContent = (cur.note ? cur.note + ". " : "") + cur.text; }
      showMove(); drawGraph();
    });
  }
  function loadEvalSources() {
    return fetch("/api/eval/sources").then(function (r) { return r.ok ? r.json() : null; }).then(function (s) {
      var sel = $("evalSrc");
      if (!s || (!s.solver && !s.nets.length)) { $("evalBar").hidden = true; sel.closest(".evalpanel").hidden = true; return; }
      var saved = null;
      try { saved = localStorage.getItem("c4.evalSrc"); } catch (e) { }
      s.nets.forEach(function (n) { var o = document.createElement("option"); o.value = "nn:" + n; o.textContent = "Net: " + n; sel.appendChild(o); });
      if (s.solver) { var o = document.createElement("option"); o.value = "solver"; o.textContent = "Perfect play (Pons solver)"; sel.appendChild(o); }
      sel.value = saved && [].some.call(sel.options, function (o) { return o.value === saved; }) ? saved : s.default;
    }).catch(function () { $("evalBar").hidden = true; });
  }

  function place(col, player, animate) {
    var r = dropRow(col);
    grid[r][col] = player;
    moves += String(col + 1);
    evaluate();
    var d = document.createElement("div");
    d.className = "disc " + cls(player);
    if (animate) {
      var ms = 260 + (r + 1) * 60; // taller fall = a bit longer
      d.style.setProperty("--rows", r + 1);
      d.style.setProperty("--ms", ms + "ms");
      d.className += " drop";
      cells[r][col].appendChild(d);
      return ms + 250;
    }
    cells[r][col].appendChild(d);
    return 0;
  }

  function setStatus(text, kind) {
    statusEl.textContent = text;
    statusEl.className = "status" + (kind ? " " + kind : "");
  }
  function refresh() {
    for (var c = 0; c < COLS; c++) buttons[c].disabled = over || busy || colFull(c);
  }
  var hovered = -1;
  function hover(col) {
    hovered = col;
    var ok = col >= 0 && !over && !busy && !colFull(col);
    for (var c = 0; c < COLS; c++) {
      ghosts[c].className = "ghost " + cls(human) + (ok && c === col ? " show" : "");
      for (var r = 0; r < ROWS; r++) cells[r][c].classList.toggle("hov", ok && c === col);
    }
  }

  function endGame(win) {
    over = true;
    if (win) {
      win.cells.forEach(function (rc) {
        var d = cells[rc[0]][rc[1]].querySelector(".disc");
        if (d) d.classList.add("win");
      });
      if (win.winner === human) setStatus("You win!", "win");
      else setStatus("Bot wins.", "lose");
    } else {
      setStatus("Draw: the board is full.");
    }
    busy = false; refresh(); hover(-1);
  }

  function humanMove(col) {
    if (over || busy || col < 0 || col >= COLS || colFull(col) || turn !== human) return;
    busy = true; refresh(); hover(-1);
    var wait = place(col, human, true);
    var id = gameId;
    var win = findWin(grid);
    setTimeout(function () {
      if (id !== gameId) return;
      if (win) return endGame(win);
      if (isFull()) return endGame(null);
      turn = bot;
      botMove();
    }, wait);
  }

  function fmt(n) { return Math.round(n).toLocaleString("en-US"); }
  function evalStr(s) {
    if (Math.abs(s) >= 100000) return s > 0 ? "forced win" : "forced loss";
    return "eval " + (s > 0 ? "+" : "") + Math.round(s);
  }

  function botMove() {
    var id = gameId;
    busy = true; refresh();
    setStatus("Bot is thinking...", "think");
    fetch("/api/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(useNN ? { board: grid, engine: "nn", run: $("nnRun").value, sims: parseInt($("nnSims").value, 10) }
                                 : { board: grid, level: parseInt(levelEl.value, 10) })
    }).then(function (res) {
      return res.json().then(function (j) { return { ok: res.ok, j: j }; });
    }).then(function (o) {
      if (id !== gameId) return;
      if (!o.ok) throw new Error(o.j.error || "server error");
      var j = o.j;
      if (j.engine === "nn") {
        var winPct = Math.round(50 * (j.score + 1));
        reasonEl.textContent = "Neural net (" + j.nn.run + "): " + (j.nn.sims ? fmt(j.nn.sims) + " simulations" : "policy only") +
          " in " + fmt(j.ms) + " ms, expects to score about " + winPct + "%";
      } else if (j.random) reasonEl.textContent = "Bot played a random move";
      else reasonEl.textContent = "Bot searched " + fmt(j.nodes) + (j.nodes === 1 ? " position in " : " positions in ") + fmt(j.ms) +
        " ms (depth " + j.depth + ", " + evalStr(j.score) + ")";
      var wait = place(j.col, bot, true);
      setTimeout(function () {
        if (id !== gameId) return;
        var win = findWin(grid);
        if (win) return endGame(win);
        if (isFull()) return endGame(null);
        turn = human; busy = false; refresh();
        setStatus("Your move (you are " + (human === 1 ? "red" : "yellow") + ")");
        if (hovered >= 0) hover(hovered);
      }, wait);
    }).catch(function (e) {
      if (id !== gameId) return;
      busy = false; over = true; refresh();
      setStatus("Error: " + e.message + " (start a new game)", "lose");
    });
  }

  function newGame() {
    gameId++;
    grid = emptyGrid();
    cells.forEach(function (row) { row.forEach(function (el) { el.innerHTML = ""; }); });
    human = humanFirst ? 1 : -1;
    bot = -human;
    turn = 1; over = false; busy = false;
    reasonEl.innerHTML = "&nbsp;";
    resetEval();
    refresh(); hover(-1);
    var yc = cls(human);
    if (humanFirst) setStatus("Your move (you are " + (yc === "p1" ? "red" : "yellow") + ")");
    else { turn = bot; botMove(); }
  }

  function setFirst(h) {
    humanFirst = h;
    $("firstHuman").classList.toggle("on", h); $("firstHuman").setAttribute("aria-checked", h);
    $("firstBot").classList.toggle("on", !h); $("firstBot").setAttribute("aria-checked", !h);
    newGame();
  }
  function updateLevel() {
    var L = LEVELS[levelEl.value];
    $("levelNum").textContent = levelEl.value;
    $("levelLabel").textContent = "Search depth " + L[0] + ", " + L[1] + "% random moves";
  }

  function setOpponent(nn) {
    useNN = nn;
    $("oppMinimax").classList.toggle("on", !nn); $("oppMinimax").setAttribute("aria-checked", !nn);
    $("oppNN").classList.toggle("on", nn); $("oppNN").setAttribute("aria-checked", nn);
    $("nnRun").hidden = !nn; $("nnSims").hidden = !nn;
    $("levelCtl").hidden = nn;
    newGame();
  }
  function loadModels() {
    fetch("/api/nn/models").then(function (r) { return r.ok ? r.json() : []; }).then(function (list) {
      if (!list.length) return;
      var sel = $("nnRun");
      list.forEach(function (m) {
        var o = document.createElement("option");
        o.value = m.run;
        o.textContent = m.run + (m.elo ? " (Elo " + Math.round(m.elo) + ")" : "");
        sel.appendChild(o);
      });
      $("oppCtl").hidden = false;
    }).catch(function () {});
  }

  build();
  $("oppMinimax").addEventListener("click", function () { setOpponent(false); });
  $("oppNN").addEventListener("click", function () { setOpponent(true); });
  $("nnRun").addEventListener("change", newGame);
  loadModels();
  levelEl.addEventListener("input", updateLevel);
  levelEl.addEventListener("change", function () { levelEl.blur(); });
  $("newGame").addEventListener("click", newGame);
  $("firstHuman").addEventListener("click", function () { setFirst(true); });
  $("firstBot").addEventListener("click", function () { setFirst(false); });
  document.addEventListener("keydown", function (e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "SELECT")) return;
    if (e.key >= "1" && e.key <= "7") humanMove(parseInt(e.key, 10) - 1);
  });
  $("evalSrc").addEventListener("change", function () {
    try { localStorage.setItem("c4.evalSrc", this.value); } catch (e) { }
    reevaluateAll();
  });
  updateLevel();
  newGame();
  loadEvalSources().then(reevaluateAll);
})();
