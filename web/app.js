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

  function place(col, player, animate) {
    var r = dropRow(col);
    grid[r][col] = player;
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
    if (e.target && e.target.tagName === "INPUT") return;
    if (e.key >= "1" && e.key <= "7") humanMove(parseInt(e.key, 10) - 1);
  });
  updateLevel();
  newGame();
})();
