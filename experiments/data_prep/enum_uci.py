"""Enumerate all distinct legal 8-ply Connect-4 positions (as boards) with no winner yet.
Reports counts under several definitions of 'forced next move'."""
import itertools, sys
W, H = 7, 6

def wins(cols, p, c):
    # cols: tuple of tuples (0/1 per stone bottom-up); last stone of player p placed in column c
    r = len(cols[c]) - 1
    def at(cc, rr):
        return 0 <= cc < W and 0 <= rr < len(cols[cc]) and cols[cc][rr] == p
    for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
        n = 1
        for s in (1, -1):
            cc, rr = c + s * dc, r + s * dr
            while at(cc, rr):
                n += 1; cc += s * dc; rr += s * dr
        if n >= 4:
            return True
    return False

def play(cols, c, p):
    l = list(cols); l[c] = cols[c] + (p,); return tuple(l)

def can_win(cols, p):
    for c in range(W):
        if len(cols[c]) < H and wins(play(cols, c, p), p, c):
            return True
    return False

def wins_cols(cols, p):
    return [c for c in range(W) if len(cols[c]) < H and wins(play(cols, c, p), p, c)]

def enumerate_positions(plies=8):
    level = {tuple(() for _ in range(W)): None}  # position -> a move sequence
    start = tuple(() for _ in range(W))
    level = {start: ""}
    for ply in range(plies):
        p = ply % 2
        nxt = {}
        for pos, seq in level.items():
            for c in range(W):
                if len(pos[c]) >= H: continue
                n = play(pos, c, p)
                if wins(n, p, c): continue
                if n not in nxt: nxt[n] = seq + str(c + 1)
        level = nxt
    return level

if __name__ == "__main__":
    lv = enumerate_positions(8)
    print("non-won 8-ply positions:", len(lv))
    # x (player 0) to move
    a = sum(1 for p in lv if not can_win(p, 0))           # x has no immediate win
    b = sum(1 for p in lv if not can_win(p, 0) and not can_win(p, 1))  # nobody threatens
    c = sum(1 for p in lv if not can_win(p, 1))
    d = sum(1 for p in lv if not can_win(p, 0) and len(wins_cols(p, 1)) == 0)
    print("x cannot win now:", a, "| neither can win now:", b, "| o cannot win now:", c)
