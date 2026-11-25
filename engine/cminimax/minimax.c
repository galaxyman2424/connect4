/* C port of engine/minimax.py (negamax + alpha-beta over bitboards, hand-crafted heuristic).
 *
 * Bit-for-bit the same algorithm as the Python version: same exact shortcuts, same
 * transposition-table rule (cutoff only at the same remaining depth, entries never evicted
 * within one search), same killer / history / TT-move ordering, same centre-first root
 * tie-break, same iterative deepening and early stop on a proven win/loss.  So it returns
 * the same column, the same score and the same node count (tests/test_cminimax.py checks
 * thousands of positions).  Only the time-limited mode can differ, because the clock does.
 *
 * Build: python -m engine.cbuild   (gcc / clang / MinGW / MSVC; output next to this file)
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#define EXPORT __declspec(dllexport)
#else
#include <time.h>
#define EXPORT __attribute__((visibility("default")))
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
static int popcnt64(uint64_t x) { return (int)__popcnt64(x); }
static int ctz64(uint64_t x) { unsigned long i; _BitScanForward64(&i, x); return (int)i; }
#else
static int popcnt64(uint64_t x) { return __builtin_popcountll(x); }
static int ctz64(uint64_t x) { return __builtin_ctzll(x); }
#endif

#define ROWS 6
#define COLS 7
#define H1 7
#define WIN 1000000LL
#define INF 1000000000LL
#define MATE_THRESHOLD (WIN - 100)
#define EXACT 0
#define LOWER 1
#define UPPER 2

static const int CENTER_ORDER[COLS] = {3, 2, 4, 1, 5, 0, 6};
static int CENTER_RANK[COLS];
static uint64_t BOTTOM_MASK, BOARD_MASK, COLUMN_MASK[COLS], CENTER_MASK;
static int inited = 0;

/* heuristic weights (engine/heuristic.py) */
#define W_CENTER 3
#define W_THREE 5
#define W_TWO 2
#define W_OPP_THREE 4
#define W_OPP_TWO 2

static void init(void) {
    if (inited) return;
    BOTTOM_MASK = 0;
    for (int c = 0; c < COLS; c++) BOTTOM_MASK |= (uint64_t)1 << (c * H1);
    BOARD_MASK = BOTTOM_MASK * (((uint64_t)1 << ROWS) - 1);
    for (int c = 0; c < COLS; c++) COLUMN_MASK[c] = (((uint64_t)1 << ROWS) - 1) << (c * H1);
    for (int i = 0; i < COLS; i++) CENTER_RANK[CENTER_ORDER[i]] = i;
    CENTER_MASK = COLUMN_MASK[3];
    inited = 1;
}

static uint64_t winning_cells_mask(uint64_t pos, uint64_t mask) {
    uint64_t r = (pos << 1) & (pos << 2) & (pos << 3);
    const int ps[3] = {H1, H1 - 1, H1 + 1};
    for (int k = 0; k < 3; k++) {
        int p = ps[k];
        uint64_t p2 = (pos << p) & (pos << (2 * p));
        r |= p2 & (pos << (3 * p));
        r |= p2 & (pos >> p);
        p2 = (pos >> p) & (pos >> (2 * p));
        r |= p2 & (pos << p);
        r |= p2 & (pos >> (3 * p));
    }
    return r & (BOARD_MASK ^ mask);
}

static void counts(uint64_t me, uint64_t freec, int *threes, int *twos) {
    const int dirs[4] = {1, H1, H1 - 1, H1 + 1};
    int t = 0, w = 0;
    for (int k = 0; k < 4; k++) {
        int d = dirs[k], d2 = d + d, d3 = d2 + d;
        uint64_t a = freec & (freec >> d) & (freec >> d2) & (freec >> d3);
        if (!a) continue;
        uint64_t m0 = me, m1 = me >> d, m2 = me >> d2, m3 = me >> d3;
        uint64_t s1 = m0 ^ m1, c1 = m0 & m1, s2 = m2 ^ m3, c2 = m2 & m3;
        uint64_t low = s1 ^ s2, mid = c1 ^ c2 ^ (s1 & s2);
        uint64_t x = a & mid;
        t += popcnt64(x & low);
        w += popcnt64(x & ~low);
    }
    *threes = t; *twos = w;
}

static int64_t evaluate_bits(uint64_t me, uint64_t opp) {
    int t, w, ot, ow;
    counts(me, BOARD_MASK & ~opp, &t, &w);
    counts(opp, BOARD_MASK & ~me, &ot, &ow);
    return (int64_t)W_CENTER * (popcnt64(me & CENTER_MASK) - popcnt64(opp & CENTER_MASK))
           + W_THREE * t + W_TWO * w - W_OPP_THREE * ot - W_OPP_TWO * ow;
}

/* ------------------------------------------------------------ TT (growable, no eviction) */
typedef struct { uint64_t key; int64_t val; int8_t depth, flag, move; } Entry;
typedef struct { Entry *t; size_t cap, n; } Table;

static int tt_init(Table *tt) {
    tt->cap = 1 << 16; tt->n = 0;
    tt->t = (Entry *)calloc(tt->cap, sizeof(Entry));
    return tt->t != NULL;
}
static size_t tt_slot(const Table *tt, uint64_t key) {
    uint64_t h = key * 0x9E3779B97F4A7C15ULL;
    size_t i = (size_t)(h >> 20) & (tt->cap - 1);
    while (tt->t[i].key && tt->t[i].key != key) i = (i + 1) & (tt->cap - 1);
    return i;
}
static Entry *tt_get(Table *tt, uint64_t key) {
    size_t i = tt_slot(tt, key);
    return tt->t[i].key ? &tt->t[i] : NULL;
}
static int tt_grow(Table *tt) {
    Table nt; nt.cap = tt->cap * 2; nt.n = 0;
    nt.t = (Entry *)calloc(nt.cap, sizeof(Entry));
    if (!nt.t) return 0;
    for (size_t i = 0; i < tt->cap; i++)
        if (tt->t[i].key) { nt.t[tt_slot(&nt, tt->t[i].key)] = tt->t[i]; nt.n++; }
    free(tt->t); *tt = nt;
    return 1;
}
static void tt_put(Table *tt, uint64_t key, int depth, int flag, int64_t val, int move) {
    if ((tt->n + 1) * 2 > tt->cap) tt_grow(tt);     /* on OOM keeps probing the old table */
    size_t i = tt_slot(tt, key);
    if (!tt->t[i].key) { tt->t[i].key = key; tt->n++; }
    tt->t[i].depth = (int8_t)depth; tt->t[i].flag = (int8_t)flag;
    tt->t[i].val = val; tt->t[i].move = (int8_t)move;
}

/* ------------------------------------------------------------------- clock */
static double now_sec(void) {
#ifdef _WIN32
    LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

/* ---------------------------------------------------------------- searcher */
typedef struct {
    int64_t nodes;
    int use_tt, ordering, abort_ok, aborted;
    double deadline;            /* < 0: none */
    Table tt;
    int killers[ROWS * COLS + 2][2];
    int64_t history[COLS * H1];
} S;

static int first_col(uint64_t bits) {
    for (int i = 0; i < COLS; i++) if (bits & COLUMN_MASK[CENTER_ORDER[i]]) return CENTER_ORDER[i];
    return -1;
}

static int64_t negamax(S *s, uint64_t cur, uint64_t mask, int depth, int64_t alpha, int64_t beta, int ply) {
    s->nodes++;
    if (s->deadline >= 0 && !(s->nodes & 1023) && s->abort_ok && now_sec() >= s->deadline) {
        s->aborted = 1;
        return 0;
    }
    if (mask == BOARD_MASK) return 0;
    uint64_t opp = cur ^ mask;
    if (depth == 0) return evaluate_bits(cur, opp);
    uint64_t possible = (mask + BOTTOM_MASK) & BOARD_MASK;
    if (winning_cells_mask(cur, mask) & possible) return WIN - ply - 1;
    if (depth >= 2) {
        uint64_t ow = winning_cells_mask(opp, mask);
        uint64_t forced = possible & ow;
        if (forced) {
            if (forced & (forced - 1)) return -(WIN - ply - 2);
            possible = forced;
        }
        possible &= ~(ow >> 1);
        if (!possible) return -(WIN - ply - 2);
    }
    int64_t alpha_orig = alpha, beta_orig = beta;
    int tt_move = -1;
    uint64_t key = cur + mask;
    if (s->use_tt) {
        Entry *e = tt_get(&s->tt, key);
        if (e) {
            if (e->depth == depth) {
                int64_t val = e->val;
                if (e->flag == EXACT) return val;
                if (e->flag == LOWER) { if (val > alpha) alpha = val; }
                else if (val < beta) beta = val;
                if (alpha >= beta) return val;
            }
            tt_move = e->move;
        }
    }
    int mcol[COLS], n = 0;
    uint64_t mmv[COLS];
    if (s->ordering) {
        int64_t key_[COLS];
        int *killers = s->killers[ply];
        for (int i = 0; i < COLS; i++) {
            int c = CENTER_ORDER[i];
            uint64_t mv = possible & COLUMN_MASK[c];
            if (!mv) continue;
            int64_t k;
            if (c == tt_move) k = (int64_t)1 << 40;
            else if (c == killers[0]) k = (int64_t)1 << 39;
            else if (c == killers[1]) k = (int64_t)1 << 38;
            else k = s->history[ctz64(mv)];
            /* insertion sort: key desc, then centre rank asc (= Python's sort of
               (k, -rank, c, mv) in reverse) */
            int j = n++;
            while (j > 0 && (key_[j - 1] < k || (key_[j - 1] == k && CENTER_RANK[mcol[j - 1]] > CENTER_RANK[c]))) {
                key_[j] = key_[j - 1]; mcol[j] = mcol[j - 1]; mmv[j] = mmv[j - 1]; j--;
            }
            key_[j] = k; mcol[j] = c; mmv[j] = mv;
        }
    } else {
        for (int c = 0; c < COLS; c++) {
            uint64_t mv = possible & COLUMN_MASK[c];
            if (mv) { mcol[n] = c; mmv[n] = mv; n++; }
        }
    }
    int64_t best = -INF;
    int best_col = mcol[0];
    for (int i = 0; i < n; i++) {
        int64_t v = -negamax(s, opp, mask | mmv[i], depth - 1, -beta, -alpha, ply + 1);
        if (s->aborted) return 0;
        if (v > best) {
            best = v; best_col = mcol[i];
            if (v > alpha) {
                alpha = v;
                if (alpha >= beta) {
                    if (s->ordering) {
                        int *ks = s->killers[ply];
                        if (ks[0] != mcol[i]) { ks[1] = ks[0]; ks[0] = mcol[i]; }
                        s->history[ctz64(mmv[i])] += (int64_t)depth * depth;
                    }
                    break;
                }
            }
        }
    }
    if (s->use_tt) {
        int flag = best <= alpha_orig ? UPPER : (best >= beta_orig ? LOWER : EXACT);
        tt_put(&s->tt, key, depth, flag, best, best_col);
    }
    return best;
}

static int root(S *s, uint64_t cur, uint64_t mask, int depth, int prev_best, int allow_abort, int64_t *score) {
    s->abort_ok = allow_abort;
    s->nodes++;
    uint64_t opp = cur ^ mask;
    uint64_t possible = (mask + BOTTOM_MASK) & BOARD_MASK;
    uint64_t wins = winning_cells_mask(cur, mask) & possible;
    if (wins) { *score = WIN - 1; return first_col(wins); }
    if (depth >= 2) {
        uint64_t ow = winning_cells_mask(opp, mask);
        uint64_t forced = possible & ow;
        if (forced) {
            if (forced & (forced - 1)) { *score = -(WIN - 2); return first_col(forced); }
            possible = forced;
        }
        uint64_t safe = possible & ~(ow >> 1);
        if (!safe) { *score = -(WIN - 2); return first_col(possible); }
        possible = safe;
    }
    int cols[COLS], n = 0;
    for (int i = 0; i < COLS; i++) {
        int c = s->ordering ? CENTER_ORDER[i] : i;
        if (possible & COLUMN_MASK[c]) cols[n++] = c;
    }
    if (s->ordering && prev_best >= 0) {
        for (int i = 0; i < n; i++) if (cols[i] == prev_best) {
            for (int j = i; j > 0; j--) cols[j] = cols[j - 1];
            cols[0] = prev_best;
            break;
        }
    }
    int64_t best = -INF;
    int best_col = cols[0];
    for (int i = 0; i < n; i++) {
        int c = cols[i];
        uint64_t mv = possible & COLUMN_MASK[c];
        uint64_t nm = mask | mv;
        int64_t v;
        if (nm == BOARD_MASK) v = 0;
        else if (i == 0) v = -negamax(s, opp, nm, depth - 1, -INF, INF, 1);
        else v = -negamax(s, opp, nm, depth - 1, -INF, -(best - 1), 1);
        if (s->aborted) return -1;
        if (v > best || (v == best && CENTER_RANK[c] < CENTER_RANK[best_col])) { best = v; best_col = c; }
    }
    *score = best;
    return best_col;
}

/* Returns the column (0-6), or -2 on bad input / out of memory.
   time_limit <= 0: none. */
EXPORT int c4_search(uint64_t cur, uint64_t mask, int moves_played, int depth, double time_limit,
                     int use_tt, int ordering, int64_t *score_out, int64_t *nodes_out,
                     int *depth_reached, int *timed_out) {
    init();
    if (depth < 1 || moves_played >= ROWS * COLS) return -2;
    double t0 = now_sec();
    S *s = (S *)calloc(1, sizeof(S));
    if (!s) return -2;
    s->use_tt = use_tt; s->ordering = ordering;
    s->deadline = time_limit > 0 ? t0 + time_limit : -1.0;
    for (int i = 0; i < ROWS * COLS + 2; i++) s->killers[i][0] = s->killers[i][1] = -1;
    if (use_tt && !tt_init(&s->tt)) { free(s); return -2; }
    int max_depth = depth < ROWS * COLS - moves_played ? depth : ROWS * COLS - moves_played;
    int best_col = -2, reached = 0;
    int64_t best_score = 0;
    *timed_out = 0;
    if (!use_tt && time_limit <= 0) {
        best_col = root(s, cur, mask, max_depth, -1, 0, &best_score);
        reached = max_depth;
    } else {
        int prev = -1;
        for (int d = 1; d <= max_depth; d++) {
            int64_t sc;
            int col = root(s, cur, mask, d, prev, d > 1, &sc);
            if (s->aborted) { *timed_out = 1; break; }
            best_col = col; best_score = sc; reached = d; prev = col;
            if (sc >= MATE_THRESHOLD || sc <= -MATE_THRESHOLD) break;
            if (s->deadline >= 0 && now_sec() >= s->deadline) break;
        }
    }
    *score_out = best_score;
    *nodes_out = s->nodes;
    *depth_reached = reached;
    if (s->tt.t) free(s->tt.t);
    free(s);
    return best_col;
}

/* Static evaluation only (for tests): same as engine.heuristic.evaluate_bits. */
EXPORT int64_t c4_evaluate(uint64_t me, uint64_t opp) { init(); return evaluate_bits(me, opp); }

/* Batch: n positions searched one after another (saves Python call overhead). */
EXPORT void c4_search_many(int n, const uint64_t *cur, const uint64_t *mask, const int *moves_played,
                           int depth, int use_tt, int ordering, int *cols, int64_t *scores, int64_t *nodes) {
    for (int i = 0; i < n; i++) {
        int dr, to;
        cols[i] = c4_search(cur[i], mask[i], moves_played[i], depth, 0.0, use_tt, ordering,
                            &scores[i], &nodes[i], &dr, &to);
    }
}
