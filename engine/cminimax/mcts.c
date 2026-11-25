/* C port of the PUCT tree in nn/mcts.py (select / expand / back up).
 *
 * The network stays in Python: each step the C side walks every tree down to its next
 * unevaluated leaf and hands the batch of leaf positions back; Python evaluates them in
 * one call and passes priors + values back in.  Same arithmetic, same order of operations
 * and same tie-breaks as nn/mcts.py, so visit counts are identical (tests/test_cmcts.py).
 *
 *   h = c4m_create(n, cur, mask, c_puct, fpu)
 *   k = c4m_collect(h, cur_out, mask_out)      // first call: the n roots; later: one simulation
 *   c4m_apply(h, priors[k*7], values[k], exact[k] or NULL)
 *   ... repeat collect/apply sims times ...
 *   c4m_root_stats(h, i, N[7], W[7], P[7], &n, &value); c4m_pv(...); c4m_free(h)
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#define COLS 7
#define ROWS 6
#define H1 7
#define MAXPATH 48

static const int CENTER_ORDER[COLS] = {3, 2, 4, 1, 5, 0, 6};
static const int CENTER_RANK[COLS] = {5, 3, 1, 0, 2, 4, 6};

static uint64_t board_mask(void) {
    uint64_t b = 0;
    for (int c = 0; c < COLS; c++) b |= (uint64_t)1 << (c * H1);
    return b * (((uint64_t)1 << ROWS) - 1);
}

static int alignment(uint64_t pos) {
    uint64_t m = pos & (pos >> H1);
    if (m & (m >> (2 * H1))) return 1;
    m = pos & (pos >> (H1 - 1));
    if (m & (m >> (2 * (H1 - 1)))) return 1;
    m = pos & (pos >> (H1 + 1));
    if (m & (m >> (2 * (H1 + 1)))) return 1;
    m = pos & (pos >> 1);
    if (m & (m >> 2)) return 1;
    return 0;
}

typedef struct {
    uint64_t cur, mask;
    double terminal;        /* valid if has_terminal */
    double value;
    double P[COLS];
    double W[COLS];
    int N[COLS];
    int32_t child[COLS];    /* -1 = not created */
    int n;
    int8_t has_terminal, expanded, nlegal;
    int8_t legal[COLS];
} Node;

typedef struct {
    Node *nodes;
    int count, cap;
    int nroots;
    int32_t *roots;
    double c_puct, fpu;
    int started;            /* 0: roots not expanded yet */
    uint64_t full;
    /* pending leaves of the current step */
    int npend;
    int32_t *pend_leaf;
    int32_t *pend_len;
    int32_t *pend_path;     /* nroots * MAXPATH * 2 (node, action) */
} Tree;

static int new_node(Tree *t, uint64_t cur, uint64_t mask, int has_term, double term) {
    if (t->count == t->cap) {
        int nc = t->cap * 2;
        Node *nn = (Node *)realloc(t->nodes, (size_t)nc * sizeof(Node));
        if (!nn) return -1;
        t->nodes = nn; t->cap = nc;
    }
    Node *x = &t->nodes[t->count];
    memset(x, 0, sizeof(Node));
    x->cur = cur; x->mask = mask;
    x->has_terminal = (int8_t)has_term; x->terminal = term;
    for (int a = 0; a < COLS; a++) x->child[a] = -1;
    return t->count++;
}

static int get_child(Tree *t, int idx, int a) {
    Node *x = &t->nodes[idx];
    if (x->child[a] >= 0) return x->child[a];
    uint64_t colmask = (((uint64_t)1 << ROWS) - 1) << (a * H1);
    uint64_t mv = (x->mask + ((uint64_t)1 << (a * H1))) & colmask;
    uint64_t me = x->cur | mv, nmask = x->mask | mv;
    int ht = 0; double tv = 0.0;
    if (alignment(me)) { ht = 1; tv = -1.0; }
    else if (nmask == t->full) { ht = 1; tv = 0.0; }
    int c = new_node(t, me ^ nmask, nmask, ht, tv);   /* may realloc: re-fetch x after */
    if (c >= 0) t->nodes[idx].child[a] = c;
    return c;
}

static void set_priors(Node *x, const double *pri, double value) {
    int nl = 0;
    double s = 0.0;
    for (int c = 0; c < COLS; c++) {
        if (!(x->mask & ((uint64_t)1 << (ROWS - 1 + c * H1)))) x->legal[nl++] = (int8_t)c;
    }
    x->nlegal = (int8_t)nl;
    for (int c = 0; c < COLS; c++) x->P[c] = 0.0;
    for (int i = 0; i < nl; i++) s += pri[x->legal[i]];
    if (s <= 0) { for (int i = 0; i < nl; i++) x->P[x->legal[i]] = 1.0 / nl; }
    else { for (int i = 0; i < nl; i++) x->P[x->legal[i]] = pri[x->legal[i]] / s; }
    x->value = value;
    x->expanded = 1;
}

static int select_a(const Node *x, double c_puct, double fpu) {
    double sq = c_puct * sqrt((double)(x->n > 0 ? x->n : 1));
    int best_a = -1;
    double best = -1e18;
    for (int i = 0; i < x->nlegal; i++) {
        int a = x->legal[i], n = x->N[a];
        double u = (n ? x->W[a] / n : fpu) + sq * x->P[a] / (1 + n);
        if (u > best) { best = u; best_a = a; }
    }
    return best_a;
}

static void backup(Tree *t, const int32_t *path, int len, double v) {
    double val = -v;
    for (int i = len - 1; i >= 0; i--) {
        Node *x = &t->nodes[path[2 * i]];
        int a = path[2 * i + 1];
        x->N[a] += 1; x->W[a] += val; x->n += 1;
        val = -val;
    }
}

EXPORT void *c4m_create(int n, const uint64_t *cur, const uint64_t *mask, double c_puct, double fpu) {
    Tree *t = (Tree *)calloc(1, sizeof(Tree));
    if (!t) return NULL;
    t->cap = n * 64 + 64;
    t->nodes = (Node *)malloc((size_t)t->cap * sizeof(Node));
    t->roots = (int32_t *)malloc(sizeof(int32_t) * (n ? n : 1));
    t->pend_leaf = (int32_t *)malloc(sizeof(int32_t) * (n ? n : 1));
    t->pend_len = (int32_t *)malloc(sizeof(int32_t) * (n ? n : 1));
    t->pend_path = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n ? n : 1) * MAXPATH * 2);
    if (!t->nodes || !t->roots || !t->pend_leaf || !t->pend_len || !t->pend_path) return NULL;
    t->nroots = n; t->c_puct = c_puct; t->fpu = fpu; t->full = board_mask();
    for (int i = 0; i < n; i++) t->roots[i] = new_node(t, cur[i], mask[i], 0, 0.0);
    return t;
}

EXPORT void c4m_free(void *h) {
    Tree *t = (Tree *)h;
    if (!t) return;
    free(t->nodes); free(t->roots); free(t->pend_leaf); free(t->pend_len); free(t->pend_path); free(t);
}

/* Fills the positions to evaluate; returns how many (or -1 on out of memory). */
EXPORT int c4m_collect(void *h, uint64_t *cur_out, uint64_t *mask_out) {
    Tree *t = (Tree *)h;
    t->npend = 0;
    if (!t->started) {
        for (int i = 0; i < t->nroots; i++) {
            Node *r = &t->nodes[t->roots[i]];
            if (!r->expanded && !r->has_terminal) {
                t->pend_leaf[t->npend] = t->roots[i]; t->pend_len[t->npend] = 0;
                cur_out[t->npend] = r->cur; mask_out[t->npend] = r->mask; t->npend++;
            }
        }
        return t->npend;
    }
    for (int i = 0; i < t->nroots; i++) {
        int node = t->roots[i];
        if (t->nodes[node].has_terminal) continue;
        int32_t *path = t->pend_path + (size_t)t->npend * MAXPATH * 2;
        int len = 0;
        for (;;) {
            int a = select_a(&t->nodes[node], t->c_puct, t->fpu);
            path[2 * len] = node; path[2 * len + 1] = a; len++;
            int ch = get_child(t, node, a);
            if (ch < 0) return -1;
            Node *c = &t->nodes[ch];
            if (c->has_terminal) { backup(t, path, len, c->terminal); break; }
            if (!c->expanded) {
                t->pend_leaf[t->npend] = ch; t->pend_len[t->npend] = len;
                cur_out[t->npend] = c->cur; mask_out[t->npend] = c->mask; t->npend++;
                break;
            }
            node = ch;
        }
    }
    return t->npend;
}

EXPORT void c4m_apply(void *h, const double *priors, const double *values, const uint8_t *exact) {
    Tree *t = (Tree *)h;
    for (int k = 0; k < t->npend; k++) {
        Node *x = &t->nodes[t->pend_leaf[k]];
        set_priors(x, priors + 7 * k, values[k]);
        if (!t->started) continue;
        if (exact && exact[k]) { x->has_terminal = 1; x->terminal = values[k]; }
        backup(t, t->pend_path + (size_t)k * MAXPATH * 2, t->pend_len[k], values[k]);
    }
    t->started = 1;
    t->npend = 0;
}

EXPORT void c4m_root_stats(void *h, int i, int *N, double *W, double *P, int *n, double *value) {
    Tree *t = (Tree *)h;
    Node *r = &t->nodes[t->roots[i]];
    for (int a = 0; a < COLS; a++) { N[a] = r->N[a]; W[a] = r->W[a]; P[a] = r->P[a]; }
    *n = r->n; *value = r->value;
}

/* Many roots at once: N (n*7) and the chosen column (most visited, centre-first tie-break). */
EXPORT void c4m_choose_all(void *h, int *cols, int *N) {
    Tree *t = (Tree *)h;
    for (int i = 0; i < t->nroots; i++) {
        Node *r = &t->nodes[t->roots[i]];
        int best = -1, bc = -1;
        for (int k = 0; k < r->nlegal; k++) if (r->N[r->legal[k]] > best) best = r->N[r->legal[k]];
        for (int k = 0; k < COLS; k++) {
            int c = CENTER_ORDER[k];
            int legal = 0;
            for (int j = 0; j < r->nlegal; j++) if (r->legal[j] == c) legal = 1;
            if (legal && r->N[c] == best) { bc = c; break; }
        }
        cols[i] = bc;
        if (N) for (int a = 0; a < COLS; a++) N[7 * i + a] = r->N[a];
    }
}

/* Principal variation (most-visited line, centre-first tie-break) and the deepest visited node. */
EXPORT int c4m_pv(void *h, int i, int *out, int max_len) {
    Tree *t = (Tree *)h;
    int node = t->roots[i], len = 0;
    while (node >= 0 && t->nodes[node].expanded && t->nodes[node].n > 0 && len < max_len) {
        Node *x = &t->nodes[node];
        int a = -1;
        for (int k = 0; k < x->nlegal; k++) {
            int c = x->legal[k];
            if (a < 0 || x->N[c] > x->N[a] || (x->N[c] == x->N[a] && CENTER_RANK[c] < CENTER_RANK[a])) a = c;
        }
        out[len++] = a;
        node = x->child[a];
    }
    return len;
}

EXPORT int c4m_max_depth(void *h, int i) {
    Tree *t = (Tree *)h;
    int best = 0;
    int cap = 1024, sp = 0;
    int *st = (int *)malloc(sizeof(int) * 2 * cap);
    if (!st) return -1;
    st[0] = t->roots[i]; st[1] = 0; sp = 1;
    while (sp) {
        sp--;
        int node = st[2 * sp], d = st[2 * sp + 1];
        if (d > best) best = d;
        Node *x = &t->nodes[node];
        if (!x->expanded) continue;
        for (int k = 0; k < x->nlegal; k++) {
            int a = x->legal[k];
            if (x->child[a] >= 0 && x->N[a] > 0) {
                if (sp == cap) { cap *= 2; int *ns = (int *)realloc(st, sizeof(int) * 2 * cap); if (!ns) { free(st); return best; } st = ns; }
                st[2 * sp] = x->child[a]; st[2 * sp + 1] = d + 1; sp++;
            }
        }
    }
    free(st);
    return best;
}

EXPORT int c4m_node_count(void *h) { return ((Tree *)h)->count; }
