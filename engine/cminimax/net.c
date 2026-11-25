/* Native network forward pass + fully native MCTS (tree + network, no Python per step).
 *
 * Runs an exported model.npz (BatchNorm already folded into the convs, nn/export.py) with the
 * same maths as nn/numpy_net.py: optional PCA input projection, residual trunk, policy / WDL /
 * score heads, mirror averaging, the play head (policy | score | combo) and the value source
 * (wdl | score).  Float32 like numpy, so outputs agree to ~1e-5 and moves are the same
 * (tests/test_native.py).  Weights arrive from Python as one float32 buffer (nn/native.py).
 *
 * The search reuses the C tree of mcts.c (same file set, same library) and evaluates the
 * leaves here, with an optional position cache that lives as long as the caller wants
 * (one game for an MCTSAgent: consecutive moves re-use most leaf evaluations).
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define C4_AVX2 1
#endif

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#define ROWS 6
#define COLS 7
#define H1 7
#define CELLS 42
#define SCORE_SCALE 10.0f
#define OB 8   /* output channels per register block */
#define NB 8   /* columns per register block (one AVX register) */

/* tree API from mcts.c */
void *c4m_create(int n, const uint64_t *cur, const uint64_t *mask, double c_puct, double fpu);
void c4m_free(void *h);
int c4m_collect(void *h, uint64_t *cur_out, uint64_t *mask_out);
void c4m_apply(void *h, const double *priors, const double *values, const uint8_t *exact);
void c4m_root_stats(void *h, int i, int *N, double *W, double *P, int *n, double *value);
int c4m_pv(void *h, int i, int *out, int max_len);
int c4m_max_depth(void *h, int i);

typedef struct {
    int blocks, C, pc, vc, vh, has_score, sc, sh, k;
    int play_head;            /* 0 policy, 1 score, 2 combo */
    float play_temp, play_w;
    int mirror, value_score;
    const float *pmean, *pU;
    const float *stem_w, *stem_b;
    const float **bw1, **bb1, **bw2, **bb2;
    float *pk_stem, **pk1, **pk2;     /* conv weights repacked [o/8][k][8] for the kernel */
    const float *pol_w, *pol_b, *polfc_w, *polfc_b;
    const float *val_w, *val_b, *v1_w, *v1_b, *v2_w, *v2_b;
    const float *sc_w, *sc_b, *s1_w, *s1_b, *s2_w, *s2_b;
    float *buf;               /* owned copy of all weights */
    /* scratch, grown on demand (for nb positions at once) */
    int cap_b;
    float *x, *h, *y, *y2, *col, *head, *hid;
} Net;

/* W (co x K) -> [ceil(co/8)][K][8], zero-padded output channels */
static float *pack_w(const float *w, int co, int K) {
    int nob = (co + 7) / 8;
    float *pk = (float *)calloc((size_t)nob * K * 8, sizeof(float));
    if (!pk) return NULL;
    for (int o = 0; o < co; o++)
        for (int k = 0; k < K; k++) pk[((size_t)(o / 8) * K + k) * 8 + (o % 8)] = w[(size_t)o * K + k];
    return pk;
}

static const float *take(const float **p, size_t n) { const float *r = *p; *p += n; return r; }

/* Weight order: see nn/native.py _pack() */
EXPORT void *c4n_create(const int *cfg, const float *play, const float *weights, long long n_weights) {
    Net *t = (Net *)calloc(1, sizeof(Net));
    if (!t) return NULL;
    t->blocks = cfg[0]; t->C = cfg[1]; t->pc = cfg[2]; t->vc = cfg[3]; t->vh = cfg[4];
    t->has_score = cfg[5]; t->sc = cfg[6]; t->sh = cfg[7]; t->k = cfg[8];
    t->play_head = cfg[9]; t->mirror = cfg[10]; t->value_score = cfg[11];
    t->play_temp = play[0]; t->play_w = play[1];
    t->buf = (float *)malloc(sizeof(float) * (size_t)n_weights);
    t->bw1 = (const float **)calloc(t->blocks ? t->blocks : 1, sizeof(float *));
    t->bb1 = (const float **)calloc(t->blocks ? t->blocks : 1, sizeof(float *));
    t->bw2 = (const float **)calloc(t->blocks ? t->blocks : 1, sizeof(float *));
    t->bb2 = (const float **)calloc(t->blocks ? t->blocks : 1, sizeof(float *));
    if (!t->buf || !t->bw1 || !t->bb1 || !t->bw2 || !t->bb2) return NULL;
    memcpy(t->buf, weights, sizeof(float) * (size_t)n_weights);
    const float *p = t->buf;
    int C = t->C;
    if (t->k) { t->pmean = take(&p, 2 * CELLS); t->pU = take(&p, (size_t)2 * CELLS * t->k); }
    t->stem_w = take(&p, (size_t)C * 4 * 9); t->stem_b = take(&p, C);
    for (int i = 0; i < t->blocks; i++) {
        t->bw1[i] = take(&p, (size_t)C * C * 9); t->bb1[i] = take(&p, C);
        t->bw2[i] = take(&p, (size_t)C * C * 9); t->bb2[i] = take(&p, C);
    }
    t->pol_w = take(&p, (size_t)t->pc * C); t->pol_b = take(&p, t->pc);
    t->polfc_w = take(&p, (size_t)7 * t->pc * CELLS); t->polfc_b = take(&p, 7);
    t->val_w = take(&p, (size_t)t->vc * C); t->val_b = take(&p, t->vc);
    t->v1_w = take(&p, (size_t)t->vh * t->vc * CELLS); t->v1_b = take(&p, t->vh);
    t->v2_w = take(&p, (size_t)3 * t->vh); t->v2_b = take(&p, 3);
    if (t->has_score) {
        t->sc_w = take(&p, (size_t)t->sc * C); t->sc_b = take(&p, t->sc);
        t->s1_w = take(&p, (size_t)t->sh * t->sc * CELLS); t->s1_b = take(&p, t->sh);
        t->s2_w = take(&p, (size_t)7 * t->sh); t->s2_b = take(&p, 7);
    }
    if ((long long)(p - t->buf) != n_weights) { free(t->buf); free(t); return NULL; }
    t->pk_stem = pack_w(t->stem_w, C, 4 * 9);
    t->pk1 = (float **)calloc(t->blocks ? t->blocks : 1, sizeof(float *));
    t->pk2 = (float **)calloc(t->blocks ? t->blocks : 1, sizeof(float *));
    if (!t->pk_stem || !t->pk1 || !t->pk2) return NULL;
    for (int i = 0; i < t->blocks; i++) {
        t->pk1[i] = pack_w(t->bw1[i], C, C * 9);
        t->pk2[i] = pack_w(t->bw2[i], C, C * 9);
        if (!t->pk1[i] || !t->pk2[i]) return NULL;
    }
    return t;
}

EXPORT void c4n_free(void *h) {
    Net *t = (Net *)h;
    if (!t) return;
    for (int i = 0; t->pk1 && i < t->blocks; i++) { free(t->pk1[i]); free(t->pk2[i]); }
    free(t->pk_stem); free(t->pk1); free(t->pk2);
    free(t->buf); free(t->bw1); free(t->bb1); free(t->bw2); free(t->bb2);
    free(t->x); free(t->h); free(t->y); free(t->y2); free(t->col); free(t->head); free(t->hid);
    free(t);
}

static int ensure(Net *t, int nb) {
    if (nb <= t->cap_b) return 1;
    int C = t->C, mx = C > 4 ? C : 4;
    int hc = t->pc; if (t->vc > hc) hc = t->vc; if (t->has_score && t->sc > hc) hc = t->sc;
    int hd = t->vh; if (t->has_score && t->sh > hd) hd = t->sh; if (hd < 7) hd = 7;
    free(t->x); free(t->h); free(t->y); free(t->y2); free(t->col); free(t->head); free(t->hid);
    size_t N = (size_t)nb * CELLS;
    t->x = (float *)malloc(sizeof(float) * 4 * N);
    t->h = (float *)malloc(sizeof(float) * C * N);
    t->y = (float *)malloc(sizeof(float) * C * N);
    t->y2 = (float *)malloc(sizeof(float) * C * N);
    t->col = (float *)malloc(sizeof(float) * (size_t)9 * mx * (N + NB));
    t->head = (float *)malloc(sizeof(float) * hc * N);
    t->hid = (float *)malloc(sizeof(float) * (size_t)hd * nb);
    t->cap_b = nb;
    return t->x && t->h && t->y && t->y2 && t->col && t->head && t->hid;
}

/* Activations are [channel][N] with N = nb*42 (position-major, then row, then column).
   3x3 conv (padding 1) as im2col + matrix product: out[o][n] = b[o] + sum_k W[o][k] col[k][n]. */
/* 3x3 conv (padding 1) as im2col + a register-blocked matrix product:
   out[o][n] = b[o] + sum_k W[o][k] col[k][n],  k = i*9 + dy*3 + dx.
   col rows are padded to a multiple of 8 columns so the inner block is one AVX register. */
static void conv3x3(const float *in, int ci, float *out, int co, const float *pk, const float *b,
                    float *col, int nb) {
    int N = nb * CELLS, Np = (N + NB - 1) / NB * NB, K = ci * 9;
    for (int i = 0; i < ci; i++)
        for (int dy = 0; dy < 3; dy++)
            for (int dx = 0; dx < 3; dx++) {
                float *dst = col + (size_t)((i * 9) + dy * 3 + dx) * Np;
                const float *src = in + (size_t)i * N;
                for (int p = 0; p < nb; p++)
                    for (int r = 0; r < ROWS; r++) {
                        int rr = r + dy - 1;
                        for (int c = 0; c < COLS; c++) {
                            int cc = c + dx - 1;
                            dst[p * CELLS + r * COLS + c] =
                                (rr < 0 || rr >= ROWS || cc < 0 || cc >= COLS) ? 0.0f : src[p * CELLS + rr * COLS + cc];
                        }
                    }
                for (int n = N; n < Np; n++) dst[n] = 0.0f;
            }
    int nob = (co + OB - 1) / OB;
    for (int n0 = 0; n0 < Np; n0 += NB)            /* column block outer: its im2col slice stays in L1 */
    for (int ob_i = 0; ob_i < nob; ob_i++) {
        const float *wp = pk + (size_t)ob_i * K * OB;
        int o0 = ob_i * OB, ob = co - o0 < OB ? co - o0 : OB;
        {
            float res[OB][NB];
#ifdef C4_AVX2
            __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
            for (int k = 0; k < K; k++) {
                __m256 c = _mm256_loadu_ps(col + (size_t)k * Np + n0);
                const float *wk = wp + (size_t)k * OB;
                a0 = _mm256_fmadd_ps(_mm256_broadcast_ss(wk + 0), c, a0);
                a1 = _mm256_fmadd_ps(_mm256_broadcast_ss(wk + 1), c, a1);
                a2 = _mm256_fmadd_ps(_mm256_broadcast_ss(wk + 2), c, a2);
                a3 = _mm256_fmadd_ps(_mm256_broadcast_ss(wk + 3), c, a3);
                a4 = _mm256_fmadd_ps(_mm256_broadcast_ss(wk + 4), c, a4);
                a5 = _mm256_fmadd_ps(_mm256_broadcast_ss(wk + 5), c, a5);
                a6 = _mm256_fmadd_ps(_mm256_broadcast_ss(wk + 6), c, a6);
                a7 = _mm256_fmadd_ps(_mm256_broadcast_ss(wk + 7), c, a7);
            }
            _mm256_storeu_ps(res[0], a0); _mm256_storeu_ps(res[1], a1); _mm256_storeu_ps(res[2], a2);
            _mm256_storeu_ps(res[3], a3); _mm256_storeu_ps(res[4], a4); _mm256_storeu_ps(res[5], a5);
            _mm256_storeu_ps(res[6], a6); _mm256_storeu_ps(res[7], a7);
#else
            for (int j = 0; j < OB; j++)
                for (int v = 0; v < NB; v++) res[j][v] = 0.0f;
            for (int k = 0; k < K; k++) {
                const float *cr = col + (size_t)k * Np + n0;
                const float *wk = wp + (size_t)k * OB;
                for (int j = 0; j < OB; j++)
                    for (int v = 0; v < NB; v++) res[j][v] += wk[j] * cr[v];
            }
#endif
            for (int j = 0; j < ob; j++) {
                float bo = b[o0 + j];
                float *dst = out + (size_t)(o0 + j) * N;
                for (int v = 0; v < NB && n0 + v < N; v++) dst[n0 + v] = bo + res[j][v];
            }
        }
    }
}

static void conv1x1_relu(const float *in, int ci, float *out, int co, const float *w, const float *b, int nb) {
    int N = nb * CELLS;
    for (int o = 0; o < co; o++) {
        float *dst = out + (size_t)o * N;
        for (int n = 0; n < N; n++) dst[n] = b[o];
        for (int i = 0; i < ci; i++) {
            float wk = w[(size_t)o * ci + i];
            const float *src = in + (size_t)i * N;
            for (int n = 0; n < N; n++) dst[n] += wk * src[n];
        }
        for (int n = 0; n < N; n++) if (dst[n] < 0) dst[n] = 0;
    }
}

/* head features of position p in numpy's flatten order: [channel][row][col] */
static void dense_from_head(const float *head, int hc, int nb, int p, const float *w, const float *b,
                            int out_n, float *out, int relu) {
    int N = nb * CELLS, in_n = hc * CELLS;
    for (int o = 0; o < out_n; o++) {
        const float *wr = w + (size_t)o * in_n;
        float s = b[o];
        for (int c = 0; c < hc; c++) {
            const float *src = head + (size_t)c * N + p * CELLS;
            const float *wc = wr + c * CELLS;
            for (int j = 0; j < CELLS; j++) s += wc[j] * src[j];
        }
        out[o] = relu && s < 0 ? 0 : s;
    }
}

static void dense(const float *in, int in_n, const float *w, const float *b, int out_n, float *out, int relu) {
    for (int o = 0; o < out_n; o++) {
        float s = b[o];
        const float *wr = w + (size_t)o * in_n;
        for (int i = 0; i < in_n; i++) s += wr[i] * in[i];
        out[o] = relu && s < 0 ? 0 : s;
    }
}

/* nb encoded positions in t->x ([plane][nb*42]) -> logits (nb*7), wdl logits (nb*3), scores (nb*7) */
static void forward(Net *t, int nb, float *pol, float *wdl, float *sco) {
    int N = nb * CELLS, C = t->C;
    if (t->k) {
        float z[2 * CELLS], tmp[256];
        for (int p = 0; p < nb; p++) {
            for (int pl = 0; pl < 2; pl++)
                for (int j = 0; j < CELLS; j++) z[pl * CELLS + j] = t->x[(size_t)pl * N + p * CELLS + j] - t->pmean[pl * CELLS + j];
            int k = t->k < 256 ? t->k : 256;
            for (int a = 0; a < k; a++) {
                float s = 0;
                for (int j = 0; j < 2 * CELLS; j++) s += z[j] * t->pU[(size_t)j * t->k + a];
                tmp[a] = s;
            }
            for (int j = 0; j < 2 * CELLS; j++) {
                float s = 0;
                for (int a = 0; a < k; a++) s += tmp[a] * t->pU[(size_t)j * t->k + a];
                t->x[(size_t)(j / CELLS) * N + p * CELLS + (j % CELLS)] = s + t->pmean[j];
            }
        }
    }
    conv3x3(t->x, 4, t->h, C, t->pk_stem, t->stem_b, t->col, nb);
    for (size_t i = 0; i < (size_t)C * N; i++) if (t->h[i] < 0) t->h[i] = 0;
    for (int b = 0; b < t->blocks; b++) {
        conv3x3(t->h, C, t->y, C, t->pk1[b], t->bb1[b], t->col, nb);
        for (size_t i = 0; i < (size_t)C * N; i++) if (t->y[i] < 0) t->y[i] = 0;
        conv3x3(t->y, C, t->y2, C, t->pk2[b], t->bb2[b], t->col, nb);
        for (size_t i = 0; i < (size_t)C * N; i++) { float v = t->h[i] + t->y2[i]; t->h[i] = v < 0 ? 0 : v; }
    }
    conv1x1_relu(t->h, C, t->head, t->pc, t->pol_w, t->pol_b, nb);
    for (int p = 0; p < nb; p++) dense_from_head(t->head, t->pc, nb, p, t->polfc_w, t->polfc_b, 7, pol + 7 * p, 0);
    conv1x1_relu(t->h, C, t->head, t->vc, t->val_w, t->val_b, nb);
    for (int p = 0; p < nb; p++) {
        dense_from_head(t->head, t->vc, nb, p, t->v1_w, t->v1_b, t->vh, t->hid, 1);
        dense(t->hid, t->vh, t->v2_w, t->v2_b, 3, wdl + 3 * p, 0);
    }
    if (t->has_score && sco) {
        conv1x1_relu(t->h, C, t->head, t->sc, t->sc_w, t->sc_b, nb);
        for (int p = 0; p < nb; p++) {
            dense_from_head(t->head, t->sc, nb, p, t->s1_w, t->s1_b, t->sh, t->hid, 1);
            dense(t->hid, t->sh, t->s2_w, t->s2_b, 7, sco + 7 * p, 0);
            for (int j = 0; j < 7; j++) sco[7 * p + j] *= SCORE_SCALE;
        }
    }
}

static int popc(uint64_t x) { int n = 0; while (x) { x &= x - 1; n++; } return n; }

/* encode positions [0,nb) at slots [off, off+nb) of a batch of tot (mirror: flipped columns) */
static void encode_at(Net *t, int tot, int off, int nb, const uint64_t *cur, const uint64_t *mask, int mirror) {
    int N = tot * CELLS;
    for (int p = 0; p < nb; p++) {
        uint64_t me = cur[p], them = cur[p] ^ mask[p];
        float first = popc(me) == popc(them) ? 1.0f : 0.0f;
        for (int r = 0; r < ROWS; r++)
            for (int c = 0; c < COLS; c++) {
                int sc = mirror ? COLS - 1 - c : c;
                int bit = sc * H1 + (ROWS - 1 - r);
                int j = (off + p) * CELLS + r * COLS + c;
                t->x[j] = (float)((me >> bit) & 1);
                t->x[N + j] = (float)((them >> bit) & 1);
                t->x[2 * N + j] = 1.0f;
                t->x[3 * N + j] = first;
            }
    }
}

#define CHUNK 8      /* positions per forward pass: keeps im2col in cache */

/* Same as NumpyNet.evaluate: priors (nb*7, double), values (nb, double), wdl (nb*3, float or NULL). */
static int evaluate_chunk(Net *t, int nb, const uint64_t *cur, const uint64_t *mask,
                          double *pri, double *val, float *wdl_out);

EXPORT int c4n_evaluate(void *h, int nb, const uint64_t *cur, const uint64_t *mask,
                        double *pri, double *val, float *wdl_out) {
    Net *t = (Net *)h;
    for (int i = 0; i < nb; i += CHUNK) {
        int k = nb - i < CHUNK ? nb - i : CHUNK;
        if (evaluate_chunk(t, k, cur + i, mask + i, pri + 7 * i, val + i, wdl_out ? wdl_out + 3 * i : NULL)) return -1;
    }
    return 0;
}

static int evaluate_chunk(Net *t, int nb, const uint64_t *cur, const uint64_t *mask,
                          double *pri, double *val, float *wdl_out) {
    int tot = t->mirror ? 2 * nb : nb;              /* mirror images go in the same forward pass */
    if (!ensure(t, tot)) return -1;
    float pol[2 * CHUNK * 7], wdl[2 * CHUNK * 3], scob[2 * CHUNK * 7];
    float *sco = t->has_score ? scob : NULL;
    encode_at(t, tot, 0, nb, cur, mask, 0);
    if (t->mirror) encode_at(t, tot, nb, nb, cur, mask, 1);
    forward(t, tot, pol, wdl, sco);
    if (t->mirror) {
        for (int p = 0; p < nb; p++) {
            for (int j = 0; j < 7; j++) {
                pol[7 * p + j] = (pol[7 * p + j] + pol[7 * (nb + p) + 6 - j]) / 2;
                if (sco) sco[7 * p + j] = (sco[7 * p + j] + sco[7 * (nb + p) + 6 - j]) / 2;
            }
            for (int j = 0; j < 3; j++) wdl[3 * p + j] = (wdl[3 * p + j] + wdl[3 * (nb + p) + j]) / 2;
        }
    }
    for (int p = 0; p < nb; p++) {
        int legal[7];
        for (int c = 0; c < 7; c++) legal[c] = !((mask[p] >> (c * H1 + ROWS - 1)) & 1);
        float z[7];
        for (int c = 0; c < 7; c++) z[c] = pol[7 * p + c];
        if (t->play_head && sco) {
            float lp[7];
            if (t->play_head == 2 && t->play_w != 0.0f) {
                float mx = -INFINITY, se = 0;
                for (int c = 0; c < 7; c++) if (legal[c] && pol[7 * p + c] > mx) mx = pol[7 * p + c];
                for (int c = 0; c < 7; c++) if (legal[c]) se += expf(pol[7 * p + c] - mx);
                float lse = logf(se);
                for (int c = 0; c < 7; c++) lp[c] = legal[c] ? pol[7 * p + c] - mx - lse : 0.0f;
            }
            for (int c = 0; c < 7; c++) {
                z[c] = sco[7 * p + c] / t->play_temp;
                if (t->play_head == 2 && t->play_w != 0.0f) z[c] += t->play_w * lp[c];
            }
        }
        float mx = -INFINITY, s = 0, e[7];
        for (int c = 0; c < 7; c++) if (legal[c] && z[c] > mx) mx = z[c];
        for (int c = 0; c < 7; c++) { e[c] = legal[c] ? expf(z[c] - mx) : 0.0f; s += e[c]; }
        for (int c = 0; c < 7; c++) pri[7 * p + c] = (double)(e[c] / s);
        float wm = wdl[3 * p]; if (wdl[3 * p + 1] > wm) wm = wdl[3 * p + 1]; if (wdl[3 * p + 2] > wm) wm = wdl[3 * p + 2];
        float we[3], ws = 0;
        for (int j = 0; j < 3; j++) { we[j] = expf(wdl[3 * p + j] - wm); ws += we[j]; }
        for (int j = 0; j < 3; j++) { we[j] /= ws; if (wdl_out) wdl_out[3 * p + j] = we[j]; }
        val[p] = (double)(we[0] - we[2]);
        if (t->value_score && sco) {
            double best = -INFINITY;
            for (int c = 0; c < 7; c++) if (legal[c] && sco[7 * p + c] > best) best = sco[7 * p + c];
            double a = fabs(best) < 21.0 ? fabs(best) : 21.0;
            val[p] = tanh(best) * (0.75 + 0.25 * a / 21.0);
        }
    }
    return 0;
}

/* ------------------------------------------------------------ position cache */
typedef struct { uint64_t cur, mask; double pri[7]; double val; int used; } CEntry;
typedef struct { CEntry *t; size_t cap, n; long long asked, computed; } Cache;

EXPORT void *c4n_cache_create(int log2cap) {
    Cache *c = (Cache *)calloc(1, sizeof(Cache));
    if (!c) return NULL;
    c->cap = (size_t)1 << (log2cap > 4 ? log2cap : 4);
    c->t = (CEntry *)calloc(c->cap, sizeof(CEntry));
    if (!c->t) { free(c); return NULL; }
    return c;
}
EXPORT void c4n_cache_free(void *h) { Cache *c = (Cache *)h; if (c) { free(c->t); free(c); } }
EXPORT void c4n_cache_stats(void *h, long long *asked, long long *computed) {
    Cache *c = (Cache *)h; *asked = c->asked; *computed = c->computed;
}
static size_t cslot(Cache *c, uint64_t cur, uint64_t mask) {
    uint64_t k = (cur * 0x9E3779B97F4A7C15ULL) ^ (mask * 0xC2B2AE3D27D4EB4FULL);
    size_t i = (size_t)(k >> 17) & (c->cap - 1);
    while (c->t[i].used && !(c->t[i].cur == cur && c->t[i].mask == mask)) i = (i + 1) & (c->cap - 1);
    return i;
}

/* evaluate nb positions through the cache (cache may be NULL) */
static int eval_cached(Net *t, Cache *c, int nb, const uint64_t *cur, const uint64_t *mask, double *pri, double *val) {
    if (!c) return c4n_evaluate(t, nb, cur, mask, pri, val, NULL);
    uint64_t *mc = (uint64_t *)malloc(sizeof(uint64_t) * nb * 2);
    int *where = (int *)malloc(sizeof(int) * nb);
    if (!mc || !where) return -1;
    int nm = 0;
    c->asked += nb;
    for (int p = 0; p < nb; p++) {
        size_t i = cslot(c, cur[p], mask[p]);
        if (c->t[i].used) { memcpy(pri + 7 * p, c->t[i].pri, sizeof(double) * 7); val[p] = c->t[i].val; where[p] = -1; }
        else { mc[nm] = cur[p]; mc[nb + nm] = mask[p]; where[p] = nm++; }
    }
    if (nm) {
        double *mp = (double *)malloc(sizeof(double) * 7 * nm), *mv = (double *)malloc(sizeof(double) * nm);
        if (!mp || !mv) return -1;
        if (c4n_evaluate(t, nm, mc, mc + nb, mp, mv, NULL)) return -1;
        c->computed += nm;
        if ((c->n + nm) * 2 > c->cap) { memset(c->t, 0, sizeof(CEntry) * c->cap); c->n = 0; }
        for (int p = 0; p < nb; p++) {
            if (where[p] < 0) continue;
            int q = where[p];
            memcpy(pri + 7 * p, mp + 7 * q, sizeof(double) * 7); val[p] = mv[q];
            size_t i = cslot(c, cur[p], mask[p]);
            if (!c->t[i].used) { c->t[i].used = 1; c->t[i].cur = cur[p]; c->t[i].mask = mask[p]; c->n++;
                                 memcpy(c->t[i].pri, mp + 7 * q, sizeof(double) * 7); c->t[i].val = mv[q]; }
        }
        free(mp); free(mv);
    }
    free(mc); free(where);
    return 0;
}

/* Fully native MCTS on n roots.  Outputs per root: N (n*7), W (n*7), P (n*7), root value,
   visits n, PV (n*42, -1 padded), max depth.  Returns 0, or -1 on out of memory. */
EXPORT int c4n_search(void *net, void *cache, int n, const uint64_t *cur, const uint64_t *mask, int sims,
                      double c_puct, double fpu, int *N, double *W, double *P, double *value, int *visits,
                      int *pv, int *max_depth) {
    Net *t = (Net *)net;
    void *tr = c4m_create(n, cur, mask, c_puct, fpu);
    if (!tr) return -1;
    uint64_t *oc = (uint64_t *)malloc(sizeof(uint64_t) * (n ? n : 1));
    uint64_t *om = (uint64_t *)malloc(sizeof(uint64_t) * (n ? n : 1));
    double *pri = (double *)malloc(sizeof(double) * 7 * (n ? n : 1));
    double *val = (double *)malloc(sizeof(double) * (n ? n : 1));
    int rc = 0;
    if (!oc || !om || !pri || !val) rc = -1;
    for (int step = 0; rc == 0 && step <= sims; step++) {
        int k = c4m_collect(tr, oc, om);
        if (k < 0) { rc = -1; break; }
        if (k && eval_cached(t, (Cache *)cache, k, oc, om, pri, val)) { rc = -1; break; }
        c4m_apply(tr, pri, val, NULL);
    }
    if (rc == 0) {
        for (int i = 0; i < n; i++) {
            c4m_root_stats(tr, i, N + 7 * i, W + 7 * i, P + 7 * i, visits + i, value + i);
            int *row = pv + 42 * i;
            int len = c4m_pv(tr, i, row, 42);
            for (int j = len; j < 42; j++) row[j] = -1;
            max_depth[i] = c4m_max_depth(tr, i);
        }
    }
    free(oc); free(om); free(pri); free(val);
    c4m_free(tr);
    return rc;
}

/* 1 if the AVX2/FMA network kernel was compiled in */
EXPORT int c4n_simd(void) {
#ifdef C4_AVX2
    return 1;
#else
    return 0;
#endif
}
