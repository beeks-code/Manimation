
"""

from manim import *
import numpy as np

# ─── PALETTE ────────────────────────────────────────────────────────────────
BG      = "#0f172a"
C_Q     = "#3b82f6"   # blue  — Query
C_K     = "#22c55e"   # green — Key
C_V     = "#f97316"   # orange— Value
C_GLOW  = "#a78bfa"   # purple glow
C_TEXT  = "#e2e8f0"
C_DIM   = "#475569"
C_ACCT  = "#f472b6"   # pink accent

config.background_color = BG
config.pixel_height = 1080
config.pixel_width  = 1920
config.frame_rate   = 60

# ─── MATRIX DIMENSIONS ───────────────────────────────────────────────────────
N   = 3   # tokens: "I", "love", "math"
DIM = 3   # visual d_model columns (kept small for clarity)

# ─── FIXED DATA ──────────────────────────────────────────────────────────────
np.random.seed(42)
Q_DATA = np.round(np.random.uniform(0.3, 1.2, (N, DIM)), 2)
K_DATA = np.round(np.random.uniform(0.3, 1.2, (N, DIM)), 2)
V_DATA = np.round(np.random.uniform(0.3, 1.2, (N, DIM)), 2)

S_RAW   = Q_DATA @ K_DATA.T
SCALE   = np.sqrt(DIM)
S_SCALE = S_RAW / SCALE

def softmax_row(x):
    e = np.exp(x - x.max())
    return e / e.sum()

A_DATA = np.array([softmax_row(r) for r in S_SCALE])
O_DATA = A_DATA @ V_DATA

CELL = 0.60   # cell side length — matches reference image look

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def safe_vg(*mobs):
    return VGroup(*[m for m in mobs if m is not None])


def make_grid_bg():
    lines = VGroup()
    for i in range(29):
        x = -config.frame_width / 2 + i * config.frame_width / 28
        lines.add(Line([x, -config.frame_height / 2, 0],
                       [x,  config.frame_height / 2, 0],
                       stroke_color="#1e293b", stroke_width=0.4,
                       stroke_opacity=0.5))
    for j in range(17):
        y = -config.frame_height / 2 + j * config.frame_height / 16
        lines.add(Line([-config.frame_width / 2, y, 0],
                       [ config.frame_width / 2, y, 0],
                       stroke_color="#1e293b", stroke_width=0.4,
                       stroke_opacity=0.5))
    return lines


def build_matrix(rows, cols, stroke_color, fill_color=None,
                 cell=CELL, fill_op=0.10):
    """
    Returns VGroup with:
      [0] = cells VGroup  (row-major: index = r*cols + c)
      [1] = left  bracket
      [2] = right bracket
    Does NOT include label — caller adds label separately.
    """
    fc = fill_color or stroke_color
    cells = VGroup()
    for r in range(rows):
        for c in range(cols):
            sq = Square(
                side_length=cell,
                color=stroke_color,
                fill_color=fc,
                fill_opacity=fill_op,
                stroke_width=1.6,
            )
            sq.move_to([
                c * cell - (cols - 1) * cell / 2,
                -r * cell + (rows - 1) * cell / 2,
                0
            ])
            cells.add(sq)

    # bracket style matching reference image
    bl = MathTex(r"\big[", color=stroke_color, font_size=int(cell * 130))
    br = MathTex(r"\big]", color=stroke_color, font_size=int(cell * 130))
    bl.next_to(cells, LEFT,  buff=0.04)
    br.next_to(cells, RIGHT, buff=0.04)

    return VGroup(cells, bl, br)


def matrix_label(tex, color, size=34):
    return MathTex(tex, color=color, font_size=size)


def dim_label(tex, color, size=22):
    return MathTex(tex, color=color, font_size=size)


def hl_row(mat, r, cols, color, buff=0.05, width=2.5):
    row_cells = VGroup(*[mat[0][r * cols + c] for c in range(cols)])
    return SurroundingRectangle(row_cells, color=color,
                                buff=buff, stroke_width=width)


def hl_col(mat, c, rows, cols, color, buff=0.05, width=2.5):
    col_cells = VGroup(*[mat[0][rr * cols + c] for rr in range(rows)])
    return SurroundingRectangle(col_cells, color=color,
                                buff=buff, stroke_width=width)


def cell_txt(val, color, size=16):
    return Text(f"{val:.2f}", font="Fira Code", color=color, font_size=size)


def glow_pulse(scene, mob, n=1):
    for _ in range(n):
        scene.play(mob.animate(rate_func=there_and_back,
                               run_time=0.40).scale(1.05))


def sec_title(scene, txt, color=C_TEXT, fs=36):
    t = Text(txt, font="Fira Code Bold", color=color,
             font_size=fs, weight=BOLD)
    t.to_edge(UP, buff=0.28)
    scene.play(FadeIn(t, shift=DOWN * 0.15, run_time=0.45))
    return t


# ─── SCENE ───────────────────────────────────────────────────────────────────

class AttentionFull(Scene):

    def construct(self):
        self.add(make_grid_bg())
        self.s1_qk_intro()
        self.s2_transpose_k()
        self.s3_multiply_qkt()
        self.s4_scale_softmax()
        self.s5_multiply_v()
        self.s6_final_output()

    # ═══════════════════════════════════════════════════════════════════════
    # S1 — Q and K side by side (matching reference image style)
    # ═══════════════════════════════════════════════════════════════════════
    def s1_qk_intro(self):
        title = sec_title(self, "Query (Q) and Key (K) Matrices")

        # ── Q Matrix ────────────────────────────────────────────────────
        q_mat = build_matrix(N, DIM, C_Q)
        q_mat.move_to(LEFT * 4.0 + DOWN * 0.2)

        q_lbl = matrix_label(r"Q", C_Q, size=38)
        q_lbl.next_to(q_mat, UP, buff=0.22)

        q_dim = dim_label(
            r"{\scriptscriptstyle 3 \times d_{\text{model}}}",
            C_Q, size=25
        )
        q_dim.next_to(q_mat, DOWN, buff=0.18)

        # ── K Matrix ────────────────────────────────────────────────────
        k_mat = build_matrix(N, DIM, C_K)
        k_mat.move_to(RIGHT * 4.0 + DOWN * 0.2)

        k_lbl = matrix_label(r"K", C_K, size=38)
        k_lbl.next_to(k_mat, UP, buff=0.22)

        k_dim = dim_label(
            r"{\scriptscriptstyle 3 \times d_{\text{model}}}",
            C_K, size=25
        )
        k_dim.next_to(k_mat, DOWN, buff=0.18)

        # ── Animate in ──────────────────────────────────────────────────
        self.play(
            FadeIn(q_mat, scale=0.88, run_time=0.7, rate_func=smooth),
            FadeIn(q_lbl, shift=DOWN * 0.1, run_time=0.5),
        )
        self.play(FadeIn(q_dim, run_time=0.4))

        # glow border pulse on Q
        qb = SurroundingRectangle(q_mat[0], color=C_GLOW,
                                   buff=0.08, stroke_width=2.5)
        self.play(Create(qb, run_time=0.3))
        self.play(FadeOut(qb, run_time=0.3))

        self.play(
            FadeIn(k_mat, scale=0.88, run_time=0.7, rate_func=smooth),
            FadeIn(k_lbl, shift=DOWN * 0.1, run_time=0.5),
        )
        self.play(FadeIn(k_dim, run_time=0.4))

        kb = SurroundingRectangle(k_mat[0], color=C_GLOW,
                                   buff=0.08, stroke_width=2.5)
        self.play(Create(kb, run_time=0.3))
        self.play(FadeOut(kb, run_time=0.3))

        self.wait(0.8)
        self.play(FadeOut(safe_vg(title, q_dim, k_dim), run_time=0.4))

        self.q_mat = q_mat
        self.q_lbl = q_lbl
        self.k_mat = k_mat
        self.k_lbl = k_lbl

    # ═══════════════════════════════════════════════════════════════════════
    # S2 — Transpose K
    def s2_transpose_k(self):
        title = sec_title(self, "Transpose K  →  Kᵀ", color=C_K)

        # dim Q while we work on K
        self.play(
            self.q_mat.animate(run_time=0.35).set_opacity(0.18),
            self.q_lbl.animate(run_time=0.35).set_opacity(0.18),
        )

        # ── highlight all 3 rows of K one by one (row → becomes col) ──────
        for r in range(N):
            hl = hl_row(self.k_mat, r, DIM, C_K)
            self.play(Create(hl, run_time=0.22))
            self.play(FadeOut(hl, run_time=0.18))

        # ── build Kᵀ at the same position as K, then slide it right ───────
        kt_mat = build_matrix(DIM, N, C_K)   # shape flipped: DIM rows × N cols
        kt_mat.move_to(self.k_mat.get_center())   # start at K's position

        kt_lbl = matrix_label(r"K^T", C_K, size=38)
        kt_lbl.next_to(kt_mat, UP, buff=0.22)

        # Rotate K copy 90° in place → lands as Kᵀ shape
        k_copy = self.k_mat.copy()
        self.play(
            Rotate(k_copy, angle=PI / 2, run_time=0.55, rate_func=smooth),
            FadeOut(self.k_mat, run_time=0.30),
            FadeOut(self.k_lbl, run_time=0.30),
        )
        self.play(
            ReplacementTransform(k_copy, kt_mat, run_time=0.50, rate_func=smooth),
            FadeIn(kt_lbl, run_time=0.35),
        )

        # slide Kᵀ to its final position (right side)
        self.play(
            kt_mat.animate(rate_func=smooth, run_time=0.55)
                .move_to(RIGHT * 4.0 + DOWN * 0.2),
            kt_lbl.animate(rate_func=smooth, run_time=0.55)
                .move_to(RIGHT * 4.0 + UP * (DIM * CELL / 2 + 0.38)),
        )

        self.wait(0.4)
        self.play(FadeOut(title, run_time=0.35))

        # restore Q
        self.play(
            self.q_mat.animate(run_time=0.35).set_opacity(1.0),
            self.q_lbl.animate(run_time=0.35).set_opacity(1.0),
        )

        self.kt_mat = kt_mat
        self.kt_lbl = kt_lbl

    # ═══════════════════════════════════════════════════════════════════════
    # S3 — Q × Kᵀ cell-by-cell
    # ═══════════════════════════════════════════════════════════════════════
    def s3_multiply_qkt(self):
        title = sec_title(self, "Q × Kᵀ  —  Dot Product per Cell", color=C_GLOW)

        # rearrange: Q left, Kᵀ right, result center
        self.play(
            self.q_mat.animate(rate_func=smooth, run_time=0.65)
                      .move_to(LEFT * 5.5 + DOWN * 0.2),
            self.q_lbl.animate(rate_func=smooth, run_time=0.65)
                      .next_to(LEFT * 5.5 + DOWN * 0.2 +
                               UP * (N * CELL / 2 + 0.3), ORIGIN),
            self.kt_mat.animate(rate_func=smooth, run_time=0.65)
                       .move_to(LEFT * 0.8 + DOWN * 0.2),
            self.kt_lbl.animate(rate_func=smooth, run_time=0.65)
                       .next_to(LEFT * 0.8 + DOWN * 0.2 +
                                UP * (DIM * CELL / 2 + 0.3), ORIGIN),
        )

        # result matrix shell (N × N)
        r_mat = build_matrix(N, N, C_GLOW)
        r_mat.move_to(RIGHT * 4.8 + DOWN * 0.2)
        r_lbl = matrix_label(r"QK^T\ \ (3\times 3)", C_GLOW, size=28)
        r_lbl.next_to(r_mat, UP, buff=0.22)
        self.play(FadeIn(r_mat, run_time=0.5), FadeIn(r_lbl, run_time=0.4))

        val_mobs = {}

        for i in range(N):
            for j in range(N):
                # highlight row i of Q, col j of Kᵀ
                hl_r = hl_row(self.q_mat, i, DIM, C_Q)
                hl_c = hl_col(self.kt_mat, j, DIM, N, C_K)
                self.play(Create(hl_r, run_time=0.20),
                          Create(hl_c, run_time=0.20))

                # dot product formula
                terms = " + ".join(
                    [f"{Q_DATA[i,k]:.1f}·{K_DATA[j,k]:.1f}"
                     for k in range(DIM)]
                )
                val = S_RAW[i, j]
                dp_txt = Text(f"{terms} = {val:.2f}",
                              font="Fira Code", color=C_GLOW, font_size=17)
                dp_txt.next_to(r_mat, DOWN, buff=0.32)
                self.play(FadeIn(dp_txt, run_time=0.22))

                # place value in result cell
                target = r_mat[0][i * N + j]
                vtxt = cell_txt(val, C_GLOW, size=17)
                vtxt.move_to(target)
                val_mobs[(i, j)] = vtxt

                intensity = (val - S_RAW.min()) / (S_RAW.max() - S_RAW.min() + 1e-8)
                fill_col = interpolate_color(ManimColor(C_DIM),
                                             ManimColor(C_GLOW), intensity)
                self.play(
                    target.animate(run_time=0.22).set_fill(fill_col, opacity=0.40),
                    FadeIn(vtxt, scale=0.7, run_time=0.22),
                )
                self.play(FadeOut(safe_vg(hl_r, hl_c, dp_txt), run_time=0.18))

        self.wait(0.7)
        self.play(FadeOut(safe_vg(title), run_time=0.35))

        self.r_mat  = r_mat
        self.r_lbl  = r_lbl
        self.val_mobs = val_mobs

    # ═══════════════════════════════════════════════════════════════════════
    # S4 — Scale + Softmax
    # ═══════════════════════════════════════════════════════════════════════
    def s4_scale_softmax(self):
        title = sec_title(self,
                          f"Scale ÷ √{DIM}  then  Softmax", color=C_ACCT)

        # scale formula
        scale_f = MathTex(
            r"S' = \frac{QK^T}{\sqrt{d_{\text{model}}}}",
            color=C_TEXT, font_size=32
        )
        scale_f.next_to(self.r_mat, DOWN, buff=0.38)
        self.play(Write(scale_f, run_time=0.7))
        self.wait(0.3)

        # update each cell to scaled value (numbers shrink, color shifts)
        scale_val_mobs = {}
        for i in range(N):
            for j in range(N):
                old = self.val_mobs[(i, j)]
                newv = S_SCALE[i, j]
                ntxt = cell_txt(newv, C_ACCT, size=15)
                ntxt.move_to(self.r_mat[0][i * N + j])
                scale_val_mobs[(i, j)] = ntxt
                self.play(
                    FadeOut(old, run_time=0.15),
                    self.r_mat[0][i * N + j].animate(run_time=0.18)
                        .set_fill(C_ACCT, opacity=0.20),
                    FadeIn(ntxt, run_time=0.15),
                    run_time=0.18
                )

        self.wait(0.3)

        # softmax formula
        sm_f = MathTex(
            r"A = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_{\text{model}}}}\right)",
            color=C_TEXT, font_size=30
        )
        sm_f.next_to(scale_f, DOWN, buff=0.22)
        self.play(Write(sm_f, run_time=0.7))
        self.wait(0.3)

        # softmax row by row
        attn_val_mobs = {}
        for i in range(N):
            hl = hl_row(self.r_mat, i, N, C_Q)
            self.play(Create(hl, run_time=0.22))
            for j in range(N):
                old = scale_val_mobs[(i, j)]
                av  = A_DATA[i, j]
                ntxt = cell_txt(av, C_Q, size=15)
                ntxt.move_to(self.r_mat[0][i * N + j])
                attn_val_mobs[(i, j)] = ntxt
                warm = interpolate_color(ManimColor(C_DIM),
                                         ManimColor(C_Q), av)
                self.play(
                    FadeOut(old, run_time=0.15),
                    self.r_mat[0][i * N + j].animate(run_time=0.18)
                        .set_fill(warm, opacity=av * 0.65 + 0.10),
                    FadeIn(ntxt, run_time=0.15),
                    run_time=0.18
                )
            self.play(FadeOut(hl, run_time=0.18))

        # relabel as A
        new_lbl = matrix_label(r"A\ \text{(Attention Weights)}", C_Q, size=26)
        new_lbl.next_to(self.r_mat, UP, buff=0.22)
        self.play(
            Transform(self.r_lbl, new_lbl, run_time=0.45),
        )

        sum_note = Text("Each row sums to 1.0  ✓",
                        font="Fira Code", color=C_K, font_size=20)
        sum_note.next_to(sm_f, DOWN, buff=0.22)
        self.play(FadeIn(sum_note, run_time=0.35))
        self.wait(0.8)

        self.play(FadeOut(safe_vg(title, scale_f, sm_f, sum_note),
                          run_time=0.4))

        self.attn_val_mobs = attn_val_mobs
        self.a_mat = self.r_mat

    # ═══════════════════════════════════════════════════════════════════════
    # S5 — Multiply with V
    # ═══════════════════════════════════════════════════════════════════════
    def s5_multiply_v(self):
        title = sec_title(self, "Attention Weights  ×  V  →  Output",
                          color=C_V)

        # fade / move Q and KT away, bring in V
        self.play(
            self.q_mat.animate(run_time=0.4).set_opacity(0),
            self.q_lbl.animate(run_time=0.4).set_opacity(0),
            self.kt_mat.animate(run_time=0.4).set_opacity(0),
            self.kt_lbl.animate(run_time=0.4).set_opacity(0),
            self.a_mat.animate(rate_func=smooth, run_time=0.65)
                      .move_to(LEFT * 4.8 + DOWN * 0.2),
            self.r_lbl.animate(rate_func=smooth, run_time=0.65)
                      .move_to(LEFT * 4.8 + UP * (N * CELL / 2 + 0.42)),
        )

        # V matrix
        v_mat = build_matrix(N, DIM, C_V)
        v_mat.move_to(LEFT * 0.5 + DOWN * 0.2)
        v_lbl = matrix_label(r"V\ \ (3 \times d_{\text{model}})", C_V, size=28)
        v_lbl.next_to(v_mat, UP, buff=0.22)

        self.play(FadeIn(v_mat, scale=0.88, run_time=0.6, rate_func=smooth),
                  FadeIn(v_lbl, run_time=0.45))

        # output matrix shell
        o_mat = build_matrix(N, DIM, C_V)
        o_mat.move_to(RIGHT * 4.6 + DOWN * 0.2)
        o_lbl = matrix_label(r"O\ \ (3 \times d_{\text{model}})", C_V, size=28)
        o_lbl.next_to(o_mat, UP, buff=0.22)
        self.play(FadeIn(o_mat, run_time=0.5),
                  FadeIn(o_lbl, run_time=0.4))

        o_val_mobs = {}

        for i in range(N):
            hl_a = hl_row(self.a_mat, i, N, C_Q)
            self.play(Create(hl_a, run_time=0.22))

            v_hls = []
            for j in range(N):
                wt = A_DATA[i, j]
                hv = hl_row(v_mat, j, DIM, C_V)
                wt_lbl = Text(f"×{wt:.2f}", font="Fira Code",
                              color=C_V, font_size=15)
                wt_lbl.next_to(hv, LEFT, buff=0.10)
                v_hls.append(safe_vg(hv, wt_lbl))
                self.play(Create(hv, run_time=0.18),
                          FadeIn(wt_lbl, run_time=0.18))

                # elastic stretch of V row by weight
                vcells = VGroup(*[v_mat[0][j * DIM + c] for c in range(DIM)])
                self.play(vcells.animate(rate_func=there_and_back,
                                         run_time=0.22)
                                .stretch(1 + wt * 0.28, 0))

            arr = Arrow(v_mat.get_right() + RIGHT * 0.08,
                        VGroup(*[o_mat[0][i * DIM + c]
                                 for c in range(DIM)]).get_left() + LEFT * 0.08,
                        color=C_V, stroke_width=2.2,
                        tip_length=0.18, buff=0.04)
            self.play(GrowArrow(arr, run_time=0.32))

            for c in range(DIM):
                val  = O_DATA[i, c]
                cell = o_mat[0][i * DIM + c]
                vtxt = cell_txt(val, C_V, size=15)
                vtxt.move_to(cell)
                o_val_mobs[(i, c)] = vtxt
                self.play(
                    cell.animate(run_time=0.18).set_fill(C_V, opacity=0.28),
                    FadeIn(vtxt, run_time=0.18),
                    run_time=0.18
                )

            self.play(FadeOut(safe_vg(hl_a, arr, *v_hls), run_time=0.20))
            self.wait(0.10)

        self.wait(0.6)
        self.play(FadeOut(safe_vg(title), run_time=0.35))

        self.v_mat = v_mat
        self.v_lbl = v_lbl
        self.o_mat = o_mat
        self.o_lbl = o_lbl

    # ═══════════════════════════════════════════════════════════════════════
    # S6 — Final Output: Contextual Embeddings
    # ═══════════════════════════════════════════════════════════════════════
    def s6_final_output(self):
        title = sec_title(self, "Output: Contextual Embedding Matrix",
                          color=C_GLOW)

        # fade attention and V, keep output
        self.play(
            FadeOut(safe_vg(self.a_mat, self.r_lbl,
                            self.v_mat, self.v_lbl), run_time=0.55),
            self.o_mat.animate(rate_func=smooth, run_time=0.7)
                      .move_to(ORIGIN + DOWN * 0.3),
            self.o_lbl.animate(rate_func=smooth, run_time=0.7)
                      .move_to(UP * (N * CELL / 2 + 0.52)),
        )

        # update output label
        ctx_lbl = matrix_label(
            r"\text{Contextual Embedding Matrix}\ (3 \times d_{\text{model}})",
            C_GLOW, size=30
        )
        ctx_lbl.next_to(self.o_mat, UP, buff=0.28)
        self.play(Transform(self.o_lbl, ctx_lbl, run_time=0.5))

        # glow border on whole matrix
        gb = SurroundingRectangle(self.o_mat[0], color=C_GLOW,
                                   buff=0.10, stroke_width=3.0)
        self.play(Create(gb, run_time=0.45))
        glow_pulse(self, gb, n=2)

        # row-by-row labels: e_(I), e_(love), e_(math)
        tokens = ["I", "love", "math"]
        row_labels_tex = [
            r"e_{(\text{I})}",
            r"e_{(\text{love})}",
            r"e_{(\text{math})}",
        ]
        token_colors = [C_Q, C_GLOW, C_ACCT]

        row_rects  = VGroup()
        row_arrows = VGroup()
        row_lbls   = VGroup()

        for i in range(N):
            row_cells = VGroup(*[self.o_mat[0][i * DIM + c]
                                 for c in range(DIM)])
            rhl = SurroundingRectangle(row_cells, color=token_colors[i],
                                       buff=0.06, stroke_width=2.2)
            arr = Arrow(
                row_cells.get_right() + RIGHT * 0.08,
                row_cells.get_right() + RIGHT * 1.55,
                color=token_colors[i], stroke_width=2.0,
                tip_length=0.18, buff=0
            )
            lbl = MathTex(row_labels_tex[i], color=token_colors[i], font_size=30)
            lbl.next_to(arr, RIGHT, buff=0.10)

            self.play(Create(rhl, run_time=0.28))
            self.play(GrowArrow(arr, run_time=0.30),
                      FadeIn(lbl, shift=LEFT * 0.1, run_time=0.28))
            self.wait(0.15)

            row_rects.add(rhl)
            row_arrows.add(arr)
            row_lbls.add(lbl)

        # final narration note
        note = Text(
            '"Each row is now a context-aware embedding\n'
            ' capturing meaning from the full sentence."',
            font="Fira Code", color=C_DIM, font_size=21,
            line_spacing=1.4
        )
        note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(note, run_time=0.6))
        self.wait(2.0)

        # fade to black
        black = Rectangle(
            width=config.frame_width, height=config.frame_height,
            fill_color=BLACK, fill_opacity=1, stroke_opacity=0
        )
        self.play(FadeIn(black, run_time=0.6))
        self.wait(0.3)