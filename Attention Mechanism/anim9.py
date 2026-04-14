from manim import *
import numpy as np

BG      = "#0f172a"
C_Q     = "#3b82f6"
C_K     = "#22c55e"
C_V     = "#f97316"
C_GLOW  = "#a78bfa"
C_TEXT  = "#e2e8f0"
C_DIM   = "#475569"
C_ACCT  = "#f472b6"

config.background_color = BG
config.pixel_height = 1080
config.pixel_width  = 1920
config.frame_rate   = 60

N   = 3
DIM = 3

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

CELL = 0.60


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


def build_matrix(rows, cols, stroke_color, cell=CELL):
    """
    Stroke-only grid matrix — fill is BG color at low opacity.
    [0] = cells VGroup  [1] = bracket_l  [2] = bracket_r
    """
    cells = VGroup()
    for r in range(rows):
        for c in range(cols):
            sq = Square(
                side_length=cell,
                color=stroke_color,
                fill_color=BG,         
                fill_opacity=1.0,       
                stroke_width=1.8,
                stroke_opacity=0.9,
            )
            sq.move_to([
                c * cell - (cols - 1) * cell / 2,
                -r * cell + (rows - 1) * cell / 2,
                0
            ])
            cells.add(sq)

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



class AttentionFull(Scene):

    def construct(self):
        self.add(make_grid_bg())
        self.s1_qk_intro()
        self.s2_transpose_k()
        self.s3_multiply_qkt()
        self.s4_scale_softmax()
        self.s5_multiply_v()
        self.s6_final_output()


    def s1_qk_intro(self):
        title = sec_title(self, "Query (Q) and Key (K) Matrices")

        q_mat = build_matrix(N, DIM, C_Q)
        q_mat.move_to(LEFT * 4.0 + DOWN * 0.2)
        q_lbl = matrix_label(r"Q", C_Q, size=38)
        q_lbl.next_to(q_mat, UP, buff=0.22)
        q_dim = dim_label(
            r"{\scriptscriptstyle 3 \times d_{\text{model}}",
            C_Q, size=30)
        q_dim.next_to(q_mat, DOWN, buff=0.18)

        k_mat = build_matrix(N, DIM, C_K)
        k_mat.move_to(RIGHT * 4.0 + DOWN * 0.2)
        k_lbl = matrix_label(r"K", C_K, size=38)
        k_lbl.next_to(k_mat, UP, buff=0.22)
        k_dim = dim_label(
            r"{\scriptscriptstyle 3 \times d_{\text{model}}",
            C_K, size=30)
        k_dim.next_to(k_mat, DOWN, buff=0.18)

        self.play(FadeIn(q_mat, scale=0.88, run_time=0.7, rate_func=smooth),
                  FadeIn(q_lbl, shift=DOWN * 0.1, run_time=0.5))
        self.play(FadeIn(q_dim, run_time=0.4))
        qb = SurroundingRectangle(q_mat[0], color=C_GLOW, buff=0.08, stroke_width=2.5)
        self.play(Create(qb, run_time=0.3))
        self.play(FadeOut(qb, run_time=0.3))

        self.play(FadeIn(k_mat, scale=0.88, run_time=0.7, rate_func=smooth),
                  FadeIn(k_lbl, shift=DOWN * 0.1, run_time=0.5))
        self.play(FadeIn(k_dim, run_time=0.4))
        kb = SurroundingRectangle(k_mat[0], color=C_GLOW, buff=0.08, stroke_width=2.5)
        self.play(Create(kb, run_time=0.3))
        self.play(FadeOut(kb, run_time=0.3))

        self.wait(0.8)
        self.play(FadeOut(safe_vg(title, q_dim, k_dim), run_time=0.4))

        self.q_mat = q_mat
        self.q_lbl = q_lbl
        self.k_mat = k_mat
        self.k_lbl = k_lbl

    # S2 — Transpose K
    def s2_transpose_k(self):
        title = sec_title(self, "Transpose K  →  Kᵀ", color=C_K)

        self.play(
            self.q_mat.animate(run_time=0.4).set_opacity(0.20),
            self.q_lbl.animate(run_time=0.4).set_opacity(0.20),
        )

        kt_mat = build_matrix(DIM, N, C_K)
        kt_mat.move_to(RIGHT * 4.0 + DOWN * 0.2)
        kt_lbl = matrix_label(r"K^T", C_K, size=38)
        kt_lbl.next_to(kt_mat, UP, buff=0.22)
        kt_dim = dim_label(
            r"{\scriptscriptstyle d_{\text{model}} \times 3 }",
            C_K, size=30)
        kt_dim.next_to(kt_mat, DOWN, buff=0.18)



  

        hl_r = hl_row(self.k_mat, 0, DIM, C_K)
        self.play(Create(hl_r, run_time=0.3))

        self.play(TransformFromCopy(self.k_mat, kt_mat, run_time=0.9, rate_func=smooth),
                  FadeIn(kt_lbl, run_time=0.5))
        self.play(FadeIn(kt_dim, run_time=0.35))

        hl_c = hl_col(kt_mat, 0, DIM, N, C_K)
        self.play(Create(hl_c, run_time=0.3))
        row_col_note = Text("Rows → Columns", font="Fira Code",
                            color=C_DIM, font_size=22)
        row_col_note.next_to(kt_mat, DOWN, buff=0.52)
        self.play(FadeIn(row_col_note, run_time=0.35))
        self.wait(0.5)

        self.play(FadeOut(safe_vg(title, hl_r, hl_c,
                                   kt_dim, row_col_note,
                                   self.k_mat, self.k_lbl), run_time=0.5))
        self.play(self.q_mat.animate(run_time=0.4).set_opacity(1.0),
                  self.q_lbl.animate(run_time=0.4).set_opacity(1.0))

        self.kt_mat = kt_mat
        self.kt_lbl = kt_lbl

    # S3 — Q × Kᵀ  
    def s3_multiply_qkt(self):
        title = sec_title(self, "Q × Kᵀ ", color=C_GLOW)

        self.play(
            self.q_mat.animate(rate_func=smooth, run_time=0.65)
                      .move_to(LEFT * 5.5 + DOWN * 0.2),
            self.q_lbl.animate(rate_func=smooth, run_time=0.65)
                      .move_to(LEFT * 5.5 + UP * (N * CELL / 2 + 0.5)),
            self.kt_mat.animate(rate_func=smooth, run_time=0.65)
                       .move_to(LEFT * 0.8 + DOWN * 0.2),
            self.kt_lbl.animate(rate_func=smooth, run_time=0.65)
                       .move_to(LEFT * 0.8 + UP * (DIM * CELL / 2 + 0.5)),
        )

        # result shell empty dark cells
        r_mat = build_matrix(N, N, C_GLOW)
        r_mat.move_to(RIGHT * 4.8 + DOWN * 0.2)
        r_lbl = matrix_label(r"QK^T\ \ (3\times 3)", C_GLOW, size=28)
        r_lbl.next_to(r_mat, UP, buff=0.22)
        self.play(FadeIn(r_mat, run_time=0.5), FadeIn(r_lbl, run_time=0.4))

        val_mobs = {}

        for i in range(N):
            for j in range(N):
                
                hl_r = hl_row(self.q_mat, i, DIM, C_Q)
                hl_c = hl_col(self.kt_mat, j, DIM, N, C_K)
                self.play(Create(hl_r, run_time=0.20),
                          Create(hl_c, run_time=0.20))

                terms = " + ".join([f"{Q_DATA[i,k]:.1f}·{K_DATA[j,k]:.1f}"
                                    for k in range(DIM)])
                val = S_RAW[i, j]
                dp_txt = Text(f"{terms} = {val:.2f}",
                              font="Fira Code", color=C_GLOW, font_size=17)
                dp_txt.next_to(r_mat, DOWN, buff=0.32)
                self.play(FadeIn(dp_txt, run_time=0.22))

                target = r_mat[0][i * N + j]
                vtxt = cell_txt(val, C_GLOW, size=17)
                vtxt.move_to(target)
                val_mobs[(i, j)] = vtxt

                intensity = (val - S_RAW.min()) / (S_RAW.max() - S_RAW.min() + 1e-8)
                fill_col = interpolate_color(
                    ManimColor(C_DIM), ManimColor(C_GLOW), intensity)
                self.play(
                    target.animate(run_time=0.22).set_fill(fill_col, opacity=0.45),
                    FadeIn(vtxt, scale=0.7, run_time=0.22),
                )
                self.play(FadeOut(safe_vg(hl_r, hl_c, dp_txt), run_time=0.18))

        self.wait(0.7)
        self.play(FadeOut(title, run_time=0.35))

        self.r_mat    = r_mat
        self.r_lbl    = r_lbl
        self.val_mobs = val_mobs

    def s4_scale_softmax(self):
        title = sec_title(self, "Scale by  √d\u2096  then  Softmax", color=C_ACCT)

        scale_f = MathTex(r"S' = \frac{QK^T}{\sqrt{d_{\text{model}}}}",
                          color=C_TEXT, font_size=32)
        scale_f.next_to(self.r_mat, DOWN, buff=0.38)
        self.play(Write(scale_f, run_time=0.7))
        self.wait(0.3)

        scale_val_mobs = {}
        for i in range(N):
            for j in range(N):
                old  = self.val_mobs[(i, j)]
                newv = S_SCALE[i, j]
                ntxt = cell_txt(newv, C_ACCT, size=15)
                ntxt.move_to(self.r_mat[0][i * N + j])
                scale_val_mobs[(i, j)] = ntxt
                self.play(
                    FadeOut(old, run_time=0.15),
                    self.r_mat[0][i * N + j].animate(run_time=0.18)
                        .set_fill(C_ACCT, opacity=0.22),
                    FadeIn(ntxt, run_time=0.15),
                    run_time=0.18
                )

        self.wait(0.3)

        sm_f = MathTex(
            r"A = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_{\text{model}}}}\right)",
            color=C_TEXT, font_size=30)
        sm_f.next_to(scale_f, DOWN, buff=0.22)
        self.play(Write(sm_f, run_time=0.7))
        self.wait(0.3)

        # softmax row by row with values shown
        attn_val_mobs = {}
        for i in range(N):
            hl = hl_row(self.r_mat, i, N, C_Q)
            self.play(Create(hl, run_time=0.22))
            for j in range(N):
                old  = scale_val_mobs[(i, j)]
                av   = A_DATA[i, j]
                ntxt = cell_txt(av, C_Q, size=15)
                ntxt.move_to(self.r_mat[0][i * N + j])
                attn_val_mobs[(i, j)] = ntxt
                warm = interpolate_color(ManimColor(C_DIM), ManimColor(C_Q), av)
                self.play(
                    FadeOut(old, run_time=0.15),
                    self.r_mat[0][i * N + j].animate(run_time=0.18)
                        .set_fill(warm, opacity=av * 0.65 + 0.10),
                    FadeIn(ntxt, run_time=0.15),
                    run_time=0.18
                )
            self.play(FadeOut(hl, run_time=0.18))

        new_lbl = matrix_label(r"A\ \text{(Attention Weights)}", C_Q, size=26)
        new_lbl.next_to(self.r_mat, UP, buff=0.22)
        self.play(Transform(self.r_lbl, new_lbl, run_time=0.45))

        sum_note = Text("Each row sums to 1.0  ✓",
                        font="Fira Code", color=C_K, font_size=20)
        sum_note.next_to(sm_f, DOWN, buff=0.22)
        self.play(FadeIn(sum_note, run_time=0.35))
        self.wait(0.8)

        all_nums = VGroup(*list(attn_val_mobs.values()))
        self.play(FadeOut(safe_vg(title, scale_f, sm_f, sum_note, all_nums),
                          run_time=0.4))

        self.attn_val_mobs = attn_val_mobs
        self.a_mat = self.r_mat

    def s5_multiply_v(self):
        title = sec_title(self, "Attention Weights  ×  V  →  Output", color=C_V)

        formula = MathTex(
            r"\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
            color=C_TEXT, font_size=32
        )
        formula.next_to(title, DOWN, buff=0.28)
        self.play(FadeIn(formula, run_time=0.5))
        self.wait(0.3)

        self.play(
            self.q_mat.animate(run_time=0.4).set_opacity(0),
            self.q_lbl.animate(run_time=0.4).set_opacity(0),
            self.kt_mat.animate(run_time=0.4).set_opacity(0),
            self.kt_lbl.animate(run_time=0.4).set_opacity(0),
            self.a_mat.animate(rate_func=smooth, run_time=0.65)
                    .move_to(LEFT * 4.8 + DOWN * 0.5),
            self.r_lbl.animate(rate_func=smooth, run_time=0.65)
                    .move_to(LEFT * 4.8 + UP * (N * CELL / 2 + 0.10)),
        )

        v_mat = build_matrix(N, DIM, C_V)
        v_mat.move_to(LEFT * 0.5 + DOWN * 0.5)
        v_lbl = matrix_label(r"V\ \ (3 \times d_{\text{model}})", C_V, size=28)
        v_lbl.next_to(v_mat, UP, buff=0.22)
        self.play(FadeIn(v_mat, scale=0.88, run_time=0.6, rate_func=smooth),
                FadeIn(v_lbl, run_time=0.45))

        o_mat = build_matrix(N, DIM, C_V)
        o_mat.move_to(RIGHT * 4.6 + DOWN * 0.5)
        o_lbl = matrix_label(r"O\ \ (3 \times d_{\text{model}})", C_V, size=28)
        o_lbl.next_to(o_mat, UP, buff=0.22)
        self.play(FadeIn(o_mat, run_time=0.5), FadeIn(o_lbl, run_time=0.4))

        for i in range(N):
            hl_a = hl_row(self.a_mat, i, N, C_Q)
            self.play(Create(hl_a, run_time=0.22))

            v_hls = []
            for j in range(N):
                wt = A_DATA[i, j]
                hv = hl_row(v_mat, j, DIM, C_V)
                v_hls.append(hv)
                self.play(Create(hv, run_time=0.18))
                vcells = VGroup(*[v_mat[0][j * DIM + c] for c in range(DIM)])
                self.play(vcells.animate(rate_func=there_and_back, run_time=0.22)
                                .stretch(1 + wt * 0.28, 0))

            o_row = VGroup(*[o_mat[0][i * DIM + c] for c in range(DIM)])
            arr = Arrow(v_mat.get_right() + RIGHT * 0.08,
                        o_row.get_left() + LEFT * 0.08,
                        color=C_V, stroke_width=2.2, tip_length=0.18, buff=0.04)
            self.play(GrowArrow(arr, run_time=0.32))

            for c in range(DIM):
                self.play(o_mat[0][i * DIM + c].animate(run_time=0.18)
                        .set_fill(C_V, opacity=0.30), run_time=0.18)

            self.play(FadeOut(safe_vg(hl_a, arr, *v_hls), run_time=0.20))
            self.wait(0.10)

        self.wait(0.6)

        self.play(FadeOut(safe_vg(title, formula), run_time=0.35))

        self.v_mat = v_mat
        self.v_lbl = v_lbl
        self.o_mat = o_mat
        self.o_lbl = o_lbl

    def s6_final_output(self):
        title = sec_title(self, "Output: Contextual Embedding Matrix", color=C_GLOW)

        self.play(
            FadeOut(safe_vg(self.a_mat, self.r_lbl,
                            self.v_mat, self.v_lbl), run_time=0.55),
            self.o_mat.animate(rate_func=smooth, run_time=0.7)
                      .move_to(ORIGIN + DOWN * 0.3),
            self.o_lbl.animate(rate_func=smooth, run_time=0.7)
                      .move_to(UP * (N * CELL / 2 + 0.52)),
        )

        ctx_lbl = matrix_label(
            r"\text{Contextual Embedding Matrix}\ (3 \times d_{\text{model}})",
            C_GLOW, size=30)
        ctx_lbl.next_to(self.o_mat, UP, buff=0.28)
        self.play(Transform(self.o_lbl, ctx_lbl, run_time=0.5))

        for idx in range(N * DIM):
            self.o_mat[0][idx].set_fill(BG, opacity=1.0)
            self.o_mat[0][idx].set_stroke(C_V, width=1.8, opacity=0.9)

        # glow border
        gb = SurroundingRectangle(self.o_mat[0], color=C_GLOW,
                                   buff=0.10, stroke_width=3.0)
        self.play(Create(gb, run_time=0.45))
        glow_pulse(self, gb, n=2)

        # row labels
        row_labels_tex = [r"E_{(\text{I})}", r"E_{(\text{love}})", r"E_{(\text{math})}"]
        token_colors   = [C_Q, C_GLOW, C_ACCT]

        row_rects  = VGroup()
        row_arrows = VGroup()
        row_lbls   = VGroup()

        for i in range(N):
            row_cells = VGroup(*[self.o_mat[0][i * DIM + c] for c in range(DIM)])
            rhl = SurroundingRectangle(row_cells, color=token_colors[i],
                                       buff=0.06, stroke_width=2.2)
            arr = Arrow(row_cells.get_right() + RIGHT * 0.08,
                        row_cells.get_right() + RIGHT * 1.55,
                        color=token_colors[i], stroke_width=2.0,
                        tip_length=0.18, buff=0)
            lbl = MathTex(row_labels_tex[i], color=token_colors[i], font_size=30)
            lbl.next_to(arr, RIGHT, buff=0.10)

            self.play(Create(rhl, run_time=0.28))
            self.play(GrowArrow(arr, run_time=0.30),
                      FadeIn(lbl, shift=LEFT * 0.1, run_time=0.28))
            self.wait(0.15)

            row_rects.add(rhl)
            row_arrows.add(arr)
            row_lbls.add(lbl)

        note = Text(
            '"Each row is now a context aware embedding'
            ' capturing meaning from the full sentence."',
            font="Fira Code", color=C_DIM, font_size=21, line_spacing=1.4)
        note.to_edge(DOWN, buff=0.8)
        self.play(FadeIn(note, run_time=0.6))
        self.wait(2.0)

        black = Rectangle(width=config.frame_width, height=config.frame_height,
                           fill_color=BLACK, fill_opacity=1, stroke_opacity=0)
        self.play(FadeIn(black, run_time=0.6))
        self.wait(0.3)