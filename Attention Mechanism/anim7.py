

from manim import *
import numpy as np

BG       = "#0f172a"
C_QUERY  = "#3b82f6"
C_KEY    = "#22c55e"
C_VALUE  = "#f97316"
C_GLOW   = "#a78bfa"
C_TEXT   = "#e2e8f0"
C_DIM    = "#475569"
C_ACCENT = "#f472b6"

config.background_color = BG
config.pixel_height = 1080
config.pixel_width  = 1920
config.frame_rate   = 60

N   = 3    # tokens
DIM = 3    # visual dimension (d_k)

Q_DATA = np.array([[1.0, 0.5, 0.2],
                   [0.3, 1.2, 0.8],
                   [0.9, 0.1, 1.1]])

K_DATA = np.array([[0.8, 0.6, 0.3],
                   [0.2, 1.0, 0.5],
                   [0.7, 0.4, 0.9]])

V_DATA = np.array([[1.0, 0.2, 0.5],
                   [0.4, 0.8, 0.1],
                   [0.6, 0.3, 0.9]])

S_RAW   = Q_DATA @ K_DATA.T
S_SCALE = S_RAW / np.sqrt(DIM)

def softmax_row(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

A_DATA = np.array([softmax_row(row) for row in S_SCALE])
O_DATA = A_DATA @ V_DATA

CELL = 0.62   


def safe_vgroup(*mobs):
    return VGroup(*[m for m in mobs if m is not None])


def make_grid_bg(opacity=0.04):
    lines = VGroup()
    for i in range(29):
        x = -config.frame_width / 2 + i * config.frame_width / 28
        lines.add(Line([x, -config.frame_height / 2, 0],
                       [x,  config.frame_height / 2, 0],
                       stroke_color="#1e293b", stroke_width=0.5,
                       stroke_opacity=opacity))
    for j in range(17):
        y = -config.frame_height / 2 + j * config.frame_height / 16
        lines.add(Line([-config.frame_width / 2, y, 0],
                       [ config.frame_width / 2, y, 0],
                       stroke_color="#1e293b", stroke_width=0.5,
                       stroke_opacity=opacity))
    return lines


def make_mat_mob(data, color, cell=CELL, label="",
                 fill_op=0.12, show_values=False):

    rows, cols = data.shape
    cells = VGroup()
    for r in range(rows):
        for c in range(cols):
            sq = Square(side_length=cell,
                        color=color,
                        fill_color=color,
                        fill_opacity=fill_op,
                        stroke_width=1.4)
            sq.move_to([c * cell - (cols - 1) * cell / 2,
                        -r * cell + (rows - 1) * cell / 2, 0])
            cells.add(sq)

    bl = MathTex(r"\big[", color=color, font_size=72)
    br = MathTex(r"\big]", color=color, font_size=72)
    bl.next_to(cells, LEFT,  buff=0.06)
    br.next_to(cells, RIGHT, buff=0.06)
    group = VGroup(cells, bl, br)

    if label:
        lbl = MathTex(label, color=color, font_size=36)
        lbl.next_to(group, UP, buff=0.22)
        group.add(lbl)

    if show_values:
        nums = VGroup()
        for r in range(rows):
            for c in range(cols):
                val = data[r, c]
                n = Text(f"{val:.2f}", font="Fira Code",
                         color=color, font_size=14)
                n.move_to(cells[r * cols + c])
                nums.add(n)
        group.add(nums)   # index 4 if label, else index 3

    return group


def glow_border(mat, color=C_GLOW, buff=0.08, width=2.8):
    return SurroundingRectangle(mat[0], color=color,
                                 buff=buff, stroke_width=width)


def highlight_row(mat, row_idx, cols, color, cell=CELL, buff=0.05):
    cells = VGroup(*[mat[0][row_idx * cols + c] for c in range(cols)])
    return SurroundingRectangle(cells, color=color,
                                 buff=buff, stroke_width=2.2)


def highlight_col(mat, col_idx, rows, cols, color, buff=0.05):
    cells = VGroup(*[mat[0][r * cols + col_idx] for r in range(rows)])
    return SurroundingRectangle(cells, color=color,
                                 buff=buff, stroke_width=2.2)


def section_label(scene, txt, color=C_TEXT, font_size=36):
    t = Text(txt, font="Fira Code Bold", color=color,
             font_size=font_size, weight=BOLD)
    t.to_edge(UP, buff=0.30)
    scene.play(FadeIn(t, shift=DOWN * 0.18, run_time=0.50))
    return t


def glow_pulse(scene, mob, n=1, color=C_GLOW):
    for _ in range(n):
        scene.play(mob.animate(rate_func=there_and_back,
                               run_time=0.36).scale(1.05))



class DotProductAttention(Scene):

    def construct(self):
        self.add(make_grid_bg())
        self.scene1_setup()
        self.scene2_transpose_k()
        self.scene3_multiply_qkt()
        self.scene4_name_score()
        self.scene5_scale()
        self.scene6_softmax()
        self.scene7_multiply_v()
        self.scene8_final()

    def scene1_setup(self):
        title = section_label(self, "Self-Attention: Q, K, V Matrices")

        q_mat = make_mat_mob(Q_DATA, C_QUERY, label=r"Q\ \ \text{(Query)}")
        k_mat = make_mat_mob(K_DATA, C_KEY,   label=r"K\ \ \text{(Key)}")
        v_mat = make_mat_mob(V_DATA, C_VALUE,  label=r"V\ \ \text{(Value)}")

        for mat, pos in zip([q_mat, k_mat, v_mat],
                            [LEFT * 5.2, ORIGIN, RIGHT * 5.2]):
            mat.move_to(pos + DOWN * 0.3)

        dim_note = Text(f"Each matrix: {N} tokens × {DIM} dimensions",
                        font="Fira Code", color=C_DIM, font_size=22)
        dim_note.to_edge(DOWN, buff=0.5)

        for mat, color in zip([q_mat, k_mat, v_mat],
                               [C_QUERY, C_KEY, C_VALUE]):
            self.play(FadeIn(mat, scale=0.90, run_time=0.55, rate_func=smooth))
            gb = glow_border(mat, color)
            self.play(Create(gb, run_time=0.25))
            self.play(FadeOut(gb, run_time=0.25))

        self.play(FadeIn(dim_note, run_time=0.45))
        self.wait(1.0)

        self.play(FadeOut(safe_vgroup(title, dim_note), run_time=0.4))
        self.q_mat = q_mat
        self.k_mat = k_mat
        self.v_mat = v_mat

    def scene2_transpose_k(self):
        title = section_label(self, "Step 1: Transpose K  →  Kᵀ", color=C_KEY)

        self.play(
            self.q_mat.animate(run_time=0.4).set_opacity(0.22),
            self.v_mat.animate(run_time=0.4).set_opacity(0.22),
        )

        k_copy = self.k_mat.copy()
        kt_target = make_mat_mob(K_DATA.T, C_KEY, label=r"K^T")
        kt_target.move_to(RIGHT * 2.8 + DOWN * 0.3)

        arrow_lbl = Text("Transpose", font="Fira Code",
                         color=C_GLOW, font_size=26).next_to(k_copy,DOWN+1)
        arr = Arrow(self.k_mat.get_right() + RIGHT * 0.15,
                    kt_target.get_left()   + LEFT  * 0.15,
                    color=C_GLOW, stroke_width=2.5,
                    tip_length=0.2, buff=0.05)
        arrow_lbl.next_to(arr, UP, buff=0.12)

        row_note = Text("Rows → Columns", font="Fira Code",
                        color=C_DIM, font_size=22)
        row_note.next_to(kt_target, DOWN, buff=0.28)

        self.play(GrowArrow(arr, run_time=0.45),
                  FadeIn(arrow_lbl, run_time=0.35))
        self.play(TransformFromCopy(self.k_mat, kt_target, run_time=0.85,
                                     rate_func=smooth))
        self.play(FadeIn(row_note, run_time=0.35))

        hl_row = highlight_row(self.k_mat, 0, DIM, C_KEY)
        hl_col = highlight_col(kt_target, 0, DIM, DIM, C_KEY)
        self.play(Create(hl_row, run_time=0.3))
        self.play(TransformFromCopy(hl_row, hl_col, run_time=0.5))
        self.wait(0.3)
        self.play(FadeOut(safe_vgroup(hl_row, hl_col), run_time=0.3))
        self.wait(0.5)

        self.play(FadeOut(safe_vgroup(title, arr, arrow_lbl,
                                      row_note), run_time=0.4))
        self.play(
            self.q_mat.animate(run_time=0.4).set_opacity(1.0),
            self.v_mat.animate(run_time=0.4).set_opacity(1.0),
        )
        self.kt_mat = kt_target

    def scene3_multiply_qkt(self):
        title = section_label(self, "Step 2: Q × Kᵀ  — Dot Products", color=C_GLOW)

        self.play(
            self.q_mat.animate(rate_func=smooth, run_time=0.7)
                      .move_to(LEFT * 5.8 + DOWN * 0.2),
            self.k_mat.animate(rate_func=smooth, run_time=0.5)
                      .set_opacity(0),
            self.v_mat.animate(rate_func=smooth, run_time=0.5)
                      .set_opacity(0.18),
            self.kt_mat.animate(rate_func=smooth, run_time=0.7)
                       .move_to(LEFT * 1.6 + DOWN * 0.2),
        )

        s_shell = make_mat_mob(np.zeros((N, N)), C_GLOW, label=r"S = QK^T")
        s_shell.move_to(RIGHT * 4.2 + DOWN * 0.2)
        self.play(FadeIn(s_shell, run_time=0.5))

        for i in range(N):
            for j in range(N):
                hl_row = highlight_row(self.q_mat, i, DIM, C_QUERY)
                hl_col = highlight_col(self.kt_mat, j, N, DIM, C_KEY)
                target_cell = s_shell[0][i * N + j]

                self.play(Create(hl_row, run_time=0.22),
                          Create(hl_col, run_time=0.22))

                terms = " + ".join([
                    f"q{i+1}{k+1}·k{j+1}{k+1}"
                    for k in range(DIM)
                ])
                val = S_RAW[i, j]
                formula = Text(f"{terms} = {val:.2f}",
                               font="Fira Code", color=C_GLOW, font_size=18)
                formula.next_to(s_shell, DOWN, buff=0.30)
                self.play(FadeIn(formula, run_time=0.28))

                val_txt = Text(f"{val:.1f}", font="Fira Code",
                               color=C_GLOW, font_size=18)
                val_txt.move_to(target_cell)

                self.play(
                    target_cell.animate(run_time=0.3, rate_func=smooth)
                               .set_fill(C_GLOW, opacity=0.30),
                    FadeIn(val_txt, scale=0.7, run_time=0.3),
                )
                self.play(FadeOut(safe_vgroup(hl_row, hl_col,
                                              formula), run_time=0.2))

        self.wait(0.8)
        self.play(FadeOut(title, run_time=0.4))
        self.s_mat = s_shell

    def scene4_name_score(self):
        title = section_label(self, "Score Matrix  S = QKᵀ", color=C_GLOW)

        labels = VGroup()
        for i in range(N):
            for j in range(N):
                cell = self.s_mat[0][i * N + j]
                lbl = MathTex(f"S_{{{i+1}{j+1}}}",
                              color=C_GLOW, font_size=22)
                lbl.move_to(cell.get_top() + UP * 0.22)
                labels.add(lbl)

        self.play(
            LaggedStart(*[
                FadeIn(l, scale=0.6, rate_func=there_and_back_with_pause,
                       run_time=0.32)
                for l in labels
            ], lag_ratio=0.08)
        )
        self.wait(0.5)

        hm_note = Text("Score ↑  →  Higher attention",
                       font="Fira Code", color=C_DIM, font_size=22)
        hm_note.next_to(self.s_mat, DOWN, buff=0.38)
        self.play(FadeIn(hm_note, run_time=0.4))

        max_val = S_RAW.max()
        for i in range(N):
            for j in range(N):
                cell = self.s_mat[0][i * N + j]
                intensity = S_RAW[i, j] / max_val
                warm = interpolate_color(ManimColor(C_DIM),
                                         ManimColor(C_ACCENT), intensity)
                self.play(cell.animate(run_time=0.18, rate_func=smooth)
                              .set_fill(warm, opacity=0.5 * intensity + 0.1),
                          run_time=0.18)

        self.wait(0.6)
        self.play(FadeOut(safe_vgroup(title, labels, hm_note), run_time=0.4))


    def scene5_scale(self):
        title = section_label(self, f"Step 3: Scale by  1/√d  (d={DIM})",
                              color=C_ACCENT)

        sqrt_d = np.sqrt(DIM)
        formula = MathTex(
            r"S' = \frac{S}{\sqrt{d_k}} = \frac{S}{" + f"{sqrt_d:.2f}" + r"}",
            color=C_TEXT, font_size=36
        )
        formula.next_to(self.s_mat, DOWN, buff=0.4)
        self.play(Write(formula, run_time=0.8))
        self.wait(0.3)

        for i in range(N):
            for j in range(N):
                cell = self.s_mat[0][i * N + j]
                new_val = S_SCALE[i, j]
                new_txt = Text(f"{new_val:.2f}", font="Fira Code",
                               color=C_KEY, font_size=16)
                new_txt.move_to(cell)
                self.play(
                    cell.animate(run_time=0.22, rate_func=smooth)
                        .set_fill(C_KEY, opacity=0.22),
                    FadeIn(new_txt, scale=0.8, run_time=0.22),
                    run_time=0.22
                )

        stable_note = Text("Values stabilized — variance ≈ 1",
                           font="Fira Code", color=C_KEY, font_size=22)
        stable_note.next_to(formula, DOWN, buff=0.28)
        self.play(FadeIn(stable_note, run_time=0.45))
        self.wait(0.8)

        self.play(FadeOut(safe_vgroup(title, formula, stable_note),
                          run_time=0.4))


    def scene6_softmax(self):
        title = section_label(self, "Step 4: Softmax → Attention Weights  A",
                              color=C_QUERY)

        softmax_formula = MathTex(
            r"\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}",
            color=C_TEXT, font_size=30
        )
        softmax_formula.to_edge(DOWN, buff=0.5)
        self.play(Write(softmax_formula, run_time=0.7))

        for i in range(N):
            #  highlight row
            hl = highlight_row(self.s_mat, i, N, C_QUERY)
            self.play(Create(hl, run_time=0.28))

            row_vals = S_SCALE[i]
            exp_vals = np.exp(row_vals - row_vals.max())
            probs    = exp_vals / exp_vals.sum()

            exp_str = "  →  [" + ",  ".join([f"e^{v:.2f}" for v in row_vals]) + "]"
            exp_note = Text(exp_str, font="Fira Code",
                            color=C_GLOW, font_size=18)
            exp_note.next_to(self.s_mat, LEFT, buff=0.3)
            exp_note.set_y(self.s_mat[0][i * N].get_center()[1])
            self.play(FadeIn(exp_note, run_time=0.3))

            for j in range(N):
                cell = self.s_mat[0][i * N + j]
                p = float(probs[j])
                warm = interpolate_color(ManimColor(C_DIM),
                                         ManimColor(C_QUERY), p)
                prob_txt = Text(f"{A_DATA[i,j]:.2f}", font="Fira Code",
                                color=C_QUERY, font_size=15)
                prob_txt.move_to(cell)
                self.play(
                    cell.animate(run_time=0.22).set_fill(warm, opacity=p * 0.7 + 0.1),
                    FadeIn(prob_txt, run_time=0.22),
                    run_time=0.22
                )

            self.play(FadeOut(safe_vgroup(hl, exp_note), run_time=0.2))
            self.wait(0.15)

        a_label = MathTex(r"A", color=C_QUERY, font_size=36)
        a_label.next_to(self.s_mat[0], UP, buff=0.25)
        if len(self.s_mat) > 3:   
            self.play(Transform(self.s_mat[3], a_label, run_time=0.45))
        else:
            self.play(FadeIn(a_label, run_time=0.4))

        sums_note = Text("Each row sums to 1.0  ✓",
                         font="Fira Code", color=C_KEY, font_size=22)
        sums_note.next_to(self.s_mat, RIGHT, buff=0.35)
        self.play(FadeIn(sums_note, run_time=0.45))
        self.wait(0.8)

        self.play(FadeOut(safe_vgroup(title, softmax_formula,
                                      sums_note), run_time=0.4))
        self.a_mat = self.s_mat

    def scene7_multiply_v(self):
        title = section_label(self, "Step 5: A · V  →  Output O", color=C_VALUE)

        self.play(
            self.v_mat.animate(rate_func=smooth, run_time=0.6)
                      .set_opacity(1.0)
                      .move_to(RIGHT * 2.2 + DOWN * 0.2),
            self.a_mat.animate(rate_func=smooth, run_time=0.6)
                      .move_to(LEFT * 4.0 + DOWN * 0.2),
            self.q_mat.animate(run_time=0.3).set_opacity(0),
            self.kt_mat.animate(run_time=0.3).set_opacity(0),
            self.k_mat.animate(run_time=0.3).set_opacity(0),
        )

        o_shell = make_mat_mob(np.zeros((N, DIM)), C_VALUE, label=r"O")
        o_shell.move_to(RIGHT * 6.5 + DOWN * 0.2)
        self.play(FadeIn(o_shell, run_time=0.5))

        for i in range(N):  
            hl_a = highlight_row(self.a_mat, i, N, C_QUERY)
            self.play(Create(hl_a, run_time=0.28))

            hl_vs = []
            for j in range(N):
                hl_v = highlight_row(self.v_mat, j, DIM, C_VALUE)
                wt = A_DATA[i, j]
                wt_lbl = Text(f"×{wt:.2f}", font="Fira Code",
                              color=C_VALUE, font_size=16)
                wt_lbl.next_to(hl_v, LEFT, buff=0.12)
                hl_vs.append(safe_vgroup(hl_v, wt_lbl))
                self.play(Create(hl_v, run_time=0.2),
                          FadeIn(wt_lbl, run_time=0.2))

                v_row_cells = VGroup(*[self.v_mat[0][j * DIM + c]
                                       for c in range(DIM)])
                self.play(
                    v_row_cells.animate(rate_func=there_and_back,
                                        run_time=0.25)
                               .stretch(1.0 + wt * 0.3, 0)
                )

            o_row_cells = VGroup(*[o_shell[0][i * DIM + c] for c in range(DIM)])
            arr_out = Arrow(self.v_mat.get_right() + RIGHT * 0.1,
                            o_row_cells.get_left()  + LEFT  * 0.1,
                            color=C_VALUE, stroke_width=2.2,
                            tip_length=0.18, buff=0.05)
            self.play(GrowArrow(arr_out, run_time=0.35))

            for c in range(DIM):
                val = O_DATA[i, c]
                cell = o_shell[0][i * DIM + c]
                v_txt = Text(f"{val:.2f}", font="Fira Code",
                             color=C_VALUE, font_size=14)
                v_txt.move_to(cell)
                self.play(
                    cell.animate(run_time=0.2).set_fill(C_VALUE, opacity=0.28),
                    FadeIn(v_txt, run_time=0.2),
                    run_time=0.2
                )

            # label O_i
            oi_lbl = MathTex(f"O_{i+1}", color=C_VALUE, font_size=24)
            oi_lbl.next_to(o_row_cells, RIGHT, buff=0.18)
            self.play(FadeIn(oi_lbl, run_time=0.28))

            cleanup = safe_vgroup(hl_a, arr_out, oi_lbl, *hl_vs)
            self.play(FadeOut(cleanup, run_time=0.22))
            self.wait(0.12)

        self.wait(0.6)
        self.play(FadeOut(title, run_time=0.4))
        self.o_mat = o_shell

    def scene8_final(self):
        # fade everything out
        self.play(FadeOut(safe_vgroup(self.a_mat, self.v_mat,
                                      self.o_mat), run_time=0.6))

        title = section_label(self, "Attention(Q, K, V) — Full Pipeline",
                              color=C_GLOW)

        steps = [
            ("Q, K", C_QUERY),
            ("QKᵀ",  C_GLOW),
            ("÷ √d", C_ACCENT),
            ("Softmax", C_QUERY),
            ("× V",  C_VALUE),
            ("O",    C_VALUE),
        ]

        boxes  = VGroup()
        arrows = VGroup()
        n = len(steps)
        spacing = 11.8 / (n - 1)

        for idx, (txt, col) in enumerate(steps):
            x = -5.9 + idx * spacing
            rect = RoundedRectangle(corner_radius=0.18,
                                    width=1.7, height=0.75,
                                    color=col, fill_opacity=0.12,
                                    stroke_width=2.0)
            rect.move_to([x, 0, 0])
            lbl = Text(txt, font="Fira Code Bold", color=col,
                       font_size=24, weight=BOLD)
            lbl.move_to(rect)
            boxes.add(VGroup(rect, lbl))

        for i in range(n - 1):
            a = Arrow(boxes[i].get_right()   + RIGHT * 0.04,
                      boxes[i+1].get_left()  + LEFT  * 0.04,
                      color=C_DIM, stroke_width=1.8,
                      tip_length=0.16, buff=0.02)
            arrows.add(a)

        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.15, run_time=0.38)
                          for b in boxes], lag_ratio=0.14)
        )
        self.play(
            LaggedStart(*[GrowArrow(a, run_time=0.25) for a in arrows],
                        lag_ratio=0.12)
        )

        glow_pulse(self, boxes[-1], n=2, color=C_VALUE)
        self.wait(0.4)

        # final formula
        final = MathTex(
            r"\text{Attention}(Q,K,V) ="
            r"\text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
            color=C_TEXT, font_size=44
        )
        final.move_to(DOWN * 2.0)

        glow_layers = VGroup()
        for i in (3, 2, 1):
            gl = final.copy().set_color(C_GLOW)
            gl.set_opacity(0.14 * i / 3).scale(1 + 0.011 * i)
            glow_layers.add(gl)

        self.play(
            LaggedStart(
                FadeIn(glow_layers, run_time=0.8),
                Write(final, run_time=1.2),
                lag_ratio=0.2
            )
        )
        self.play(
            VGroup(glow_layers, final).animate(rate_func=smooth,
                                               run_time=1.0).scale(1.06)
        )
        self.wait(2.0)

        black = Rectangle(width=config.frame_width, height=config.frame_height,
                           fill_color=BLACK, fill_opacity=1, stroke_opacity=0)
        self.play(FadeIn(black, run_time=0.5))
        self.wait(0.3)