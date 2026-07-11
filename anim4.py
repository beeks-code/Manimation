"""
Cinematic Manim Animation:
How X is transformed into Q, K, V using learned weight matrices
Inspired by 3Blue1Brown + Reducible
"""

from manim import *
import numpy as np

# ─── PALETTE ────────────────────────────────────────────────────────────────
BG       = "#0f172a"
C_QUERY  = "#3b82f6"
C_KEY    = "#22c55e"
C_VALUE  = "#f97316"
C_GLOW   = "#a78bfa"
C_TEXT   = "#e2e8f0"
C_DIM    = "#475569"
C_ACCENT = "#f472b6"
TOKEN_COLORS = [C_QUERY, C_GLOW, C_ACCENT]

config.background_color = BG
config.pixel_height = 1080
config.pixel_width  = 1920
config.frame_rate   = 60

# ─── LAYOUT CONSTANTS ───────────────────────────────────────────────────────
# 3-column layout:  X at col-0, W at col-1, result at col-2
COL_X = LEFT  * 5.5
COL_W = LEFT  * 1.2
COL_R = RIGHT * 3.2

# 3-row layout within each column
ROW_Q = UP    * 1.8
ROW_K = ORIGIN
ROW_V = DOWN  * 1.8

CELL  = 0.52   # cell size for all matrices
ROWS_X = 3     # embedding rows (words)
COLS_X = 4     # embedding dim (visual)
ROWS_W = 4     # W rows = embedding dim
COLS_W = 3     # W cols = d_k (visual)

# ─── HELPERS ────────────────────────────────────────────────────────────────

def safe_vgroup(*mobs):
    return VGroup(*[m for m in mobs if m is not None])


def glow_rect(rect, color, layers=3):
    g = VGroup()
    for i in range(layers, 0, -1):
        r = rect.copy()
        r.set_stroke(color, opacity=0.22 * i / layers,
                     width=rect.get_stroke_width() * (1 + 0.5 * i))
        r.set_fill(opacity=0)
        g.add(r)
    g.add(rect)
    return g


def make_matrix(rows, cols, color, cell=CELL, label="",
                row_tints=None, fill_opacity=0.10):
    """
    Returns a VGroup: [cells_vgroup, bracket_l, bracket_r, label_mob?]
    cells_vgroup is indexed [row * cols + col]
    """
    cells = VGroup()
    for r in range(rows):
        for c in range(cols):
            tint = row_tints[r] if row_tints and r < len(row_tints) else color
            sq = Square(side_length=cell,
                        color=tint,
                        fill_color=tint,
                        fill_opacity=fill_opacity,
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
        lbl = MathTex(label, color=color, font_size=34)
        lbl.next_to(group, RIGHT, buff=0.12)
        group.add(lbl)      # index 3

    return group


def highlight_matrix(scene, mat, color, scale=1.06, run_time=0.4):
    """Flash a glow rectangle around matrix, return it."""
    box = SurroundingRectangle(mat[0], color=color, buff=0.06,
                                stroke_width=2.8)
    scene.play(Create(box, run_time=run_time / 2))
    scene.play(mat.animate(rate_func=there_and_back,
                            run_time=run_time).scale(scale))
    return box


def make_grid_bg(opacity=0.05):
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


# ─── SCENE ──────────────────────────────────────────────────────────────────

class QKVProjection(Scene):

    def construct(self):
        # persistent background grid (very faint)
        grid = make_grid_bg()
        self.add(grid)

        self.scene1_setup_x()
        self.scene2_weight_matrices()
        self.scene3_matrix_multiplication()
        self.scene4_clean_transition()
        self.scene5_column_alignment()
        self.scene6_q_rows()
        self.scene7_k_rows()
        self.scene8_v_rows()

    # ════════════════════════════════════════════════════════════════════════
    # SCENE 1 — X Matrix
        # ════════════════════════════════════════════════════════════════════════
    def scene1_setup_x(self):
        # title
        title = Text("Embedding Matrix  X", font="Fira Code Bold",
                    color=C_TEXT, font_size=40, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title, shift=DOWN * 0.2, run_time=0.6))

        # build X
        x_mat = make_matrix(ROWS_X, COLS_X, C_QUERY,
                            row_tints=TOKEN_COLORS,
                            label=r"X")
        x_mat.move_to(COL_X + UP * 0.2)

        # word labels LEFT of X  (e.g. "I", "like", "Maths")
        word_labels = VGroup(*[
            Text(w, font="Fira Code", color=c, font_size=28)
            for w, c in zip(['"I"', '"like"', '"Maths"'], TOKEN_COLORS)
        ])
        for i, lbl in enumerate(word_labels):
            cell_y = x_mat[0][i * COLS_X].get_center()[1]
            lbl.move_to([x_mat.get_left()[0] - 0.9, cell_y, 0])

        # e1, e2, e3 labels RIGHT of each row
        e_labels = VGroup(*[
            MathTex(fr"e_{i+1}", color=TOKEN_COLORS[i], font_size=30)
            for i in range(ROWS_X)
        ])
        for i, lbl in enumerate(e_labels):
            cell_y = x_mat[0][i * COLS_X].get_center()[1]
            lbl.move_to([x_mat.get_right()[0] + 0.55, cell_y, 0])

        # annotation
        annot = Text("3 tokens  ×  d_model dimensions",
                    font="Fira Code", color=C_DIM, font_size=22)
        annot.next_to(x_mat, DOWN, buff=0.35)
        annot.to_edge(LEFT, buff=0.2)

           # ── animate row highlights → arrows → e labels ───────────────────────
        self.play(FadeIn(x_mat, scale=0.92, run_time=0.8, rate_func=smooth))

        # word labels stagger in from left
        self.play(
            LaggedStart(*[FadeIn(l, shift=RIGHT * 0.15, run_time=0.35)
                        for l in word_labels], lag_ratio=0.2)
        )

        # row-by-row: rectangle → arrow → e label
        row_rects  = VGroup()
        row_arrows = VGroup()

        for i, e_lbl in enumerate(e_labels):
            row_cells = VGroup(*[x_mat[0][i * COLS_X + c] for c in range(COLS_X)])

            # surround the full row
            hl = SurroundingRectangle(row_cells, color=TOKEN_COLORS[i],
                                    buff=0.06, stroke_width=2.2)

            # arrow from right edge of row → e label
            arr = Arrow(row_cells.get_right() + RIGHT * 0.08,
                        e_lbl.get_left()      + LEFT  * 0.08,
                        color=TOKEN_COLORS[i],
                        stroke_width=2.0,
                        tip_length=0.16,
                        buff=0)

            self.play(Create(hl,  run_time=0.28))
            self.play(GrowArrow(arr, run_time=0.30),
                    FadeIn(e_lbl, shift=LEFT * 0.1, run_time=0.28))
            self.wait(0.15)

            row_rects.add(hl)
            row_arrows.add(arr)

        # annotation
        annot = Text("3 tokens  ×  d_model dimensions",
                    font="Fira Code", color=C_DIM, font_size=22)
        annot.next_to(x_mat, DOWN, buff=0.35)
        annot.to_edge(LEFT, buff=0.2)
        self.play(FadeIn(annot, run_time=0.5))
        self.wait(0.8)

        # fade out everything except the matrix
        self.play(
            FadeOut(title,       run_time=0.4),
            FadeOut(annot,       run_time=0.4),
            FadeOut(row_rects,   run_time=0.4),
            FadeOut(row_arrows,  run_time=0.4),
            FadeOut(e_labels,    run_time=0.4),
            FadeOut(word_labels, run_time=0.4),
        )

        self.x_mat       = x_mat
        self.word_labels = word_labels   # already faded, kept if needed
        self.e_labels    = e_labels      # already faded, kept if needed


    # ════════════════════════════════════════════════════════════════════════
    # SCENE 2 — Weight Matrices
    # ════════════════════════════════════════════════════════════════════════
    def scene2_weight_matrices(self):
        title = Text("Learned Weight Matrices", font="Fira Code Bold",
                     color=C_TEXT, font_size=40, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.play(FadeIn(title, shift=DOWN * 0.18, run_time=0.55))
        wq = make_matrix(ROWS_W, COLS_W, C_QUERY,
    label=r"W_Q \, {\scriptscriptstyle d_{\text{model}} \times d_{\text{model}}}")

        wk = make_matrix(ROWS_W, COLS_W, C_KEY,
    label=r"W_K \, {\scriptscriptstyle d_{\text{model}} \times d_{\text{model}}}")

        wv = make_matrix(ROWS_W, COLS_W, C_VALUE,
    label=r"W_V \, {\scriptscriptstyle d_{\text{model}} \times d_{\text{model}}}")

        wq.move_to(COL_W + ROW_Q)
        wk.move_to(COL_W + ROW_K)
        wv.move_to(COL_W + ROW_V)

        # slide each in from right with glow
        for wmat, color in zip([wq, wk, wv], [C_QUERY, C_KEY, C_VALUE]):
            start_pos = wmat.get_center() + RIGHT * 2.5
            wmat.move_to(start_pos)
            self.play(
                wmat.animate(rate_func=smooth, run_time=0.55)
                    .move_to(wmat.get_center() + LEFT * 2.5)
            )
            # glow pulse
            pulse = SurroundingRectangle(wmat[0], color=C_GLOW,
                                          buff=0.06, stroke_width=2.0)
            self.play(Create(pulse, run_time=0.22))
            self.play(FadeOut(pulse, run_time=0.22))
            self.wait(0.12)

        self.wait(0.5)
        self.play(FadeOut(title, run_time=0.4))

        self.wq = wq
        self.wk = wk
        self.wv = wv

    # ════════════════════════════════════════════════════════════════════════
    # SCENE 3 — Matrix Multiplication
    # ════════════════════════════════════════════════════════════════════════
    def scene3_matrix_multiplication(self):
        title = Text("X  ×  W  →  Q, K, V", font="Fira Code Bold",
                     color=C_TEXT, font_size=40, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title, shift=DOWN * 0.18, run_time=0.55))

        # result targets
        qm = make_matrix(ROWS_X, COLS_W, C_QUERY, label=r"Q")
        km = make_matrix(ROWS_X, COLS_W, C_KEY,   label=r"K")
        vm = make_matrix(ROWS_X, COLS_W, C_VALUE,  label=r"V")

        qm.move_to(COL_R + ROW_Q)
        km.move_to(COL_R + ROW_K)
        vm.move_to(COL_R + ROW_V)

        triples = [
(self.wq, qm, C_QUERY,  r"Q = X \cdot W_Q \quad (3 \times d_{\text{model}})"), 
(self.wk, km, C_KEY,    r"K = X \cdot W_K \quad (3 \times d_{\text{model}})"), 
(self.wv, vm, C_VALUE,  r"V = X \cdot W_V \quad (3 \times d_{\text{model}})"),
        ]

        for wmat, rmat, color, formula_str in triples:
            # ── Step 1: highlight X and W ────────────────────────────────
            box_x = SurroundingRectangle(self.x_mat[0], color=color,
                                          buff=0.07, stroke_width=2.5)
            box_w = SurroundingRectangle(wmat[0], color=C_GLOW,
                                          buff=0.07, stroke_width=2.5)
            self.play(
                Create(box_x, run_time=0.3),
                Create(box_w, run_time=0.3),
                self.x_mat.animate(rate_func=there_and_back,
                                   run_time=0.4).scale(1.04),
                wmat.animate(rate_func=there_and_back,
                              run_time=0.4).scale(1.04),
            )

            # ── Step 2: data flow arc ─────────────────────────────────────
            arc = ArcBetweenPoints(
                self.x_mat.get_right(),
                wmat.get_left(),
                angle=-0.25,
                color=color,
                stroke_opacity=0.55,
                stroke_width=2.2
            )
            dot = Dot(color=color, radius=0.08).move_to(self.x_mat.get_right())
            self.play(Create(arc, run_time=0.35),
                      MoveAlongPath(dot, arc, run_time=0.45, rate_func=smooth))

            # flash multiply
            flash = SurroundingRectangle(wmat[0], color=color,
                                          buff=0.05, stroke_width=3.5)
            self.play(Create(flash, run_time=0.18))
            self.play(FadeOut(flash, run_time=0.18))

            # ── Step 3: result flows out ──────────────────────────────────
            arc2 = ArcBetweenPoints(
                wmat.get_right(),
                rmat.get_left(),
                angle=-0.25,
                color=color,
                stroke_opacity=0.55,
                stroke_width=2.2
            )
            dot2 = dot.copy().move_to(wmat.get_right())
            self.play(Create(arc2, run_time=0.35),
                      MoveAlongPath(dot2, arc2, run_time=0.45, rate_func=smooth))

            # ── Step 4: result matrix appears ────────────────────────────
            self.play(
                FadeIn(rmat, scale=0.90, run_time=0.45, rate_func=smooth),
            )

            # formula annotation
            formula = MathTex(formula_str, color=color, font_size=28)
            formula.next_to(rmat, DOWN, buff=0.28)
            self.play(FadeIn(formula, run_time=0.35))
            self.wait(0.3)

            # ── Step 5: clean arcs / boxes ───────────────────────────────
            self.play(FadeOut(safe_vgroup(arc, arc2, dot, dot2,
                                          box_x, box_w, formula),
                               run_time=0.25))
            self.wait(0.15)

        self.play(FadeOut(title, run_time=0.4))

        self.qm = qm
        self.km = km
        self.vm = vm

    # ════════════════════════════════════════════════════════════════════════
    # SCENE 4 — Clean Transition
    # ════════════════════════════════════════════════════════════════════════
    def scene4_clean_transition(self):
        # fade out X, word labels, weight matrices
        self.play(
            FadeOut(safe_vgroup(self.x_mat, self.word_labels,
                                 self.wq, self.wk, self.wv),
                    run_time=0.7)
        )
        self.wait(0.3)

    # ════════════════════════════════════════════════════════════════════════
    # SCENE 5 — Column Alignment
    # ════════════════════════════════════════════════════════════════════════
    def scene5_column_alignment(self):
        # Vertical stack: Q, K, V aligned left-center
        TARGET_X = LEFT * 3.0

        q_target = TARGET_X + UP    * 2.1
        k_target = TARGET_X + ORIGIN
        v_target = TARGET_X + DOWN  * 2.1

        self.play(
            self.qm.animate(rate_func=smooth, run_time=0.8).move_to(q_target),
            self.km.animate(rate_func=smooth, run_time=0.8).move_to(k_target),
            self.vm.animate(rate_func=smooth, run_time=0.8).move_to(v_target),
        )
        self.wait(0.4)

    # ════════════════════════════════════════════════════════════════════════
    # SCENE 6 — Q Row Explanation
    # ════════════════════════════════════════════════════════════════════════
    def scene6_q_rows(self):
        self._explain_rows(
            mat=self.qm,
            color=C_QUERY,
            title_str="Q Matrix — Query Vectors",
            labels=[r"Q_{\text{I}}", r"Q_{\text{like}}", r"Q_{\text{Maths}}"],
            description="Each row = a Query vector for that token"
        )

    # ════════════════════════════════════════════════════════════════════════
    # SCENE 7 — K Row Explanation
    # ════════════════════════════════════════════════════════════════════════
    def scene7_k_rows(self):
        self._explain_rows(
            mat=self.km,
            color=C_KEY,
            title_str="K Matrix — Key Vectors",
            labels=[r"K_{\text{I}}", r"K_{\text{like}}", r"K_{\text{Maths}}"],
            description="Each row = a Key vector for that token"
        )

    # ════════════════════════════════════════════════════════════════════════
    # SCENE 8 — V Row Explanation
    # ════════════════════════════════════════════════════════════════════════
    def scene8_v_rows(self):
        self._explain_rows(
            mat=self.vm,
            color=C_VALUE,
            title_str="V Matrix — Value Vectors",
            labels=[r"V_{\text{I}}", r"V_{\text{like}}", r"V_{\text{Maths}}"],
            description="Each row = a Value vector for that token",
            is_last=True
        )

    # ════════════════════════════════════════════════════════════════════════
    # SHARED: Row explanation helper
    # ════════════════════════════════════════════════════════════════════════
    def _explain_rows(self, mat, color, title_str, labels,
                      description, is_last=False):
        title = Text(title_str, font="Fira Code Bold",
                     color=color, font_size=38, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.play(FadeIn(title, shift=DOWN * 0.18, run_time=0.55))

        # glow mat
        glow_box = SurroundingRectangle(mat[0], color=C_GLOW,
                                         buff=0.08, stroke_width=2.5)
        self.play(Create(glow_box, run_time=0.35))
        self.play(mat.animate(rate_func=there_and_back,
                               run_time=0.4).scale(1.05))

        row_highlights = VGroup()
        arrows_group   = VGroup()
        label_group    = VGroup()

        cols = COLS_W

        for row_idx, lbl_tex in enumerate(labels):
            # row cells
            row_cells = VGroup(*[mat[0][row_idx * cols + c]
                                  for c in range(cols)])

            # highlight row
            hl = SurroundingRectangle(row_cells, color=color,
                                       buff=0.05, stroke_width=2.2)
            row_highlights.add(hl)

            # arrow: from right edge of row → label
            arrow_start = row_cells.get_right() + RIGHT * 0.08
            arrow_end   = arrow_start + RIGHT * 1.25

            arr = Arrow(arrow_start, arrow_end,
                        color=color, stroke_width=2.2,
                        tip_length=0.17, buff=0)
            arrows_group.add(arr)

            # label
            lbl = MathTex(lbl_tex, color=color, font_size=30)
            lbl.next_to(arr, RIGHT, buff=0.12)
            label_group.add(lbl)

            self.play(
                Create(hl,  run_time=0.28),
                GrowArrow(arr, run_time=0.32),
                FadeIn(lbl, shift=RIGHT * 0.1, run_time=0.3),
            )
            self.wait(0.18)

        # description line
        desc = Text(description, font="Fira Code",
                    color=C_DIM, font_size=25)
        desc.match_y(mat)
        desc.to_edge(RIGHT,buff=1)
        self.play(FadeIn(desc, run_time=0.45))
        self.wait(1.0)

        if is_last:
            # final hold — show all three matrices + labels simultaneously
            self.wait(1.2)
            # hard fade to black
            black = Rectangle(
                width=config.frame_width, height=config.frame_height,
                fill_color=BLACK, fill_opacity=1, stroke_opacity=0
            )
            self.play(FadeIn(black, run_time=0.5))
            self.wait(0.3)
        else:
            # clean up annotations before next scene
            self.play(
                FadeOut(safe_vgroup(title, glow_box, desc,
                                     row_highlights, arrows_group,
                                     label_group),
                        run_time=0.5)
            )
            self.wait(0.2)