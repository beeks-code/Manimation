"""
Cinematic Manim Animation:
Static vs Contextual Embeddings
"The bank is near the river bank."
"""

from manim import *
import numpy as np

BG       = "#0f172a"
C_QUERY  = "#3b82f6"   # first  "bank" (financial)
C_KEY    = "#22c55e"
C_VALUE  = "#f97316"   # second "bank" (river)
C_GLOW   = "#a78bfa"
C_TEXT   = "#e2e8f0"
C_DIM    = "#475569"
C_ACCENT = "#f472b6"

config.background_color = BG
config.pixel_height = 1080
config.pixel_width  = 1920
config.frame_rate   = 60

CELL = 0.42   # matrix cell size
ROWS = 6      # rows per embedding matrix (vocab preview)
COLS = 5      # cols per embedding matrix (dim preview)


def safe_vgroup(*mobs):
    return VGroup(*[m for m in mobs if m is not None])


def make_grid_bg(opacity=0.045):
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


def make_embedding_matrix(rows, cols, color, bank_rows,
                           cell=CELL, label="", same_bank=True):
    """
    Build a visual embedding matrix.
    bank_rows: list of row indices to tint as 'bank' rows.
    same_bank: if True both bank rows share the same color (static).
    """
    cells = VGroup()
    for r in range(rows):
        for c in range(cols):
            if r in bank_rows:
                if same_bank:
                    tint = C_QUERY
                else:
                    tint = C_QUERY if r == bank_rows[0] else C_VALUE
                fill_op = 0.22
            else:
                tint = color
                fill_op = 0.08
            sq = Square(side_length=cell,
                        color=tint,
                        fill_color=tint,
                        fill_opacity=fill_op,
                        stroke_width=1.2)
            sq.move_to([c * cell - (cols - 1) * cell / 2,
                        -r * cell + (rows - 1) * cell / 2, 0])
            cells.add(sq)

    bl = MathTex(r"\big[", color=color, font_size=58)
    br = MathTex(r"\big]", color=color, font_size=58)
    bl.next_to(cells, LEFT,  buff=0.05)
    br.next_to(cells, RIGHT, buff=0.05)
    group = VGroup(cells, bl, br)

    if label:
        lbl = Text(label, font="Fira Code Bold", color=color,
                   font_size=22, weight=BOLD)
        lbl.next_to(group, UP, buff=0.18)
        group.add(lbl)
    return group


def row_rect(mat, row_idx, cols, color, buff=0.06):
    row_cells = VGroup(*[mat[0][row_idx * cols + c] for c in range(cols)])
    return SurroundingRectangle(row_cells, color=color,
                                 buff=buff, stroke_width=2.5)


def glow_pulse(scene, mob, color=C_GLOW, n=2):
    for _ in range(n):
        scene.play(mob.animate(rate_func=there_and_back,
                               run_time=0.38).scale(1.04))


def section_label(scene, txt, color=C_TEXT, font_size=38):
    t = Text(txt, font="Fira Code Bold", color=color,
             font_size=font_size, weight=BOLD)
    t.to_edge(UP, buff=0.35)
    scene.play(FadeIn(t, shift=DOWN * 0.18, run_time=0.55))
    return t



class StaticVsContextual(Scene):

    def construct(self):
        self.add(make_grid_bg())
        self.scene1_sentence()
        self.scene2_highlight_banks()
        self.scene3_introduce_matrices()
        self.scene4_static_comparison()
        self.scene5_contextual_comparison()
        self.scene6_comparison_table()


    def scene1_sentence(self):
        title = section_label(self, "Static vs Contextual Embeddings")

        words = ["The", "bank", "is", "near", "the", "river", "bank."]
        word_mobs = VGroup(*[
            Text(w, font="Fira Code", color=C_TEXT, font_size=46)
            for w in words
        ])
        word_mobs.arrange(RIGHT, buff=0.28)
        word_mobs.move_to(UP * 1.8)

        self.play(
            LaggedStart(*[FadeIn(w, shift=UP * 0.14, run_time=0.32)
                          for w in word_mobs], lag_ratio=0.12)
        )
        self.wait(0.5)

        # store references to the two "bank" words
        self.word_mobs    = word_mobs
        self.bank1        = word_mobs[1]   # financial 
        self.bank2        = word_mobs[6]   
        self.scene_title  = title

    def scene2_highlight_banks(self):
        # color the two banks
        self.play(
            self.bank1.animate(rate_func=smooth, run_time=0.5)
                      .set_color(C_QUERY),
        )
        # glow pulse on bank1
        glow1 = self.bank1.copy().set_color(C_GLOW).set_opacity(0.35).scale(1.12)
        self.play(FadeIn(glow1, run_time=0.2))
        self.play(FadeOut(glow1, run_time=0.3))

        lbl1 = Text("(financial)", font="Fira Code", color=C_QUERY, font_size=20)
        lbl1.next_to(self.bank1, DOWN, buff=0.15)
        self.play(FadeIn(lbl1, run_time=0.35))
        self.wait(0.3)

        self.play(
            self.bank2.animate(rate_func=smooth, run_time=0.5)
                      .set_color(C_VALUE),
        )
        glow2 = self.bank2.copy().set_color(C_GLOW).set_opacity(0.35).scale(1.12)
        self.play(FadeIn(glow2, run_time=0.2))
        self.play(FadeOut(glow2, run_time=0.3))

        lbl2 = Text("(river)", font="Fira Code", color=C_VALUE, font_size=20)
        lbl2.next_to(self.bank2, DOWN, buff=0.15)
        self.play(FadeIn(lbl2, run_time=0.35))
        self.wait(0.6)

        question = Text(
            "Same word,same meaning?",
            font="Fira Code", color=C_DIM, font_size=26
        )
        question.move_to(UP * 0.8)
        self.play(FadeIn(question, run_time=0.5))
        self.wait(0.8)

        self.play(FadeOut(safe_vgroup(lbl1, lbl2, question), run_time=0.4))
        self.bank1_lbl = lbl1
        self.bank2_lbl = lbl2

    def scene3_introduce_matrices(self):
        self.play(FadeOut(self.scene_title, run_time=0.3))

        # move sentence up tight to top
        self.play(
            self.word_mobs.animate(rate_func=smooth, run_time=0.6)
                          .to_edge(UP, buff=0.25).scale(0.82)
        )
        self.wait(0.2)

        stat_mat = make_embedding_matrix(
            ROWS, COLS, C_DIM,
            bank_rows=[2],
            same_bank=True,
            label="Static Embedding"
        )
        stat_mat.move_to(LEFT * 4.5 + DOWN * 0.6)


        ctx_mat = make_embedding_matrix(
            ROWS, COLS, C_DIM,
            bank_rows=[2, 4],
            same_bank=False,
            label="Contextual Embedding"
        )
        ctx_mat.move_to(RIGHT * 4.5 + DOWN * 0.6)

        self.play(
            FadeIn(stat_mat, shift=RIGHT * 0.3, run_time=0.7, rate_func=smooth),
        )
        self.play(
            FadeIn(ctx_mat,  shift=LEFT  * 0.3, run_time=0.7, rate_func=smooth),
        )
        self.wait(0.4)

        self.stat_mat = stat_mat
        self.ctx_mat  = ctx_mat


    def scene4_static_comparison(self):
        stat = self.stat_mat

        sub = section_label(self, "Static Embeddings ",
                            color=C_QUERY, font_size=30)
        sub.move_to(UP * 0.85)

        self.play(self.ctx_mat.animate(run_time=0.4).set_opacity(0.25))

        bank_row = 2
        row_cells = VGroup(*[stat[0][bank_row * COLS + c] for c in range(COLS)])
        hl = SurroundingRectangle(row_cells, color=C_QUERY,
                                   buff=0.06, stroke_width=2.5)

        # arrow from bank1
        arr1 = Arrow(
            self.bank1.get_bottom() + DOWN * 0.08,
            row_cells.get_top()     + UP   * 0.08,
            color=C_QUERY, stroke_width=2.2,
            tip_length=0.18, buff=0.05
        )
        # arrow from bank2
        arr2 = Arrow(
            self.bank2.get_bottom() + DOWN * 0.08,
            row_cells.get_top()     + UP   * 0.08,
            color=C_VALUE, stroke_width=2.2,
            tip_length=0.18, buff=0.05
        )

        self.play(GrowArrow(arr1, run_time=0.5))
        self.play(GrowArrow(arr2, run_time=0.5))
        self.play(Create(hl, run_time=0.4))
        glow_pulse(self, hl, C_QUERY, n=2)

        # duplicate row to emphasize same vector
        dup = row_cells.copy().set_color(C_VALUE).set_fill(C_VALUE, opacity=0.25)
        dup.next_to(row_cells, DOWN, buff=0.08)
        self.play(FadeIn(dup, run_time=0.35))
        same_lbl = Text("← same vector!", font="Fira Code",
                        color=C_QUERY, font_size=20)
        same_lbl.next_to(hl, RIGHT, buff=0.2)
        self.play(FadeIn(same_lbl, run_time=0.35))
        self.wait(0.5)

    

        self.play(FadeOut(safe_vgroup(sub, arr1, arr2, hl, dup,
                                      same_lbl), run_time=0.5))
        self.play(self.ctx_mat.animate(run_time=0.4).set_opacity(1.0))

    def scene5_contextual_comparison(self):
        ctx = self.ctx_mat

        sub = section_label(self, "Contextual Embeddings",
                            color=C_VALUE, font_size=32)
        sub.move_to(UP * 0.85)
        self.play(self.stat_mat.animate(run_time=0.4).set_opacity(0.25))

        row2_cells = VGroup(*[ctx[0][2 * COLS + c] for c in range(COLS)])
        hl2 = SurroundingRectangle(row2_cells, color=C_QUERY,
                                    buff=0.06, stroke_width=2.5)
        arr1 = Arrow(
            self.bank1.get_bottom() + DOWN * 0.08,
            row2_cells.get_top()    + UP   * 0.08,
            color=C_QUERY, stroke_width=2.2,
            tip_length=0.18, buff=0.05
        )
        lbl_r2 = Text("financial\n bank", font="Fira Code",
                       color=C_QUERY, font_size=18)
        lbl_r2.next_to(hl2, RIGHT, buff=0.18)

        row4_cells = VGroup(*[ctx[0][4 * COLS + c] for c in range(COLS)])
        hl4 = SurroundingRectangle(row4_cells, color=C_VALUE,
                                    buff=0.06, stroke_width=2.5)
        arr2 = Arrow(
            self.bank2.get_bottom() + DOWN * 0.08,
            row4_cells.get_top()    + UP   * 0.08,
            color=C_VALUE, stroke_width=2.2,
            tip_length=0.18, buff=0.05
        )
        lbl_r4 = Text("river bank", font="Fira Code",
                       color=C_VALUE, font_size=18)
        lbl_r4.next_to(hl4, RIGHT, buff=0.18)

        self.play(GrowArrow(arr1, run_time=0.5))
        self.play(Create(hl2, run_time=0.35))
        glow_pulse(self, hl2, C_GLOW, n=1)
        self.play(FadeIn(lbl_r2, run_time=0.3))
        self.wait(0.3)

        self.play(GrowArrow(arr2, run_time=0.5))
        self.play(Create(hl4, run_time=0.35))
        glow_pulse(self, hl4, C_GLOW, n=1)
        self.play(FadeIn(lbl_r4, run_time=0.3))
        self.wait(0.5)

        diff_lbl = Text(" different vectors! ", font="Fira Code",
                         color=C_ACCENT, font_size=20)
        diff_lbl.move_to(RIGHT * 1.5 + DOWN * 1.3)
        self.play(FadeIn(diff_lbl, run_time=0.35))
        self.wait(0.3)

        self.play(FadeOut(safe_vgroup(sub, arr1, arr2, hl2, hl4,
                                      lbl_r2, lbl_r4, diff_lbl),
                           run_time=0.5))
        self.play(self.stat_mat.animate(run_time=0.4).set_opacity(1.0))

    def scene6_comparison_table(self):
        self.play(
            FadeOut(safe_vgroup(self.stat_mat, self.ctx_mat,
                                 self.word_mobs), run_time=0.6)
        )

        title = section_label(self, "TLDR: Static vs Contextual", color=C_GLOW)

        headers = ["Embedding Type", 'Representation of "bank"', "Notes"]
        rows_data = [
            ["Static\n(GloVe, Word2Vec)",  "Same for both occurrences", "Context ignored"],
            ["Contextual\n(BERT, GPT)",     "Different per occurrence",  "Depends on context"],
        ]
        row_colors = [C_QUERY, C_VALUE]

        col_widths = [3.2, 4.0, 3.2]
        row_h = 0.85
        col_x_starts = [-5.2, -2.0, 2.0]

        def make_cell(text, fg, bg, font_size=22, bold=False):
            rect = Rectangle(
                width=col_widths[0] if bold else col_widths[0],
                height=row_h,
                fill_color=bg, fill_opacity=0.15,
                stroke_color=fg, stroke_width=1.2
            )
            t = Text(text, font="Fira Code", color=fg,
                     font_size=font_size, weight=BOLD if bold else NORMAL)
            t.move_to(rect)
            return VGroup(rect, t)

        header_group = VGroup()
        for i, (hdr, cx) in enumerate(zip(headers, col_x_starts)):
            rect = Rectangle(width=col_widths[i], height=row_h,
                              fill_color=C_GLOW, fill_opacity=0.18,
                              stroke_color=C_GLOW, stroke_width=1.5)
            rect.move_to([cx + col_widths[i] / 2, 1.6, 0])
            t = Text(hdr, font="Fira Code Bold", color=C_GLOW,
                     font_size=22, weight=BOLD)
            t.move_to(rect)
            header_group.add(VGroup(rect, t))

        self.play(FadeIn(header_group, shift=DOWN * 0.15, run_time=0.6))
        all_row_groups = []
        for ri, (row_vals, rcol) in enumerate(zip(rows_data, row_colors)):
            y = 1.6 - (ri + 1) * row_h
            row_group = VGroup()
            for ci, (val, cx) in enumerate(zip(row_vals, col_x_starts)):
                rect = Rectangle(width=col_widths[ci], height=row_h,
                                  fill_color=rcol, fill_opacity=0.08,
                                  stroke_color=C_DIM, stroke_width=0.9)
                rect.move_to([cx + col_widths[ci] / 2, y, 0])
                t = Text(val, font="Fira Code", color=rcol,
                         font_size=20)
                t.move_to(rect)
                row_group.add(VGroup(rect, t))
            all_row_groups.append(row_group)

        for rg in all_row_groups:
            self.play(FadeIn(rg, shift=RIGHT * 0.12, run_time=0.45))
            self.wait(0.2)

        for _ in range(2):
            self.play(
                all_row_groups[1].animate(rate_func=there_and_back,
                                           run_time=0.4).scale(1.02)
            )
   ## show the difference
        narration = Text(
            '"Static embeddings assign the same vector to \'bank\' everywhere.\n'
            'Contextual embeddings assign vectors based on Context.\n'
            'NLP Models like BERT or GPT use attention mechanism for context aware vectors."',
            font="Fira Code", color=C_DIM, font_size=19,
            line_spacing=1.4
        )
        narration.next_to(all_row_groups[-1][0], DOWN, buff=0.45)
        narration.to_edge(LEFT, buff=0.5)
        self.play(FadeIn(narration, run_time=0.7))
        self.wait(2.0)

        black = Rectangle(
            width=config.frame_width, height=config.frame_height,
            fill_color=BLACK, fill_opacity=1, stroke_opacity=0
        )
        self.play(FadeIn(black, run_time=0.6))
        self.wait(0.3)