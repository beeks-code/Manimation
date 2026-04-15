from manim import *
import numpy as np

BG     = "#0f172a"
C_Q    = "#3b82f6"   
C_K    = "#22c55e"  
C_V    = "#f97316"   
C_GLOW = "#a78bfa"   
C_TEXT = "#e2e8f0"
C_DIM  = "#475569"
C_ACCT = "#f472b6"
TOKEN_COLORS = [C_Q, C_GLOW, C_ACCT]

config.background_color = BG
config.pixel_height = 1080
config.pixel_width  = 1920
config.frame_rate   = 60

WORDS = ["I", "Like", "Math"]


def safe_vg(*mobs):
    return VGroup(*[m for m in mobs if m is not None])


def make_grid_bg():
    lines = VGroup()
    for i in range(29):
        x = -config.frame_width / 2 + i * config.frame_width / 28
        lines.add(Line([x, -config.frame_height/2, 0],
                       [x,  config.frame_height/2, 0],
                       stroke_color="#1e293b", stroke_width=0.4,
                       stroke_opacity=0.45))
    for j in range(17):
        y = -config.frame_height / 2 + j * config.frame_height / 16
        lines.add(Line([-config.frame_width/2, y, 0],
                       [ config.frame_width/2, y, 0],
                       stroke_color="#1e293b", stroke_width=0.4,
                       stroke_opacity=0.45))
    return lines


def token_box(word, color):
    txt  = Text(word, font="Fira Code Bold", color=color,
                font_size=36, weight=BOLD)
    rect = RoundedRectangle(corner_radius=0.14,
                            width=txt.width + 0.55,
                            height=txt.height + 0.40,
                            color=color, fill_color=BG,
                            fill_opacity=1.0, stroke_width=2.0)
    rect.move_to(txt)
    g = VGroup()
    for i in (3, 2, 1):
        r = rect.copy()
        r.set_stroke(color, opacity=0.18 * i / 3,
                     width=rect.get_stroke_width() * (1 + 0.5 * i))
        r.set_fill(opacity=0)
        g.add(r)
    return VGroup(g, rect, txt)


def vec_box(color, rows=4, cell=0.38):
    cells = VGroup()
    for r in range(rows):
        sq = Square(side_length=cell, color=color,
                    fill_color=BG, fill_opacity=1.0,
                    stroke_width=1.6)
        sq.move_to(DOWN * r * cell)
        cells.add(sq)
    bl = MathTex(r"\big[", color=color, font_size=48)
    br = MathTex(r"\big]", color=color, font_size=48)
    bl.next_to(cells, LEFT,  buff=0.04)
    br.next_to(cells, RIGHT, buff=0.04)
    return VGroup(cells, bl, br)


def info_block(title_str, lines, title_color, width=5.5):
    title = Text(title_str, font="Fira Code Bold", color=title_color,
                 font_size=28, weight=BOLD)
    body_mobs = VGroup(*[
        Text(l, font="Fira Code", color=C_DIM, font_size=20)
        for l in lines
    ]).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
    body_mobs.next_to(title, DOWN, buff=0.18, aligned_edge=LEFT)
    box = VGroup(title, body_mobs)
    bg  = SurroundingRectangle(box, color=title_color, buff=0.22,
                                stroke_width=1.4, fill_color=BG,
                                fill_opacity=0.85, corner_radius=0.12)
    return VGroup(bg, box)


def sec_title(scene, txt, color=C_TEXT, fs=40):
    t = Text(txt, font="Fira Code Bold", color=color,
             font_size=fs, weight=BOLD)
    t.to_edge(UP, buff=0.30)
    scene.play(FadeIn(t, shift=DOWN * 0.15, run_time=0.50))
    return t


def glow_pulse(scene, mob, color=C_GLOW, n=1):
    for _ in range(n):
        scene.play(mob.animate(rate_func=there_and_back,
                               run_time=0.40).scale(1.05))



class QKVExplain(Scene):

    def construct(self):
        self.add(make_grid_bg())
        self.s1_title()
        self.s2_tokens()
        self.s3_embeddings()
        self.s4_projections()
        self.s5_query()
        self.s6_key()
        self.s7_scores()
        self.s8_value()
        self.s9_output()
        self.s10_summary()

    def s1_title(self):
        main = Text("Understanding Q, K, V",
                    font="Fira Code Bold", color=C_TEXT,
                    font_size=58, weight=BOLD)
        sub  = Text("in Self-Attention",
                    font="Fira Code", color=C_GLOW, font_size=36)
        sub.next_to(main, DOWN, buff=0.28)
        group = VGroup(main, sub).move_to(ORIGIN)

        glow = VGroup()
        for i in (3, 2, 1):
            g = main.copy().set_color(C_GLOW)
            g.set_opacity(0.12 * i / 3).scale(1 + 0.012 * i)
            glow.add(g)

        self.play(
            LaggedStart(
                FadeIn(glow,  run_time=0.8),
                FadeIn(main,  scale=0.90, run_time=0.9),
                FadeIn(sub,   shift=UP * 0.1, run_time=0.7),
                lag_ratio=0.2
            )
        )
        self.wait(1.2)
        self.play(FadeOut(safe_vg(glow, main, sub), run_time=0.6))
        self.wait(0.15)

    def s2_tokens(self):
        title = sec_title(self, "Input Tokens")

        lbl = Text("Input Tokens", font="Fira Code",
                   color=C_DIM, font_size=24)
        lbl.move_to(UP * 1.8)

        tokens = VGroup(*[token_box(w, c)
                          for w, c in zip(WORDS, TOKEN_COLORS)])
        tokens.arrange(RIGHT, buff=1.0)
        tokens.move_to(ORIGIN)

        self.play(FadeIn(lbl, run_time=0.45))
        self.play(
            LaggedStart(*[FadeIn(t, shift=UP * 0.18, run_time=0.5)
                          for t in tokens], lag_ratio=0.20)
        )
        self.wait(0.8)
        self.play(FadeOut(safe_vg(title, lbl), run_time=0.35))

        self.tokens = tokens

    def s3_embeddings(self):
        title = sec_title(self, "Token Embeddings")

        self.play(self.tokens.animate(rate_func=smooth, run_time=0.55)
                             .move_to(UP * 2.4))

        embed_lbl = Text("Embeddings", font="Fira Code",
                         color=C_DIM, font_size=22)
        embed_lbl.move_to(DOWN * 0.1)

        vecs = VGroup()
        arrows_down = VGroup()
        for i, (tok, color) in enumerate(zip(self.tokens, TOKEN_COLORS)):
            v = vec_box(color)
            v.move_to([tok.get_center()[0], -0.6, 0])
            vecs.add(v)

            arr = Arrow(tok.get_bottom() + DOWN * 0.08,
                        v[0][0].get_top() + UP * 0.08,
                        color=color, stroke_width=2.0,
                        tip_length=0.16, buff=0.02)
            arrows_down.add(arr)

        self.play(FadeIn(embed_lbl, run_time=0.4))
        self.play(
            LaggedStart(*[GrowArrow(a, run_time=0.4) for a in arrows_down],
                        lag_ratio=0.18)
        )
        self.play(
            LaggedStart(*[FadeIn(v, scale=0.85, run_time=0.45)
                          for v in vecs], lag_ratio=0.18)
        )
        self.wait(0.8)
        self.play(FadeOut(safe_vg(title, embed_lbl, arrows_down), run_time=0.4))

        self.vecs = vecs

    def s4_projections(self):
        title = sec_title(self, "Linear Projection  →  Q, K, V")

        col_positions = [LEFT * 4.5, ORIGIN, RIGHT * 4.5]
        col_colors    = [C_Q, C_K, C_V]
        col_names     = ["Q  (Query)", "K  (Key)", "V  (Value)"]

        q_vecs = VGroup()
        k_vecs = VGroup()
        v_vecs = VGroup()
        all_qkv = [q_vecs, k_vecs, v_vecs]

        col_lbls = VGroup()
        branch_arrows = VGroup()

        for ci, (cx, color, name) in enumerate(
                zip(col_positions, col_colors, col_names)):

            lbl = Text(name, font="Fira Code Bold", color=color,
                       font_size=24, weight=BOLD)
            lbl.move_to([cx[0], 2.55, 0])
            col_lbls.add(lbl)

            for ti, (tok, tc) in enumerate(zip(self.tokens, TOKEN_COLORS)):
                v = vec_box(color, rows=3, cell=0.36)
                row_y = [0.6, -0.5, -1.6]
                v.move_to([cx[0], row_y[ti], 0])
                all_qkv[ci].add(v)

                src = self.vecs[ti].get_center()
                dst = v[0][0].get_top() + UP * 0.06
                arr = CurvedArrow(src, dst, angle=-0.3 + ci * 0.3,
                                  color=color, stroke_width=1.5,
                                  tip_length=0.14)
                arr.set_stroke(opacity=0.55)
                branch_arrows.add(arr)

        self.play(
            FadeOut(safe_vg(self.tokens, self.vecs), run_time=0.5),
        )
        self.play(
            LaggedStart(*[FadeIn(l, run_time=0.35) for l in col_lbls],
                        lag_ratio=0.15)
        )
        self.play(
            LaggedStart(*[Create(a, run_time=0.35) for a in branch_arrows],
                        lag_ratio=0.04)
        )
        self.play(
            LaggedStart(*[FadeIn(v, scale=0.88, run_time=0.38)
                          for vgroup in all_qkv for v in vgroup],
                        lag_ratio=0.06)
        )
        self.wait(0.8)
        self.play(FadeOut(safe_vg(title, branch_arrows), run_time=0.35))

        self.q_vecs   = q_vecs
        self.k_vecs   = k_vecs
        self.v_vecs   = v_vecs
        self.col_lbls = col_lbls

    def s5_query(self):
        title = sec_title(self, "Query (Q)", color=C_Q)

        self.play(
            self.k_vecs.animate(run_time=0.4).set_opacity(0.18),
            self.v_vecs.animate(run_time=0.4).set_opacity(0.18),
            self.col_lbls[1].animate(run_time=0.4).set_opacity(0.18),
            self.col_lbls[2].animate(run_time=0.4).set_opacity(0.18),
        )

        qb = SurroundingRectangle(self.q_vecs, color=C_Q,
                                   buff=0.12, stroke_width=2.5)
        self.play(Create(qb, run_time=0.35))
        glow_pulse(self, qb, color=C_Q, n=2)

        info = info_block(
            "Query (Q)",
            ["Represents what a token is seeking",
             "from other tokens.",
             "Determines which tokens to attend to."],
            C_Q
        )
        info.move_to(RIGHT * 3.2 + DOWN * 0.2)
        self.play(FadeIn(info, shift=LEFT * 0.15, run_time=0.55))

        q0_center = self.q_vecs[0].get_center()
        attn_arrows = VGroup()
        for qv in self.q_vecs:
            a = Arrow(q0_center + LEFT * 0.05,
                      qv.get_left() + LEFT * 0.08,
                      color=C_Q, stroke_width=1.8,
                      tip_length=0.14, buff=0.04)
            a.set_stroke(opacity=0.6)
            attn_arrows.add(a)

        self.play(LaggedStart(*[GrowArrow(a, run_time=0.35)
                                for a in attn_arrows], lag_ratio=0.2))
        self.wait(1.2)

        self.play(FadeOut(safe_vg(title, qb, info, attn_arrows), run_time=0.4))
        self.play(
            self.k_vecs.animate(run_time=0.35).set_opacity(1.0),
            self.v_vecs.animate(run_time=0.35).set_opacity(1.0),
            self.col_lbls[1].animate(run_time=0.35).set_opacity(1.0),
            self.col_lbls[2].animate(run_time=0.35).set_opacity(1.0),
        )

    def s6_key(self):
        title = sec_title(self, "Key (K)", color=C_K)

        self.play(
            self.q_vecs.animate(run_time=0.4).set_opacity(0.18),
            self.v_vecs.animate(run_time=0.4).set_opacity(0.18),
            self.col_lbls[0].animate(run_time=0.4).set_opacity(0.18),
            self.col_lbls[2].animate(run_time=0.4).set_opacity(0.18),
        )

        kb = SurroundingRectangle(self.k_vecs, color=C_K,
                                   buff=0.12, stroke_width=2.5)
        self.play(Create(kb, run_time=0.35))
        glow_pulse(self, kb, color=C_K, n=2)

        info = info_block(
            "Key (K)",
            ["Represents what a token contains",
             "or offers.",
             "Determines how relevant a token is."],
            C_K
        )
        info.move_to(RIGHT * 3.2 + DOWN * 0.2)
        self.play(FadeIn(info, shift=LEFT * 0.15, run_time=0.55))

        dot_formula = MathTex(r"Q \cdot K = \text{attention score}",
                              color=C_GLOW, font_size=30)
        dot_formula.next_to(info, DOWN, buff=0.35)
        self.play(Write(dot_formula, run_time=0.7))

        q0 = self.q_vecs[0].get_center()
        arcs = VGroup()
        for kv in self.k_vecs:
            arc = CurvedArrow(q0, kv.get_top() + UP * 0.06,
                              angle=-0.4, color=C_GLOW,
                              stroke_width=1.8, tip_length=0.14)
            arc.set_stroke(opacity=0.55)
            self.play(Create(arc, run_time=0.30))
            arcs.add(arc)

        self.wait(1.0)
        self.play(FadeOut(safe_vg(title, kb, info, dot_formula, arcs), run_time=0.4))
        self.play(
            self.q_vecs.animate(run_time=0.35).set_opacity(1.0),
            self.v_vecs.animate(run_time=0.35).set_opacity(1.0),
            self.col_lbls[0].animate(run_time=0.35).set_opacity(1.0),
            self.col_lbls[2].animate(run_time=0.35).set_opacity(1.0),
        )

    def s7_scores(self):
        title = sec_title(self, "Attention Scores  QKᵀ", color=C_GLOW)

        self.play(
            FadeOut(safe_vg(self.q_vecs, self.k_vecs,
                            self.v_vecs, self.col_lbls), run_time=0.5)
        )

        fake_scores = np.array([
            [0.85, 0.10, 0.05],
            [0.20, 0.70, 0.10],
            [0.05, 0.15, 0.80],
        ])
        CELL = 0.85
        heat_cells = VGroup()
        score_lbls = VGroup()
        row_lbl_names = ["I", "Like", "Math"]

        for r in range(3):
            for c in range(3):
                v   = fake_scores[r, c]
                col = interpolate_color(ManimColor(BG),
                                        ManimColor(C_GLOW), v)
                sq  = Square(side_length=CELL,
                             color=C_DIM,
                             fill_color=col,
                             fill_opacity=0.90,
                             stroke_width=1.2)
                sq.move_to([c * CELL - CELL, -r * CELL + CELL, 0])
                heat_cells.add(sq)

                sv = Text(f"{v:.2f}", font="Fira Code",
                          color=C_TEXT, font_size=20)
                sv.move_to(sq)
                score_lbls.add(sv)

        heat_group = VGroup(heat_cells, score_lbls)
        heat_group.move_to(LEFT * 1.5 + DOWN * 0.2)

        axis_lbls = VGroup()
        for i, w in enumerate(row_lbl_names):
            rl = Text(w, font="Fira Code", color=TOKEN_COLORS[i], font_size=22)
            rl.next_to(heat_cells[i * 3], LEFT, buff=0.22)
            cl = Text(w, font="Fira Code", color=TOKEN_COLORS[i], font_size=22)
            cl.next_to(heat_cells[i], UP, buff=0.18)
            axis_lbls.add(rl, cl)
        self.play(LaggedStart(*[FadeIn(l, run_time=0.2) for l in axis_lbls], lag_ratio=0.05))

        formula = MathTex(r"S = QK^T", color=C_GLOW, font_size=36)
        formula.move_to(RIGHT * 4.5 + UP * 1.0)
        note1 = Text("High score → strong attention",
                     font="Fira Code", color=C_GLOW, font_size=22)
        note2 = Text("Low score  → weak attention",
                     font="Fira Code", color=C_DIM, font_size=22)
        note1.next_to(formula, DOWN, buff=0.35)
        note2.next_to(note1,   DOWN, buff=0.22)

        self.play(
            LaggedStart(*[FadeIn(sq, scale=0.8, run_time=0.28)
                          for sq in heat_cells], lag_ratio=0.04)
        )
        self.play(
            LaggedStart(*[FadeIn(sv, run_time=0.18)
                          for sv in score_lbls], lag_ratio=0.04)
        )
        self.play(Write(formula, run_time=0.6))
        self.play(FadeIn(note1, run_time=0.4), FadeIn(note2, run_time=0.4))
        self.wait(1.2)

        self.play(FadeOut(safe_vg(title, heat_group, axis_lbls,
                                   formula, note1, note2), run_time=0.5))

        self.play(
            FadeIn(self.q_vecs,   run_time=0.4),
            FadeIn(self.k_vecs,   run_time=0.4),
            FadeIn(self.v_vecs,   run_time=0.4),
            FadeIn(self.col_lbls, run_time=0.4),
        )

    def s8_value(self):
        title = sec_title(self, "Value (V)", color=C_V)

        self.play(
            self.q_vecs.animate(run_time=0.4).set_opacity(0.18),
            self.k_vecs.animate(run_time=0.4).set_opacity(0.18),
            self.col_lbls[0].animate(run_time=0.4).set_opacity(0.18),
            self.col_lbls[1].animate(run_time=0.4).set_opacity(0.18),
        )

        vb = SurroundingRectangle(self.v_vecs, color=C_V,
                                   buff=0.12, stroke_width=2.5)
        self.play(Create(vb, run_time=0.35))
        glow_pulse(self, vb, color=C_V, n=2)

        info = info_block(
            "Value (V)",
            ["Represents the actual information",
             "of the token.",
             "Weighted and combined to form output."],
            C_V
        )
        info.move_to(LEFT * 3.5 + DOWN * 0.2)
        self.play(FadeIn(info, shift=RIGHT * 0.15, run_time=0.55))

        weights = [0.75, 0.15, 0.10]
        output_pt = RIGHT * 5.5 + DOWN * 0.0
        wsum_arrows = VGroup()
        for vi, (vv, wt) in enumerate(zip(self.v_vecs, weights)):
            a = Arrow(vv.get_right() + RIGHT * 0.08,
                      output_pt,
                      color=C_V,
                      stroke_width=1.5 + wt * 5,
                      tip_length=0.16, buff=0.04)
            a.set_stroke(opacity=0.35 + wt * 0.6)
            wt_lbl = Text(f"×{wt:.2f}", font="Fira Code",
                          color=C_V, font_size=18)
            wt_lbl.next_to(a.get_center(), UP, buff=0.08)
            wsum_arrows.add(a, wt_lbl)

        out_dot = Dot(output_pt, color=C_V, radius=0.18)
        out_lbl = Text("Output", font="Fira Code", color=C_V, font_size=20)
        out_lbl.next_to(out_dot, RIGHT, buff=0.14)

        self.play(
            LaggedStart(*[GrowArrow(wsum_arrows[i * 2], run_time=0.4)
                          for i in range(3)], lag_ratio=0.18)
        )
        self.play(
            LaggedStart(*[FadeIn(wsum_arrows[i * 2 + 1], run_time=0.3)
                          for i in range(3)], lag_ratio=0.18)
        )
        self.play(FadeIn(out_dot, scale=0.5, run_time=0.3),
                  FadeIn(out_lbl, run_time=0.3))
        self.wait(1.2)

        self.play(FadeOut(safe_vg(title, vb, info, wsum_arrows,
                                   out_dot, out_lbl), run_time=0.4))
        self.play(
            self.q_vecs.animate(run_time=0.35).set_opacity(1.0),
            self.k_vecs.animate(run_time=0.35).set_opacity(1.0),
            self.col_lbls[0].animate(run_time=0.35).set_opacity(1.0),
            self.col_lbls[1].animate(run_time=0.35).set_opacity(1.0),
        )

    def s9_output(self):
        title = sec_title(self, "Contextual Representation", color=C_GLOW)

        self.play(
            FadeOut(safe_vg(self.q_vecs, self.k_vecs,
                            self.v_vecs, self.col_lbls), run_time=0.5)
        )

        out_vecs = VGroup()
        out_lbls = VGroup()
        for i, (word, color) in enumerate(zip(WORDS, TOKEN_COLORS)):
            v = vec_box(C_GLOW, rows=4, cell=0.42)
            x = -3.0 + i * 3.0
            v.move_to([x, -0.2, 0])

            gb = SurroundingRectangle(v[0], color=color,
                                       buff=0.07, stroke_width=2.0)

            wl = Text(word, font="Fira Code Bold", color=color,
                      font_size=26, weight=BOLD)
            wl.next_to(v, UP, buff=0.22)

            el = MathTex(fr"e'_{{{i+1}}}", color=C_GLOW, font_size=26)
            el.next_to(v, DOWN, buff=0.18)

            out_vecs.add(VGroup(v, gb))
            out_lbls.add(VGroup(wl, el))

        ctx_note = Text("Context-aware — each vector now contains",
                        font="Fira Code", color=C_DIM, font_size=22)
        ctx_note2 = Text("information from ALL tokens in the sequence.",
                         font="Fira Code", color=C_DIM, font_size=22)
        ctx_note.move_to(DOWN * 2.5)
        ctx_note2.next_to(ctx_note, DOWN, buff=0.14)

        self.play(
            LaggedStart(*[FadeIn(v, scale=0.85, run_time=0.5)
                          for v in out_vecs], lag_ratio=0.20)
        )
        self.play(
            LaggedStart(*[FadeIn(l, run_time=0.35)
                          for l in out_lbls], lag_ratio=0.20)
        )
        self.play(FadeIn(ctx_note,  run_time=0.5),
                  FadeIn(ctx_note2, run_time=0.5))
        self.wait(1.2)

        self.play(FadeOut(safe_vg(title, out_vecs, out_lbls,
                                   ctx_note, ctx_note2), run_time=0.5))

    def s10_summary(self):
        title = sec_title(self, "Summary", color=C_TEXT)

        rows = [
            ("Query  (Q)", "→", "What to look for ",   C_Q),
            ("Key    (K)", "→", "What is available",  C_K),
            ("Value  (V)", "→", "What is taken",      C_V),
        ]

        row_groups = VGroup()
        for term, arrow, meaning, color in rows:
            t1 = Text(term,   font="Fira Code Bold", color=color,
                      font_size=34, weight=BOLD)
            t2 = Text(arrow,  font="Fira Code", color=C_DIM,   font_size=30)
            t3 = Text(meaning,font="Fira Code", color=C_TEXT,  font_size=30)
            row = VGroup(t1, t2, t3).arrange(RIGHT, buff=0.35)
            row_groups.add(row)

        row_groups.arrange(DOWN, buff=0.55, aligned_edge=LEFT)
        row_groups.move_to(ORIGIN + DOWN * 0.2)

        for rg in row_groups:
            self.play(FadeIn(rg, shift=RIGHT * 0.15, run_time=0.50))
            self.wait(0.25)

        for rg, (_, _, _, col) in zip(row_groups, rows):
            box = SurroundingRectangle(rg, color=col, buff=0.14,
                                        stroke_width=1.8, corner_radius=0.10)
            self.play(Create(box, run_time=0.28))
            self.play(FadeOut(box, run_time=0.22))

        self.wait(1.5)

        formula = MathTex(
            r"\text{Attention}(Q,K,V)="
            r"\text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
            color=C_TEXT, font_size=40
        )
        formula.next_to(row_groups, DOWN, buff=0.55)

        glow_f = VGroup()
        for i in (3, 2, 1):
            gl = formula.copy().set_color(C_GLOW)
            gl.set_opacity(0.13 * i / 3).scale(1 + 0.011 * i)
            glow_f.add(gl)

        self.play(
            LaggedStart(
                FadeIn(glow_f,   run_time=0.7),
                Write(formula,   run_time=1.1),
                lag_ratio=0.2
            )
        )
        self.wait(2.0)

        black = Rectangle(width=config.frame_width, height=config.frame_height,
                           fill_color=BLACK, fill_opacity=1, stroke_opacity=0)
        self.play(FadeIn(black, run_time=0.6))
        self.wait(0.3)