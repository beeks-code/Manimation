from manim import *
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

class SelfAttentionCurved(Scene):
    def construct(self):
        title_text = Text(
            "What Self Attention does",
            font="Fira Code",
            color=C_TEXT,
            font_size=48
        )
        question_mark = Text(
            "?",
            font="Fira Code",
            color=C_ACCENT,
            font_size=52
        )
        title = VGroup(title_text, question_mark).arrange(RIGHT, buff=0.2)
        title.to_edge(UP)

        self.play(Write(title_text), FadeIn(question_mark, scale=0.5))
        self.wait(0.5)

        for _ in range(3):
            self.play(question_mark.animate.set_opacity(0.2), run_time=0.5)
            self.play(question_mark.animate.set_opacity(1), run_time=0.5)

        words = ["I", "Like", "Maths"]
        word_mobs = VGroup()

        for i, word in enumerate(words):
            txt = Text(
                word,
                font="Fira Code",
                color=TOKEN_COLORS[i % len(TOKEN_COLORS)],
                font_size=44
            )
            word_mobs.add(txt)

        word_mobs.arrange(RIGHT, buff=2)
        word_mobs.shift(DOWN)

        self.play(LaggedStart(*[FadeIn(w, shift=UP) for w in word_mobs], lag_ratio=0.3))
        self.wait(0.5)

        centers = [t.get_center() for t in word_mobs]
        attn_arcs = VGroup()
        pairs = [(0, 1), (0, 2), (1, 2)]

        for i, j in pairs:
            arc_1 = CurvedArrow(
                centers[i] + DOWN * 0.4,
                centers[j] + DOWN * 0.4,
                angle=TAU / 6,
                color=C_GLOW,
                stroke_opacity=0.6,
                stroke_width=2.5,
                tip_length=0.2
            )
            arc_2 = CurvedArrow(
                centers[j] + DOWN * 0.4,
                centers[i] + DOWN * 0.4,
                angle=-TAU / 6,
                color=C_GLOW,
                stroke_opacity=0.4,
                stroke_width=2.0,
                tip_length=0.18
            )
            attn_arcs.add(arc_1, arc_2)

        self.play(
            LaggedStart(*[Create(a, run_time=0.4) for a in attn_arcs], lag_ratio=0.15)
        )
        self.wait(0.5)

        self.play(attn_arcs.animate.set_color(C_ACCENT), run_time=0.6)
        self.play(attn_arcs.animate.set_color(C_GLOW), run_time=0.6)
        self.wait(1)