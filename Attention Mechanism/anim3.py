from manim import *

BG       = "#0f172a"
C_QUERY  = "#3b82f6"   # blue   → Q
C_KEY    = "#22c55e"   # green  → K
C_VALUE  = "#f97316"   # orange → V
C_GLOW   = "#a78bfa"   # purple-glow
C_TEXT   = "#e2e8f0"
C_DIM    = "#475569"
C_ACCENT = "#f472b6"   # pink accent

TOKEN_COLORS = [C_QUERY, C_GLOW, C_ACCENT]

config.background_color = BG
config.pixel_height = 1080
config.pixel_width  = 1920
config.frame_rate   = 60

def make_grid(rows=3, cols=3, size=0.6, color=WHITE):
    """Return a simple square grid as VGroup"""
    grid = VGroup()
    for i in range(rows):
        for j in range(cols):
            square = Square(side_length=size, color=color)
            square.move_to(RIGHT * (j - (cols-1)/2) * (size+0.05) +
                           UP * ((rows-1)/2 - i) * (size+0.05))
            grid.add(square)
    return grid

class HeaderDemo(Scene):
    def construct(self):
        title = Text("Self-Attention",
                     font="Fira Code", color=C_TEXT,
                     font_size=50, weight=BOLD)
        title.to_edge(UP, buff=0.2)

        # Glow layers
        glow_layers = VGroup()
        for i in (3, 2, 1):
            gl = title.copy().set_color(C_GLOW)
            gl.set_opacity(0.18 * i / 3).scale(1 + 0.015 * i)
            glow_layers.add(gl)

        # Animate glow + title
        self.play(
            LaggedStart(
                FadeIn(glow_layers, run_time=0.9),
                Write(title, run_time=1.1),
                lag_ratio=0.2
            )
        )

        sweep = Rectangle(width=0.3, height=1.2,
                          fill_color=WHITE, fill_opacity=0.08,
                          stroke_opacity=0)
        sweep.move_to(title.get_left() + LEFT * 0.5)
        self.play(sweep.animate(rate_func=smooth, run_time=0.9)
                      .move_to(title.get_right() + RIGHT * 0.5))
        self.remove(sweep)
        self.wait(0.5)

        grid = make_grid()
        self.play(FadeIn(grid, run_time=1.5))
        self.play(
            grid.animate(rate_func=linear, run_time=3)
                .shift(RIGHT * 0.25 + UP * 0.12)
        )

        words_raw = ["I", "like", "Maths"]
        word_mobs = VGroup(*[
            Text(w, font="Fira Code", color=c, font_size=64)
            for w, c in zip(words_raw, TOKEN_COLORS)
        ])
        word_mobs.arrange(RIGHT, buff=0.6)

        for wm in word_mobs:
            self.play(FadeIn(wm, shift=UP * 0.18, run_time=0.55, rate_func=smooth))
            self.wait(0.12)
        self.wait(0.3)