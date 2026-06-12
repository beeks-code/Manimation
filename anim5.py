from manim import *

# ─────────────────────────── THEME COLORS ─────────────────────────────
BG       = "#0f172a"
C_QUERY  = "#3b82f6"   # blue
C_KEY    = "#22c55e"   # green
C_VALUE  = "#f97316"   # orange
C_GLOW   = "#a590e4"   # purple-glow
C_TEXT   = "#e2e8f0"
C_DIM    = "#475569"
C_ACCENT = "#f472b6"

config.background_color = BG

# ───────────────────────────── ANIMATION SCENE ─────────────────────────
class VectorEmbeddingScene(Scene):
    def construct(self):
        # ------------------- TITLE -------------------
        title = Text(
            "Vector Embedding",
            font="Fira Code Bold",
            font_size=52,
            color=C_TEXT
        )
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title, shift=DOWN * 0.3, run_time=1.0))
        self.wait(0.5)

        # ------------------- DEFINITION -------------------
        definition = Text(
            "Vector embeddings are numerical representations\nof data such as text, images, or audio in n-dim vectors",
            font="Fira Code",
            font_size=26,
            color=C_GLOW,
            line_spacing=1.2
        )
        definition.next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(definition, shift=UP * 0.2, run_time=1.2))
        self.wait(2.0)
        self.play(FadeOut(definition, run_time=1.0))

        # ------------------- PULL DOWN TITLE & RECTANGLE -------------------
        self.play(title.animate.shift(DOWN * 0.5), run_time=1.0)

        title_rect = RoundedRectangle(
            corner_radius=0.3,
            width=title.width + 1.0,
            height=title.height + 0.5,
            fill_color=C_ACCENT,
            fill_opacity=0.2,
            stroke_color=C_ACCENT,
            stroke_width=2
        ).move_to(title.get_center())
        self.play(FadeIn(title_rect, scale=0.8, run_time=1.0))
        self.wait(0.5)

        # ------------------- TWO ARROWS DOWN -------------------
        arrow_offset = 3.0   # increased horizontal offset for spacing
        arrow_length = 1.6   # shorter so arrows stop above rectangles

        arrow_left = Arrow(
            start=title_rect.get_bottom(),
            end=title_rect.get_bottom() + DOWN * arrow_length + LEFT * arrow_offset,
            buff=0,
            color=C_GLOW,
            stroke_width=3
        )
        arrow_right = Arrow(
            start=title_rect.get_bottom(),
            end=title_rect.get_bottom() + DOWN * arrow_length + RIGHT * arrow_offset,
            buff=0,
            color=C_GLOW,
            stroke_width=3
        )

        # Glow effect
        glow_left = arrow_left.copy().set_stroke(width=10, opacity=0.15)
        glow_right = arrow_right.copy().set_stroke(width=10, opacity=0.15)

        self.play(
            GrowArrow(arrow_left),
            GrowArrow(arrow_right),
            FadeIn(glow_left),
            FadeIn(glow_right),
            run_time=1.5,
        )
        self.wait(0.5)

        # ------------------- SUB-RECTANGLES FOR EMBEDDING TYPES -------------------
        rect_width = 3.0
        rect_height = 1.2

        # Left rectangle: Static Embedding
        rect_left = RoundedRectangle(
            corner_radius=0.2,
            width=rect_width,
            height=rect_height,
            fill_color=C_ACCENT,
            fill_opacity=0.2,
            stroke_color=C_TEXT,
            stroke_width=2
        ).next_to(arrow_left.get_end(), DOWN, buff=0.05)  # just below arrow tip

        text_left = Text(
            "Static Embedding",
            font="Fira Code Bold",
            font_size=24,
            color=C_TEXT
        ).move_to(rect_left.get_center())

        # Right rectangle: Contextual Embedding
        rect_right = RoundedRectangle(
            corner_radius=0.2,
            width=rect_width+1,
            height=rect_height,
            fill_color=C_ACCENT,
            fill_opacity=0.2,
            stroke_color=C_TEXT,
            stroke_width=2
        ).next_to(arrow_right.get_end(), DOWN, buff=0.05)  # just below arrow tip

        text_right = Text(
            "Contextual Embedding",
            font="Fira Code Bold",
            font_size=24,
            color=C_TEXT
        ).move_to(rect_right.get_center())

        # Optional pop-in scaling effect
        self.play(
            LaggedStart(
                FadeIn(rect_left, scale=0.5),
                FadeIn(text_left, scale=0.5),
                FadeIn(rect_right, scale=0.5),
                FadeIn(text_right, scale=0.5),
                lag_ratio=0.2,
                run_time=1.2
            )
        )

        # Slight glow effect on rectangles
        glow_rect_left = rect_left.copy().set_stroke(width=8, opacity=0.12, color=C_GLOW)
        glow_rect_right = rect_right.copy().set_stroke(width=8, opacity=0.12, color=C_GLOW)
        self.add(glow_rect_left, glow_rect_right)

        self.wait(2)
