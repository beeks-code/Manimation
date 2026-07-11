from manim import *

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


class SelfAttentionFlowClean(Scene):
    def construct(self):

        sentence_text = "I Like Maths"
        input_box = Rectangle(width=4, height=1, color=C_ACCENT, fill_opacity=0.2)
        input_label = Text(sentence_text, font="Fira Code", font_size=30, color=C_TEXT)
        input_group = VGroup(input_box, input_label).to_edge(UP,buff=0.2).shift(LEFT*6)
        input_label.move_to(input_box.get_center())
        self.play(FadeIn(input_group))
        self.wait(0.5)

        steps = [
            "Tokenization",
            "Word Embedding",
            "Compute Q, K, V",
            "Q·Kᵀ Multiplication",
            "Scaling & Softmax",
            "Final Output"
        ]

        # Create rectangles and labels
        step_boxes = VGroup()
        step_labels = VGroup()
        for step in steps:
            rect = Rectangle(width=4, height=1, color=C_GLOW, fill_opacity=0.2)
            label = Text(step, font="Fira Code", font_size=30, color=C_TEXT)
            step_boxes.add(rect)
            step_labels.add(label)

        spacing_x = 6
        spacing_y = 3

        step_boxes[0].move_to(input_group.get_center() + RIGHT*spacing_x)  # Tokenization
        step_boxes[1].next_to(step_boxes[0], RIGHT*spacing_x)             # Word Embedding
        step_boxes[2].next_to(step_boxes[1], DOWN*spacing_y)              # Compute Q,K,V
        step_boxes[3].next_to(step_boxes[2], LEFT*spacing_x)              
        step_boxes[4].next_to(step_boxes[3], LEFT*spacing_x)              # Scaling & Softmax
        step_boxes[5].next_to(step_boxes[4], DOWN*spacing_y)              # Final Output


        for rect, label in zip(step_boxes, step_labels):
            label.move_to(rect.get_center())

        self.play(
            LaggedStart(
                *[FadeIn(rect) for rect in step_boxes],
                *[FadeIn(label) for label in step_labels],
                lag_ratio=0.2
            )
        )
        self.wait(0.5)

        arrows = VGroup()

        arrows.add(Arrow(input_group.get_right(), step_boxes[0].get_left(), color=C_ACCENT, buff=0.2))
        arrows.add(Arrow(step_boxes[0].get_right(), step_boxes[1].get_left(), color=C_ACCENT, buff=0.2))
        arrows.add(Arrow(step_boxes[1].get_bottom(), step_boxes[2].get_top(), color=C_ACCENT, buff=0.2))
        arrows.add(Arrow(step_boxes[2].get_left(), step_boxes[3].get_right(), color=C_ACCENT, buff=0.2))
        arrows.add(Arrow(step_boxes[3].get_left(), step_boxes[4].get_right(), color=C_ACCENT, buff=0.2))
        arrows.add(Arrow(step_boxes[4].get_bottom(), step_boxes[5].get_top(), color=C_ACCENT, buff=0.2))

        self.play(LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.3))
        self.wait(1)

        for arrow in arrows:
            self.play(arrow.animate.set_color(C_ACCENT).scale(1.05), run_time=0.3)
            self.play(arrow.animate.set_color(C_ACCENT), run_time=0.2)