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
 
def glow_color(base, alpha=0.25):
    return ManimColor(base).interpolate(WHITE, alpha)
 
def make_label(text, color=C_TEXT, size=0.45):
    return Text(text, color=color, font_size=int(size * 48)).set_stroke(color, 0.5)
 
def neon_line(start, end, color=C_GLOW, stroke=2):
    return Line(start, end, color=color, stroke_width=stroke,
                stroke_opacity=0.85)
 
 
def make_thought_cloud(center, width=3.8, height=1.6, color=C_GLOW):
    """Build a cartoon thought-cloud from overlapping ellipses."""
    cloud = VGroup()
    specs = [
        (0.0,   0.0,  width*0.55, height*0.72),
        (-width*0.28,  0.05, width*0.42, height*0.58),
        ( width*0.28,  0.05, width*0.42, height*0.58),
        (-width*0.14,  height*0.32, width*0.35, height*0.44),
        ( width*0.14,  height*0.32, width*0.35, height*0.44),
        ( width*0.00,  height*0.38, width*0.28, height*0.38),
    ]
    for (cx, cy, w, h) in specs:
        e = Ellipse(width=w, height=h,
                    fill_color="#1e293b", fill_opacity=0.92,
                    stroke_color=color, stroke_width=2.0)
        e.move_to(center + np.array([cx, cy, 0]))
        cloud.add(e)
 
    # Bubble tail (three diminishing circles)
    tail_positions = [
        center + np.array([-width*0.30, -height*0.52, 0]),
        center + np.array([-width*0.42, -height*0.72, 0]),
        center + np.array([-width*0.50, -height*0.90, 0]),
    ]
    tail_radii = [0.18, 0.12, 0.07]
    for pos, r in zip(tail_positions, tail_radii):
        b = Circle(radius=r,
                   fill_color="#1e293b", fill_opacity=0.92,
                   stroke_color=color, stroke_width=1.8)
        b.move_to(pos)
        cloud.add(b)
    return cloud
 
 
class Scene1_Hook(Scene):
    def construct(self):
        self.camera.background_color = BG
 
        title = Text("How Transformers Understand Language",
                     font_size=38, color=C_GLOW, weight=BOLD)
        title.to_edge(UP, buff=0.35)
 
        underbar = Line(
            title.get_left()  + DOWN*0.08,
            title.get_right() + DOWN*0.08,
            color=C_GLOW, stroke_width=1.5, stroke_opacity=0.45
        )
        self.play(
            FadeIn(title, shift=DOWN*0.4, rate_func=smooth),
            run_time=1.3
        )
        self.play(Create(underbar), run_time=0.6)
        self.wait(0.4)
 
        words = ["The", "animal", "didn't", "cross", "the",
                 "street", "because", "it", "was", "too", "tired"]
 
        word_mobs = VGroup(*[
            Text(w, font_size=32, color=C_DIM)
            for w in words
        ]).arrange(RIGHT, buff=0.20).move_to(ORIGIN + DOWN*0.1)
 
        for wm in word_mobs:
            self.play(FadeIn(wm, shift=UP*0.12,
                             rate_func=smooth), run_time=0.11)
        self.wait(0.35)
 
        animal_mob = word_mobs[1]
        street_mob = word_mobs[5]
        it_mob     = word_mobs[7]
        tired_mob  = word_mobs[10]
 
        self.play(
            animal_mob.animate.set_color(C_QUERY).scale(1.08),
            street_mob.animate.set_color(C_KEY  ).scale(1.08),
            it_mob    .animate.set_color(C_ACCENT).scale(1.15),
            tired_mob .animate.set_color(C_VALUE ).scale(1.08),
            run_time=0.7, rate_func=smooth
        )
 
        pulse_box = SurroundingRectangle(
            it_mob, color=C_ACCENT, corner_radius=0.12,
            buff=0.10, stroke_width=2.5
        )
        self.play(Create(pulse_box), run_time=0.5)
        for _ in range(2):
            self.play(pulse_box.animate.scale(1.12).set_stroke(opacity=0.4),
                      run_time=0.22, rate_func=there_and_back)
        self.wait(0.3)
 
        qmark = Text("?", font_size=52, color=C_ACCENT, weight=BOLD)
        qmark.next_to(it_mob, UP, buff=0.18)
        self.play(FadeIn(qmark, shift=UP*0.2, rate_func=smooth), run_time=0.5)
        self.wait(0.2)
 
        cloud_center = it_mob.get_center() + UP*2.55
        cloud = make_thought_cloud(cloud_center, width=4.2, height=1.7,
                                   color=C_GLOW)
        self.play(
            LaggedStart(
                *[GrowFromCenter(part, rate_func=smooth)
                  for part in cloud],
                lag_ratio=0.06
            ),
            run_time=1.1
        )
 
        option_a = Text("animal  ?", font_size=24,
                        color=C_QUERY, weight=BOLD)
        option_b = Text("street  ?", font_size=24,
                        color=C_KEY,   weight=BOLD)
        vs_txt   = Text("vs", font_size=20, color=C_DIM)
 
        thought_row = VGroup(option_a, vs_txt, option_b)
        thought_row.arrange(RIGHT, buff=0.28)
        thought_row.move_to(cloud_center + UP*0.05)
 
        self.play(
            FadeIn(option_a, shift=RIGHT*0.2, rate_func=smooth),
            run_time=0.5
        )
        self.play(FadeIn(vs_txt), run_time=0.25)
        self.play(
            FadeIn(option_b, shift=LEFT*0.2, rate_func=smooth),
            run_time=0.5
        )
        self.wait(0.4)
 
        # Flicker debate  — options alternate brightness 3×
        for _ in range(3):
            self.play(
                option_a.animate.set_opacity(1.0),
                option_b.animate.set_opacity(0.25),
                run_time=0.22, rate_func=smooth
            )
            self.play(
                option_a.animate.set_opacity(0.25),
                option_b.animate.set_opacity(1.0),
                run_time=0.22, rate_func=smooth
            )
        # Reset both
        self.play(
            option_a.animate.set_opacity(1.0),
            option_b.animate.set_opacity(1.0),
            run_time=0.2
        )
        self.wait(0.3)
 

        arr_animal = CurvedArrow(
            it_mob.get_top() + LEFT*0.05,
            animal_mob.get_top() + UP*0.05,
            color=C_QUERY, stroke_width=2.8,
            angle=TAU / 5
        )
        arr_street = CurvedArrow(
            it_mob.get_bottom() + RIGHT*0.05,
            street_mob.get_bottom() + UP*0.05,
            color=C_KEY, stroke_width=2.8,
            angle=-TAU / 5
        )
 
        self.play(Create(arr_animal), run_time=0.75)
        self.play(Create(arr_street), run_time=0.75)
        self.wait(0.4)

        cross = Line(
            option_b.get_left(), option_b.get_right(),
            color=RED, stroke_width=3
        )
        self.play(
            FadeOut(arr_street),
            option_b.animate.set_opacity(0.3),
            run_time=0.5
        )
        self.play(Create(cross), run_time=0.4)
 

        self.play(
            arr_animal.animate.set_stroke(color=C_QUERY, width=4.5),
            option_a.animate.scale(1.12).set_color(C_QUERY),
            run_time=0.5
        )
 

        check = Text("✓", font_size=28, color=C_QUERY, weight=BOLD)
        check.next_to(option_a, RIGHT, buff=0.12)
        self.play(FadeIn(check, shift=UP*0.1), run_time=0.4)
        self.wait(0.5)
 
        tired_glow = SurroundingRectangle(
            tired_mob, color=C_VALUE, corner_radius=0.1,
            buff=0.08, stroke_width=2
        )
        self.play(Create(tired_glow), run_time=0.4)
 
        connect = DashedLine(
            animal_mob.get_bottom() + DOWN*0.05,
            tired_mob .get_bottom() + DOWN*0.05,
            color=C_GLOW, stroke_width=1.8,
            dash_length=0.12, stroke_opacity=0.7
        )
        self.play(Create(connect), run_time=0.6)
        self.wait(0.5)
 
        caption1 = Text(
            "Humans resolve this instantly.",
            font_size=23, color=C_DIM, slant=ITALIC
        )
        caption2 = Text(
            "Transformers learn to do it too — using Self-Attention.",
            font_size=23, color=C_GLOW, weight=BOLD
        )
        captions = VGroup(caption1, caption2).arrange(DOWN, buff=0.18)
        captions.to_edge(DOWN, buff=0.42)
 
        self.play(FadeIn(caption1, shift=UP*0.15), run_time=0.7)
        self.wait(0.3)
        self.play(FadeIn(caption2, shift=UP*0.15), run_time=0.8)
        self.wait(2.0)
 
        self.play(
            FadeOut(Group(*self.mobjects), rate_func=smooth),
            run_time=1.0
        )