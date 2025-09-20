from manim import (
    Scene, Text, RoundedRectangle, Arrow,
    Write, Create, UP, DOWN, RIGHT,
    GREEN, ORANGE, YELLOW, RED, PURPLE, ORIGIN
)

class GatedAttentionFlow(Scene):
    def construct(self):
        """Bottom-to-top vertical animation for gated attention (7 steps)."""

        # Style constants
        box_w = 4.2
        box_h = 0.85
        corner = 0.18
        gap = 1.1
        arrow_stroke = 4
        font_box = 22
        font_step = 20
        font_formula = 26

        # Bottom anchor
        base_y = DOWN * 3.2

        def make_step_box(y_level, text, color):
            box = RoundedRectangle(width=box_w, height=box_h, corner_radius=corner, color=color).move_to(ORIGIN + y_level)
            label = Text(text, font_size=font_box).move_to(box.get_center())
            return box, label

        # Step 1: Introduce L_g
        step1_box, step1_label = make_step_box(base_y, "1. Linear Layer L_g", ORANGE)
        step1_formula = Text("L_g: d_model -> heads*head_dim", font_size=font_step).next_to(step1_box, RIGHT, buff=0.4)
        self.play(Create(step1_box), Write(step1_label))
        self.play(Write(step1_formula))
        self.wait(0.3)

        # Step 2: Q_g = X @ L_g
        step2_box, step2_label = make_step_box(base_y + UP * (gap), "2. Compute Q_g", ORANGE)
        formula2 = Text("Q_g = X · L_g", font_size=font_formula).next_to(step2_box, RIGHT, buff=0.4)
        arrow1 = Arrow(step1_box.get_top(), step2_box.get_bottom(), buff=0.1, stroke_width=arrow_stroke)
        self.play(Create(arrow1), Create(step2_box), Write(step2_label))
        self.play(Write(formula2))
        self.wait(0.3)

        # Step 3: Standard projections Q,K,V
        step3_box, step3_label = make_step_box(base_y + UP * (gap*2), "3. Project Q,K,V", GREEN)
        formula3 = Text("Q = XW_Q, K = XW_K, V = XW_V", font_size=font_step).next_to(step3_box, RIGHT, buff=0.4)
        arrow2 = Arrow(step2_box.get_top(), step3_box.get_bottom(), buff=0.1, stroke_width=arrow_stroke)
        self.play(Create(arrow2), Create(step3_box), Write(step3_label))
        self.play(Write(formula3))
        self.wait(0.3)

        # Step 4: Dot-product attention computation (scores)
        step4_box, step4_label = make_step_box(base_y + UP * (gap*3), "4. Scaled Dot-Prod", YELLOW)
        formula4 = Text("Scores = QK^T / √d_h", font_size=font_formula).next_to(step4_box, RIGHT, buff=0.4)
        arrow3 = Arrow(step3_box.get_top(), step4_box.get_bottom(), buff=0.1, stroke_width=arrow_stroke)
        self.play(Create(arrow3), Create(step4_box), Write(step4_label))
        self.play(Write(formula4))
        self.wait(0.3)

        # Step 5: attention_output = softmax(scores)V
        step5_box, step5_label = make_step_box(base_y + UP * (gap*4), "5. Attention Output", YELLOW)
        formula5 = Text("Attn = softmax(Scores) · V", font_size=font_formula).next_to(step5_box, RIGHT, buff=0.4)
        arrow4 = Arrow(step4_box.get_top(), step5_box.get_bottom(), buff=0.1, stroke_width=arrow_stroke)
        self.play(Create(arrow4), Create(step5_box), Write(step5_label))
        self.play(Write(formula5))
        self.wait(0.3)

        # Step 6: Gating: gated_attention = Attn * sigmoid(Q_g)
        step6_box, step6_label = make_step_box(base_y + UP * (gap*5), "6. Apply Gating", RED)
        formula6 = Text("Gated = Attn * σ(Q_g)", font_size=font_formula).next_to(step6_box, RIGHT, buff=0.4)
        arrow5 = Arrow(step5_box.get_top(), step6_box.get_bottom(), buff=0.1, stroke_width=arrow_stroke)
        self.play(Create(arrow5), Create(step6_box), Write(step6_label))
        self.play(Write(formula6))
        self.wait(0.3)

        # Step 7: final_output = gated_attention @ O
        step7_box, step7_label = make_step_box(base_y + UP * (gap*6), "7. Final Output", PURPLE)
        formula7 = Text("Output = Gated · W_O", font_size=font_formula).next_to(step7_box, RIGHT, buff=0.4)
        arrow6 = Arrow(step6_box.get_top(), step7_box.get_bottom(), buff=0.1, stroke_width=arrow_stroke)
        self.play(Create(arrow6), Create(step7_box), Write(step7_label))
        self.play(Write(formula7))
        self.wait(0.6)

        summary = Text("Gated Attention: Adaptive modulation improves context handling", font_size=24).next_to(step7_box, UP, buff=0.8)
        self.play(Write(summary))
        self.wait(2)
