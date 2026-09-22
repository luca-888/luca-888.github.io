# GC lifecycle imagegen trial

Status: candidate used in the local article preview; not a user-approved site-wide style.
Tool: built-in image_gen, one generation and one edit.
Asset: lifecycle-imagegen-v2.png

## Editorial purpose

Explain the difference between keeping an internal activation until backward and freeing it then recomputing. This figure replaces multiple low-value diagrams. Exact formulas remain KaTeX, and measured data remains ECharts and tables. The retained boundary input is explicitly visible. No measured timing or memory proportion is implied.

## Evaluation

The first render had a dark background and glow that made its labels unreadable. The edit fixes contrast and exposes the intended timeline. The final result is useful for a conceptual overview, but its value comes from limiting the scope to a single comparison, not from raster generation alone. It is not a benchmark, an exact operator trace, or an approved reusable style.

## Initial prompt

Use case: scientific-educational.
Asset type: one candidate explanatory illustration inside a technical article about Gradient Checkpointing. This is a visual prototype to judge whether a reader understands the mechanism immediately, not a cover or decorative hero.
Primary request: Explain ONLY the lifetime of an internal activation, with a very clear two-lane time comparison: Eager keeps that internal activation until backward; GC frees it after forward and recomputes it when backward needs it. Saved boundary inputs remain available under GC; do not imply every tensor disappears.
Composition: landscape 1536x1024, clean pure white background, flat precise editorial infographic with abundant whitespace. Three aligned time positions left-to-right labeled "Forward", "Later layers", "Backward". Two spacious horizontal lanes labeled "Eager" and "GC". In the Eager lane, one solid blue-grey ribbon runs from Forward through Later layers until Backward, labeled "Keep activation". In the GC lane, show only a short blue-grey ribbon during Forward, a large EMPTY gap during Later layers (very faint dotted guide only), and a short warm-orange ribbon at Backward labeled "Recompute". Below the GC lane draw a separate thin muted-green continuous line spanning the interval, clearly labeled "Keep boundary input". The visual emphasis is the long retained ribbon versus the large gap and small recomputation at the end. At the end of both lanes show the same small understated check mark, for use by backward, not extra copies of a calculation graph. Time moves right. Distinguish activation ribbons from the thin boundary-input line.
Typography: exact short English labels only, large readable dark grey sans-serif, no Chinese, no mathematical equations, no tiny explanatory text. Main title top-left "Save less. Recompute later." Small bottom note "Schematic — not measured proportions".
Visual hierarchy: first see the long ribbon vs gap, second the orange recomputation, third the thin retained-input line. No enclosing cards, boxes, dashboards, panels, extra memory charts, decorative nodes, shadows, gradients, 3D, textures, glowing circuits, or badges. Do not show invented memory percentages or timings. Use simple crisp shapes, align columns and avoid any crossing lines. Every mark must explain the mechanism. Scientific accuracy and immediate readability over decoration.

## Revision prompt

Edit this GC diagram. Preserve the two-row scientific meaning, labels, alignment and layout. The sole revision is visual finish and legibility: make the entire background solid opaque pure white, no transparency anywhere. Remove ALL glow, blur, shadows, dark vignette and gradients completely. Make all text crisp dark charcoal (#24282b), fully readable. Make the retained-activation ribbons flat light blue-grey with dark lettering, the Recompute ribbon flat pale orange with dark lettering, and boundary-input line flat muted green. Pure flat editorial illustration, no effects. Do not move or add any elements. Keep every existing label spelled exactly, including title, Eager, GC, Forward, Later layers, Backward, Keep activation, Recompute, Keep boundary input, and bottom Schematic note. Output an opaque white-background readable diagram.
