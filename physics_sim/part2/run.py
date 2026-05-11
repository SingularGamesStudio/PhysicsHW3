from pyglm import glm

from ..common import Vec3, v_to_list, vec3
from .scenarios import make_part2_solvers


def build_part2_render_data(results, half_extents, columns=2, radius=None):
    half_extents = half_extents if isinstance(half_extents, Vec3) else vec3(*half_extents)
    if radius is None:
        radius = float(max(5.0, 4.0 * glm.length(half_extents)))
    sims = []
    for _, sim in results.items():
        frames = []
        for s in sim["samples"]:
            bodies = []
            colors = ["#8fbcd4", "#d49f8f"]
            for i, b in enumerate(s["bodies"]):
                bodies.append({
                    "pos": b["x"],
                    "q": b["q"],
                    "half": v_to_list(half_extents),
                    "color": colors[i % len(colors)],
                })
            frames.append({
                "bodies": bodies,
                "props": {
                    "time": round(float(s["t"]), 4),
                    "C": round(float(s["C"]), 6),
                    "lambda": round(float(s["lambda"]), 6),
                    "E": round(float(s["E"]), 6),
                },
                "graph": ["C", "E"],
            })
        sims.append({
            "caption": sim["label"],
            "dt": float(sim["dt"]),
            "radius": radius,
            "frames": frames,
        })
    return {
        "title": "Part 2 — springs and off-center distance constraints",
        "columns": int(columns),
        "sims": sims,
    }

def simulate_part2(body_spring, body_a, body_b, dt=1.0 / 240.0, steps=2400):
    solvers = make_part2_solvers(body_spring, body_a, body_b, dt)
    results = {}
    for case, solver in solvers.items():
        samples = [solver.snapshot(0.0)]
        for i in range(steps):
            solver.step()
            samples.append(solver.snapshot((i + 1) * dt))
        results[case.name] = {
            "label": solver_label(case),
            "dt": dt,
            "samples": samples,
        }
    return results


def solver_label(case):
    from .types import CASE_LABELS

    return CASE_LABELS[case]
