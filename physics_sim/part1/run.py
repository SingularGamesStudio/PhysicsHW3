import math

from pyglm import glm

from ..common import Vec3, v_to_list, vec3, q_to_list
from .solver import Part1Case, Part1Solver


def build_part1_render_data(results, half_extents, columns=2, radius=None):
    half_extents = half_extents if isinstance(half_extents, Vec3) else vec3(*half_extents)
    if radius is None:
        radius = float(max(4.0, 3.0 * glm.length(half_extents)))
    sims = []
    for _, sim in results.items():
        frames = []
        for s in sim["samples"]:
            L = s["L_world"]
            L0 = s["L0_world"]
            l_norm = math.sqrt(L[0] * L[0] + L[1] * L[1] + L[2] * L[2])
            l0_norm = math.sqrt(L0[0] * L0[0] + L0[1] * L0[1] + L0[2] * L0[2])
            frames.append({
                "bodies": [
                    {
                        "pos": [0.0, 0.0, 0.0],
                        "q": s["q"],
                        "half": v_to_list(half_extents),
                        "color": "#8fbcd4",
                    }
                ],
                "props": {
                    "time": round(float(s["t"]), 4),
                    "E": round(float(s["E"]), 6),
                    "E0": round(float(s["E0"]), 6),
                    "|L|": round(float(l_norm), 6),
                    "|L0|": round(float(l0_norm), 6),
                },
                "graph": ["E"],
            })
        sims.append({
            "caption": sim["label"],
            "dt": float(sim["dt"]),
            "radius": radius,
            "frames": frames,
        })
    return {
        "title": "Part 1",
        "columns": int(columns),
        "sims": sims,
    }


def simulate_part1(body, dt=1.0 / 240.0, steps=1800):
    results = {}
    for case in Part1Case:
        solver = Part1Solver(body, dt=dt, case=case, newton_iters=10)
        samples = [snapshot_part1(solver, 0.0)]
        for i in range(steps):
            solver.step()
            samples.append(snapshot_part1(solver, (i + 1) * dt))
        results[case.name] = {
            "label": solver_label(case),
            "dt": dt,
            "samples": samples,
        }
    return results


def snapshot_part1(solver: Part1Solver, t):
    L = solver.body.L_world()
    return {
        "t": float(t),
        "q": q_to_list(solver.body.state.q),
        "w_body": v_to_list(solver.body.state.w_body),
        "L_world": v_to_list(L),
        "E": float(solver.body.kinetic_energy_rot()),
        "E0": float(solver.E0),
        "L0_world": v_to_list(solver.L0_world),
    }


def solver_label(case: Part1Case):
    from .solver import CASE_LABELS

    return CASE_LABELS[case]
