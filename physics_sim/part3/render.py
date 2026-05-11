from ..common import Vec3, v_to_list, vec3


def build_part3_render_data(results, half_extents, columns=2, radius=8.0):
    half_extents = half_extents if isinstance(half_extents, Vec3) else vec3(*half_extents)

    palette = [
        "#8fbcd4", "#d49f8f", "#9fd48f", "#d4c78f", "#b39fd4",
        "#8fd4c7", "#d48fb1", "#9fb4d4", "#c1d48f", "#d4ab8f",
    ]

    sims = []
    for _, sim in results.items():
        frames = []
        for s in sim["samples"]:
            bodies = []
            for i, b in enumerate(s["bodies"]):
                bodies.append({
                    "pos": b["x"],
                    "q": b["q"],
                    "half": v_to_list(half_extents),
                    "color": palette[i % len(palette)],
                })

            frames.append({
                "bodies": bodies,
                "props": {
                    "time": round(float(s["t"]), 4),
                    "contacts": int(s["contacts"]),
                    "manifolds": int(s["manifolds"]),
                    "pairs": int(s["candidate_pairs"]),
                    "E": round(float(s["E"]), 6),
                    "hz": round(float(s["hz"]), 2),
                },
                "graph": ["contacts", "E", "hz"],
            })

        sims.append({
            "caption": sim["label"],
            "dt": float(sim["dt"]),
            "radius": float(radius),
            "frames": frames,
        })

    return {
        "title": "Part 3 — small 3D box pile with full contacts",
        "columns": int(columns),
        "sims": sims,
    }
