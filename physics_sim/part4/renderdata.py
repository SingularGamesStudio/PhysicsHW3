def v_to_list(v):
    return [float(v.x), float(v.y), float(v.z)]

def build_part4_render_data(results, bodies0, columns=2, radius=8.0):
    palette = [
        "#8fbcd4", "#d49f8f", "#9fd48f", "#d4c78f", "#b39fd4",
        "#8fd4c7", "#d48fb1", "#9fb4d4", "#c1d48f", "#d4ab8f",
    ]

    body_half_extents = [
        v_to_list(b.shape.half_extents)
        for b in bodies0
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
                    "half": body_half_extents[i],
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
        "title": "Part 4 — many mixed-size rigid bodies with XPBD friction",
        "columns": int(columns),
        "sims": sims,
    }