"""Generate docs/assets/graph.svg — an animated, dark-theme hero of a Cartograph knowledge graph.

Nodes scale in (staggered), edges draw, then a repeating 'query pulse' lights up the relevant nodes —
the core idea (you type, your relevant work lights up) shown as motion. Pure SMIL SVG: it animates on
GitHub when it can, and degrades to a complete static graph if not. No build deps."""
from __future__ import annotations

import math
import random
from pathlib import Path

W, H = 1200, 440
BG = "#0d1117"
random.seed(7)

# field -> color (dark-theme friendly)
FIELDS = {
    "ml": "#5ea0ff", "web": "#7ee787", "quant": "#ffa657", "data": "#d2a8ff",
    "hpc": "#ff7b72", "agents": "#79c0ff", "library": "#56d364", "devops": "#e3b341",
}
FK = list(FIELDS)

# lay out ~18 nodes in loose per-field clusters across the canvas
CLUSTERS = {
    "ml": (250, 150), "web": (520, 110), "quant": (840, 150), "data": (980, 300),
    "hpc": (140, 320), "agents": (430, 330), "library": (700, 300), "devops": (1040, 120),
}
nodes = []
for f, (cx, cy) in CLUSTERS.items():
    for _ in range(random.randint(2, 3)):
        ang = random.uniform(0, 2 * math.pi)
        r = random.uniform(18, 70)
        nodes.append({"x": cx + r * math.cos(ang), "y": cy + r * math.sin(ang),
                      "f": f, "rad": random.uniform(6, 13)})
N = len(nodes)

# edges: within-cluster + a few cross-cluster
edges = []
for i in range(N):
    for j in range(i + 1, N):
        same = nodes[i]["f"] == nodes[j]["f"]
        dx, dy = nodes[i]["x"] - nodes[j]["x"], nodes[i]["y"] - nodes[j]["y"]
        d = math.hypot(dx, dy)
        if (same and d < 95) or (not same and d < 130 and random.random() < 0.12):
            edges.append((i, j))

# nodes that "light up" on the query pulse (a relevant subset)
lit = sorted(random.sample(range(N), 5))


def build() -> str:
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="Segoe UI, system-ui, Arial, sans-serif">']
    s.append('<defs>'
             '<radialGradient id="glow" cx="50%" cy="50%" r="50%">'
             '<stop offset="0%" stop-color="#ffd166" stop-opacity="0.9"/>'
             '<stop offset="100%" stop-color="#ffd166" stop-opacity="0"/></radialGradient>'
             '<filter id="soft"><feGaussianBlur stdDeviation="2.2"/></filter></defs>')
    s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')

    # edges (draw-in via stroke-dashoffset)
    for k, (i, j) in enumerate(edges):
        a, b = nodes[i], nodes[j]
        ln = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
        beg = 0.9 + k * 0.018
        s.append(
            f'<line x1="{a["x"]:.0f}" y1="{a["y"]:.0f}" x2="{b["x"]:.0f}" y2="{b["y"]:.0f}" '
            f'stroke="#30363d" stroke-width="1.4" stroke-dasharray="{ln:.0f}" stroke-dashoffset="0">'
            f'<animate attributeName="stroke-dashoffset" from="{ln:.0f}" to="0" dur="0.7s" '
            f'begin="{beg:.2f}s" fill="freeze"/></line>')

    # nodes (scale in, staggered, then gentle continuous pulse)
    for idx, n in enumerate(nodes):
        c = FIELDS[n["f"]]
        beg = 0.05 + idx * 0.06
        s.append(f'<g transform="translate({n["x"]:.0f},{n["y"]:.0f})">'
                 f'<circle r="{n["rad"]:.0f}" fill="{c}" opacity="0.92">'
                 f'<animate attributeName="opacity" from="0" to="0.92" dur="0.4s" begin="{beg:.2f}s" fill="freeze"/>'
                 f'<animateTransform attributeName="transform" type="scale" from="0" to="1" dur="0.45s" '
                 f'begin="{beg:.2f}s" fill="freeze" calcMode="spline" keySplines="0.2 0.8 0.2 1" keyTimes="0;1"/>'
                 f'<animate attributeName="r" values="{n["rad"]:.0f};{n["rad"]+1.5:.0f};{n["rad"]:.0f}" '
                 f'dur="{3.5+idx%3}s" begin="{beg+0.5:.2f}s" repeatCount="indefinite"/></circle></g>')

    # the repeating QUERY PULSE: a ring expands from the prompt bar, lit nodes flash
    px, py = 600, 400
    s.append(f'<circle cx="{px}" cy="{py}" r="6" fill="none" stroke="#ffd166" stroke-width="2" opacity="0">'
             f'<animate attributeName="r" values="6;520" dur="2.6s" begin="3s;pulse.end+1.6s" id="pulse"/>'
             f'<animate attributeName="opacity" values="0.8;0" dur="2.6s" begin="3s;pulse.end+1.6s"/></circle>')
    for i in lit:
        n = nodes[i]
        s.append(f'<circle cx="{n["x"]:.0f}" cy="{n["y"]:.0f}" r="{n["rad"]+10:.0f}" fill="url(#glow)" opacity="0">'
                 f'<animate attributeName="opacity" values="0;0.95;0" dur="1.1s" '
                 f'begin="3.6s;pulse.end+2.2s" repeatCount="1"/>'
                 f'<animate attributeName="opacity" values="0;0.95;0" dur="1.1s" '
                 f'begin="pulse.end+2.2s" repeatCount="indefinite"/></circle>')

    # title + tagline + prompt caption
    s.append(f'<text x="60" y="74" font-size="46" font-weight="800" fill="#e7e9ee">🗺 Cartograph</text>')
    s.append(f'<text x="62" y="106" font-size="18" fill="#9aa3b2">your work, mapped into one graph — '
             f'and plugged into your AI agents</text>')
    s.append(f'<g opacity="1"><animate attributeName="opacity" from="0" to="1" dur="0.6s" begin="2.6s" fill="freeze"/>'
             f'<rect x="{px-220}" y="{py-20}" width="440" height="34" rx="17" fill="#161b22" stroke="#30363d"/>'
             f'<text x="{px}" y="{py+3}" text-anchor="middle" font-size="14" fill="#9aa3b2">'
             f'❯ "how did I handle auth?" — your relevant work lights up</text></g>')
    s.append('</svg>')
    return "\n".join(s)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "docs" / "assets" / "graph.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(), encoding="utf-8")
    print("wrote", out)
