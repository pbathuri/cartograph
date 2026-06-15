"""Render docs/assets/graph.gif — a guaranteed-motion (GIF) version of the Cartograph hero, for viewers
that don't play SMIL SVG (e.g. GitHub's image proxy). Nodes pop in, edges draw, then a repeating query
pulse lights up the relevant nodes. Pure Pillow; build-time only (NOT a runtime dependency).

    pip install pillow && python scripts/gen_hero_gif.py
"""
from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, SS = 1000, 360, 2          # SS = supersample for antialiasing
BG = (13, 17, 23)
random.seed(7)

FIELDS = {"ml": (94, 160, 255), "web": (126, 231, 135), "quant": (255, 166, 87),
          "data": (210, 168, 255), "hpc": (255, 123, 114), "agents": (121, 198, 255),
          "library": (86, 211, 100), "devops": (227, 179, 65)}
CLUSTERS = {"ml": (210, 150), "web": (430, 110), "quant": (700, 140), "data": (820, 250),
            "hpc": (120, 270), "agents": (360, 275), "library": (580, 250), "devops": (870, 110)}
nodes = []
for f, (cx, cy) in CLUSTERS.items():
    for _ in range(random.randint(2, 3)):
        ang, r = random.uniform(0, 2 * math.pi), random.uniform(16, 58)
        nodes.append({"x": cx + r * math.cos(ang), "y": cy + r * math.sin(ang),
                      "c": FIELDS[f], "rad": random.uniform(6, 12), "delay": 0.0})
N = len(nodes)
for i, n in enumerate(nodes):
    n["delay"] = i / N * 0.45        # stagger fraction of the intro
edges = []
for i in range(N):
    for j in range(i + 1, N):
        d = math.hypot(nodes[i]["x"] - nodes[j]["x"], nodes[i]["y"] - nodes[j]["y"])
        same = nodes[i]["c"] == nodes[j]["c"]
        if (same and d < 80) or (not same and d < 115 and random.random() < 0.12):
            edges.append((i, j))
lit = sorted(random.sample(range(N), 5))
PROMPT = (500, 330)


def _font(sz):
    for p in (r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def _font_r(sz):
    for p in (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def ease(t):
    return 1 - (1 - t) ** 3


def frame(p: float) -> Image.Image:
    """p in [0,1) over the whole loop."""
    im = Image.new("RGB", (W * SS, H * SS), BG)
    d = ImageDraw.Draw(im, "RGBA")
    s = SS
    intro = min(1.0, p / 0.40)                    # 0..1 during first 40% of loop
    pulse_p = max(0.0, (p - 0.55) / 0.40)         # 0..1 during the pulse window

    # edges draw in (after their endpoints appear)
    for (i, j) in edges:
        a, b = nodes[i], nodes[j]
        ap = max(0.0, min(1.0, (intro - max(a["delay"], b["delay"])) / 0.25))
        ap = ease(ap)
        if ap <= 0:
            continue
        ex, ey = a["x"] + (b["x"] - a["x"]) * ap, a["y"] + (b["y"] - a["y"]) * ap
        d.line([a["x"] * s, a["y"] * s, ex * s, ey * s], fill=(48, 54, 61, 230), width=int(1.4 * s))

    # query pulse ring + lit glows
    if pulse_p > 0:
        ring_r = pulse_p * 620
        alpha = int(200 * (1 - pulse_p))
        d.ellipse([(PROMPT[0] - ring_r) * s, (PROMPT[1] - ring_r) * s,
                   (PROMPT[0] + ring_r) * s, (PROMPT[1] + ring_r) * s],
                  outline=(255, 209, 102, alpha), width=int(2 * s))
        glow = math.sin(min(1.0, pulse_p * 1.6) * math.pi)        # rise then fall
        for i in lit:
            n = nodes[i]
            for k in range(6, 0, -1):
                rr = (n["rad"] + k * 3.2) * s
                d.ellipse([n["x"] * s - rr, n["y"] * s - rr, n["x"] * s + rr, n["y"] * s + rr],
                          fill=(255, 209, 102, int(22 * glow)))

    # nodes (scale + fade in, gentle breathing after)
    for n in nodes:
        ap = max(0.0, min(1.0, (intro - n["delay"]) / 0.25))
        if ap <= 0:
            continue
        e = ease(ap)
        breathe = 1 + 0.06 * math.sin(p * 2 * math.pi * 2 + n["x"])
        rad = n["rad"] * e * (breathe if intro >= 1 else 1) * s
        col = n["c"] + (int(235 * e),)
        d.ellipse([n["x"] * s - rad, n["y"] * s - rad, n["x"] * s + rad, n["y"] * s + rad], fill=col)

    # downscale for antialiasing, then draw crisp text
    im = im.resize((W, H), Image.LANCZOS)
    d2 = ImageDraw.Draw(im, "RGBA")
    d2.text((54, 30), "Cartograph", font=_font(42), fill=(231, 233, 238))
    d2.text((56, 84), "your work, mapped into one graph — and plugged into your AI agents",
            font=_font_r(17), fill=(154, 163, 178))
    # prompt bar appears after intro
    if intro >= 1:
        bw = 430
        d2.rounded_rectangle([PROMPT[0] - bw // 2, PROMPT[1] - 17, PROMPT[0] + bw // 2, PROMPT[1] + 15],
                             radius=16, fill=(22, 27, 34), outline=(48, 54, 61))
        txt = '>  "how did I handle auth?"  ->  your relevant work lights up'
        f = _font_r(14)
        tw = d2.textlength(txt, font=f)
        d2.text((PROMPT[0] - tw / 2, PROMPT[1] - 9), txt, font=f, fill=(154, 163, 178))
    return im


def main():
    nframes = 56
    frames = [frame(i / nframes) for i in range(nframes)]
    out = Path(__file__).resolve().parents[1] / "docs" / "assets" / "graph.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    # palette + per-frame durations (hold a beat on the full graph + after the pulse)
    durs = []
    for i in range(nframes):
        p = i / nframes
        durs.append(900 if abs(p - 0.52) < 0.02 else (70 if p < 0.42 or p > 0.55 else 110))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durs, loop=0,
                   optimize=True, disposal=2)
    kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({kb:.0f} KB, {nframes} frames)")


if __name__ == "__main__":
    main()
