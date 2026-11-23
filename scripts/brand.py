"""Build Cartograph's brand assets from code: the Moley badge, README banner, social card and favicons.

    python scripts/brand.py

Writes SVGs to docs/assets/ and cartograph/viz/, then renders PNGs with headless Chrome (or Chromium/Brave).
Text is converted to outlines so the SVGs look identical everywhere. Fonts are fetched once from
Google Fonts: Outfit and Source Code Pro, both SIL Open Font License.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
VIZ = ROOT / "cartograph" / "viz"
FONTS = Path(tempfile.gettempdir()) / "cartograph-brand-fonts"
_GF = "https://github.com/google/fonts/raw/main/ofl/"
OUTFIT = _GF + "outfit/Outfit%5Bwght%5D.ttf"
MONO = _GF + "sourcecodepro/SourceCodePro%5Bwght%5D.ttf"

INK, NAVY, GOLD, TEAL = "#0b1020", "#0f1a33", "#f5b942", "#3fb7a8"
TEXT, DIM, PINK, FUR = "#eef0f4", "#9aa3b2", "#f28b9b", "#4a505f"


def _fetch(url: str) -> Path:
    FONTS.mkdir(parents=True, exist_ok=True)
    p = FONTS / url.rsplit("/", 1)[-1].replace("%5B", "[").replace("%5D", "]")
    if not p.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "cartograph-brand"})
        p.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    return p


def _kerning(font: TTFont):
    if "GPOS" not in font:
        return lambda a, b: 0
    gpos = font["GPOS"].table
    wanted = {i for fr in gpos.FeatureList.FeatureRecord if fr.FeatureTag == "kern"
              for i in fr.Feature.LookupListIndex}
    subs = []
    for i in sorted(wanted):
        lk = gpos.LookupList.Lookup[i]
        for st in lk.SubTable:
            if lk.LookupType == 9:
                if st.ExtensionLookupType != 2:
                    continue
                st = st.ExtSubTable
            elif lk.LookupType != 2:
                continue
            subs.append(st)

    def kern(a: str, b: str) -> int:
        for st in subs:
            cov = st.Coverage.glyphs
            if a not in cov:
                continue
            if st.Format == 1:
                for rec in st.PairSet[cov.index(a)].PairValueRecord:
                    if rec.SecondGlyph == b:
                        return getattr(rec.Value1, "XAdvance", 0) or 0
            else:
                c1 = st.ClassDef1.classDefs.get(a, 0)
                c2 = st.ClassDef2.classDefs.get(b, 0)
                v = st.Class1Record[c1].Class2Record[c2].Value1
                x = getattr(v, "XAdvance", 0) or 0
                if x:
                    return x
        return 0

    return kern


class Font:
    def __init__(self, path: Path, wght: int):
        f = TTFont(path)
        if "fvar" in f:
            f = instantiateVariableFont(f, {"wght": wght})
        self.glyphs, self.cmap = f.getGlyphSet(), f.getBestCmap()
        self.upm, self.kern = f["head"].unitsPerEm, _kerning(f)

    def _names(self, s: str) -> list[str]:
        missing = [ch for ch in s if ord(ch) not in self.cmap]
        if missing:
            raise ValueError(f"font has no glyph for {missing!r}")
        return [self.cmap[ord(ch)] for ch in s]

    def width(self, s: str, size: float, tracking: float = 0) -> float:
        names, k = self._names(s), size / self.upm
        adv = sum(self.glyphs[n].width for n in names)
        adv += sum(self.kern(a, b) for a, b in zip(names, names[1:]))
        return adv * k + tracking * (len(names) - 1)

    def path(self, s: str, size: float, x: float, y: float, fill: str, tracking: float = 0,
             anchor: str = "start") -> str:
        if anchor == "middle":
            x -= self.width(s, size, tracking) / 2
        names, k = self._names(s), size / self.upm
        pen = SVGPathPen(self.glyphs, ntos=lambda v: f"{v:.2f}".rstrip("0").rstrip("."))
        for i, n in enumerate(names):
            self.glyphs[n].draw(TransformPen(pen, (k, 0, 0, -k, x, y)))
            nxt = names[i + 1] if i + 1 < len(names) else None
            x += (self.glyphs[n].width + (self.kern(n, nxt) if nxt else 0)) * k + tracking
        return f'<path fill="{fill}" d="{pen.getCommands()}"/>'


def moley(u: str = "m") -> tuple[str, str]:
    """Moley holding up a brass lantern, as (defs, body) in a 256x256 box."""
    defs = f"""
<clipPath id="{u}c"><circle cx="128" cy="128" r="116"/></clipPath>
<radialGradient id="{u}bg" cx="42%" cy="36%" r="78%"><stop offset="0" stop-color="#1e3062"/><stop offset="1" stop-color="#0b1328"/></radialGradient>
<radialGradient id="{u}glow" cx="192" cy="142" r="128" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#ffd27a" stop-opacity=".6"/><stop offset=".34" stop-color="#ffc55a" stop-opacity=".2"/><stop offset="1" stop-color="#ffc55a" stop-opacity="0"/></radialGradient>
<linearGradient id="{u}fur" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#565d6e"/><stop offset="1" stop-color="#3b404d"/></linearGradient>
<linearGradient id="{u}rim" x1="1" y1="0" x2=".3" y2="0"><stop offset="0" stop-color="#ffcf6e" stop-opacity=".5"/><stop offset="1" stop-color="#ffcf6e" stop-opacity="0"/></linearGradient>
<linearGradient id="{u}brass" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ffdc7f"/><stop offset=".55" stop-color="#d9a13a"/><stop offset="1" stop-color="#9c6a22"/></linearGradient>
<radialGradient id="{u}glass" cx="50%" cy="58%" r="70%"><stop offset="0" stop-color="#fff6d2"/><stop offset="1" stop-color="#ffc964"/></radialGradient>
<radialGradient id="{u}flame" cx="50%" cy="68%" r="62%"><stop offset="0" stop-color="#fffbe8"/><stop offset=".45" stop-color="#ffd25a"/><stop offset="1" stop-color="#ff8a3d"/></radialGradient>"""
    o = f'stroke="{INK}" stroke-linecap="round" stroke-linejoin="round"'
    brass = f"url(#{u}brass)"
    body = f"""
<circle cx="128" cy="128" r="123" fill="{INK}"/>
<circle cx="128" cy="128" r="116" fill="url(#{u}bg)"/>
<g clip-path="url(#{u}c)">
  <circle cx="192" cy="142" r="128" fill="url(#{u}glow)"/>
  <g stroke="{GOLD}" stroke-opacity=".38" stroke-width="1.8" fill="none">
    <path d="M34 104 L58 70 L94 50 L132 40 M58 70 L48 40 M94 50 L84 24"/>
  </g>
  <g fill="{GOLD}">
    <circle cx="34" cy="104" r="3.4" fill-opacity=".55"/><circle cx="58" cy="70" r="4.2" fill-opacity=".8"/>
    <circle cx="94" cy="50" r="3.6" fill-opacity=".65"/><circle cx="132" cy="40" r="3" fill-opacity=".5"/>
    <circle cx="48" cy="40" r="2.6" fill-opacity=".5"/><circle cx="84" cy="24" r="2.4" fill-opacity=".4"/>
    <circle cx="214" cy="62" r="2" fill-opacity=".5"/><circle cx="226" cy="98" r="1.6" fill-opacity=".45"/>
    <circle cx="30" cy="150" r="1.8" fill-opacity=".4"/>
  </g>
  <ellipse cx="114" cy="238" rx="80" ry="66" fill="url(#{u}fur)" stroke="{INK}" stroke-width="5"/>
  <path d="M126 196 L140 242 L119 246 L112 200 Z" fill="{TEAL}" {o} stroke-width="4"/>
  <path d="M122 246 l-1 8 M129 245 l0 8 M136 244 l1 8" stroke="{TEAL}" stroke-width="3.2" stroke-linecap="round"/>
  <path d="M58 172 Q114 204 170 172 L173 191 Q114 224 55 191 Z" fill="{TEAL}" {o} stroke-width="4.5"/>
  <path d="M60 182 Q114 213 168 182" fill="none" stroke="#2a8c81" stroke-width="3"/>
  <ellipse cx="72" cy="216" rx="16" ry="12" fill="#eaa5b0" {o} stroke-width="4"/>
  <path d="M64 206 l.6 5 M72 204.5 v6 M80 206 l-.6 5" {o} stroke-width="2.6"/>
  <ellipse cx="112" cy="122" rx="60" ry="54" fill="url(#{u}fur)" stroke="{INK}" stroke-width="5"/>
  <ellipse cx="112" cy="122" rx="57.5" ry="51.5" fill="url(#{u}rim)"/>
  <path d="M100 73 Q103 58 112 69 Q117 55 125 71" fill="none" {o} stroke-width="4"/>
  <ellipse cx="112" cy="149" rx="33" ry="23" fill="#6a7183"/>
  <circle cx="74" cy="140" r="8.5" fill="{PINK}" fill-opacity=".42"/>
  <circle cx="150" cy="140" r="8.5" fill="{PINK}" fill-opacity=".42"/>
  <path d="M84 146 L62 142 M84 152 L63 156 M140 146 L160 142 M140 152 L159 156" stroke="#cdd2dc" stroke-opacity=".7" stroke-width="2" stroke-linecap="round"/>
  <path d="M79 91 Q90 85 101 90 M123 90 Q134 85 145 91" fill="none" {o} stroke-width="3.6"/>
  <circle cx="91" cy="118" r="7" fill="{INK}"/><circle cx="135" cy="118" r="7" fill="{INK}"/>
  <circle cx="94" cy="115" r="2.5" fill="#fff"/><circle cx="138" cy="115" r="2.5" fill="#fff"/>
  <circle cx="89" cy="121" r="1.1" fill="#fff" fill-opacity=".8"/><circle cx="133" cy="121" r="1.1" fill="#fff" fill-opacity=".8"/>
  <circle cx="90" cy="116" r="17.5" fill="#fff" fill-opacity=".1" stroke="{INK}" stroke-width="5"/>
  <circle cx="134" cy="116" r="17.5" fill="#fff" fill-opacity=".1" stroke="{INK}" stroke-width="5"/>
  <path d="M107.5 114 Q112 108 116.5 114 M72.5 112 L55 107 M151.5 112 L168 107" fill="none" {o} stroke-width="4.5"/>
  <path d="M79 108 Q83 102 90 101 M143 108 Q147 102 154 104" fill="none" stroke="#fff" stroke-opacity=".5" stroke-width="2.4" stroke-linecap="round"/>
  <ellipse cx="112" cy="140" rx="15.5" ry="11" fill="{PINK}" stroke="{INK}" stroke-width="3.6"/>
  <ellipse cx="107" cy="136.5" rx="4.6" ry="2.6" fill="#fff" fill-opacity=".6"/>
  <path d="M100 155 Q112 166 124 155" fill="none" {o} stroke-width="3.6"/>
  <path d="M107.5 159.5 h9 v6.5 q-4.5 2.6 -9 0 Z" fill="#fff" {o} stroke-width="2.6"/>
  <path d="M158 206 Q230 176 196 104" fill="none" {o} stroke-width="27"/>
  <path d="M158 206 Q230 176 196 104" fill="none" stroke="{FUR}" stroke-width="20" stroke-linecap="round"/>
  <path d="M180 120 Q192 98 204 120" fill="none" {o} stroke-width="7"/>
  <path d="M180 120 Q192 98 204 120" fill="none" stroke="{brass}" stroke-width="3.2" stroke-linecap="round"/>
  <path d="M179 124 Q192 110 205 124 Z" fill="{brass}" {o} stroke-width="3.6"/>
  <rect x="176" y="122" width="32" height="8" rx="3" fill="{brass}" {o} stroke-width="3.6"/>
  <rect x="179" y="130" width="26" height="34" rx="5" fill="url(#{u}glass)" {o} stroke-width="3.6"/>
  <circle cx="192" cy="149" r="12" fill="#fff2b8" fill-opacity=".55"/>
  <path d="M192 137 C197 144 198 150 195 155 C193.5 157.5 190.5 157.5 189 155 C186 150 187 144 192 137 Z" fill="url(#{u}flame)"/>
  <path d="M184.5 131.5 V162.5 M199.5 131.5 V162.5" stroke="#a8742a" stroke-width="2.2"/>
  <path d="M182.5 135 V156" stroke="#fff" stroke-opacity=".6" stroke-width="2.2" stroke-linecap="round"/>
  <rect x="175" y="163" width="34" height="9" rx="3" fill="{brass}" {o} stroke-width="3.6"/>
  <rect x="181" y="171" width="22" height="5" rx="2" fill="#9c6a22" {o} stroke-width="3"/>
  <ellipse cx="193" cy="104" rx="13" ry="10.5" fill="#eaa5b0" {o} stroke-width="4"/>
  <path d="M187 95 l.5 5 M193 93.5 v6 M199 95 l-.5 5" {o} stroke-width="2.4"/>
</g>
<circle cx="128" cy="128" r="116" fill="none" stroke="{GOLD}" stroke-width="4"/>"""
    return defs, body


def _svg(w: int, h: int, defs: str, body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f"<defs>{defs}</defs>{body}</svg>\n")


def badge_svg() -> str:
    d, b = moley()
    return _svg(256, 256, d, f'<title>Moley, your local guide</title>{b}')


def favicon_svg() -> str:
    """The badge zoomed onto Moley's face and lantern, so it still reads at 16-32 px."""
    d, b = moley("f")
    d += '<clipPath id="fv"><circle cx="128" cy="128" r="116"/></clipPath>'
    return _svg(256, 256, d, f'<circle cx="128" cy="128" r="124" fill="{INK}"/>'
                f'<g clip-path="url(#fv)"><g transform="translate(-62 -46) scale(1.35)">{b}</g></g>'
                f'<circle cx="128" cy="128" r="116" fill="none" stroke="{GOLD}" stroke-width="9"/>')


def _backdrop(w: int, h: int, glow_x: float, glow_y: float, r: float, rx: int) -> tuple[str, str]:
    defs = (f'<radialGradient id="bgl" cx="{glow_x}" cy="{glow_y}" r="{r}" gradientUnits="userSpaceOnUse">'
            '<stop offset="0" stop-color="#ffc55a" stop-opacity=".16"/>'
            '<stop offset="1" stop-color="#ffc55a" stop-opacity="0"/></radialGradient>'
            '<linearGradient id="bgn" x1="0" y1="0" x2="1" y2="1">'
            '<stop offset="0" stop-color="#132147"/><stop offset="1" stop-color="#0b1328"/></linearGradient>')
    body = (f'<rect width="{w}" height="{h}" rx="{rx}" fill="url(#bgn)"/>'
            f'<rect width="{w}" height="{h}" rx="{rx}" fill="url(#bgl)"/>')
    return defs, body


def _graph(points: list[tuple[float, float]], edges: list[tuple[int, int]], opacity: float = .22) -> str:
    lines = " ".join(f"M{points[a][0]} {points[a][1]} L{points[b][0]} {points[b][1]}" for a, b in edges)
    dots = "".join(f'<circle cx="{x}" cy="{y}" r="{2.2 + (i % 3)}"/>' for i, (x, y) in enumerate(points))
    return (f'<g stroke="{GOLD}" stroke-opacity="{opacity}" stroke-width="1.5" fill="none"><path d="{lines}"/></g>'
            f'<g fill="{GOLD}" fill-opacity="{opacity * 2:.2f}">{dots}</g>')


def banner_svg(title: Font, text: Font) -> str:
    W, H = 1280, 360
    bd, bb = _backdrop(W, H, 184, 180, 420, 28)
    md, mb = moley()
    tag = "Your files, cited. Your agents, remembering."
    assert 360 + text.width(tag, 38) < W - 40, "tagline too wide"
    pts = [(1040, 50), (1110, 92), (1188, 64), (1150, 150), (1232, 182), (1084, 214), (1200, 268), (1120, 310)]
    body = (bb + _graph(pts, [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (4, 6), (5, 7), (6, 7)])
            + f'<g transform="translate(56 52)">{mb}</g>'
            + title.path("Cartograph", 118, 352, 178, TEXT, tracking=-1.5)
            + text.path(tag, 38, 358, 240, GOLD)
            + text.path("Moley \u00b7 your local guide", 26, 360, 292, TEAL, tracking=.6))
    return _svg(W, H, bd + md, '<title>Cartograph \u2014 Moley, your local guide</title>' + body)


def social_svg(title: Font, text: Font, mono: Font) -> str:
    W, H = 1280, 640
    bd, bb = _backdrop(W, H, 1000, 300, 520, 0)
    md, mb = moley()
    x = 80
    bullets = ["PDFs, Office files, code and notes, cited to the page and line",
               "One memory for Claude, Cursor, Codex, Gemini and more",
               "Runs on your machine. No LLM needed to build the graph"]
    cmd = "$ pip install cartograph__v1 && carto connect all"
    for s in bullets:
        assert x + 26 + text.width(s, 25) < 770, f"bullet too wide: {s}"
    assert x + mono.width(cmd, 23) < 790, "command too wide"
    pts = [(700, 70), (770, 40), (820, 110), (900, 60), (740, 560), (820, 600), (880, 540)]
    parts = [bb, _graph(pts, [(0, 1), (1, 2), (2, 3), (1, 3), (4, 5), (5, 6)], .18),
             f'<g transform="translate(790 92) scale(1.66)">{mb}</g>',
             text.path("Moley \u00b7 your local guide", 26, 1002, 562, TEAL, tracking=.6, anchor="middle"),
             title.path("Cartograph", 108, x - 4, 196, TEXT, tracking=-1.5),
             text.path("Your files, cited.", 46, x, 268, GOLD),
             text.path("Your agents, remembering.", 46, x, 324, GOLD)]
    for i, s in enumerate(bullets):
        y = 398 + i * 42
        parts.append(f'<circle cx="{x + 7}" cy="{y - 8}" r="6" fill="{TEAL}"/>')
        parts.append(text.path(s, 25, x + 26, y, TEXT))
    parts.append(f'<rect x="{x - 14}" y="510" width="{mono.width(cmd, 23) + 28:.0f}" height="46" rx="10" '
                 'fill="#070c18" fill-opacity=".7" stroke="#24304f"/>')
    parts.append(mono.path(cmd, 23, x, 541, DIM))
    return _svg(W, H, bd + md, '<title>Cartograph \u2014 your files, cited; your agents, remembering</title>'
                + "".join(parts))


def _chrome() -> str:
    for c in (os.environ.get("CHROME"), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              "/Applications/Chromium.app/Contents/MacOS/Chromium",
              "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
              shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chromium-browser")):
        if c and Path(c).exists():
            return c
    raise SystemExit("needs Chrome/Chromium for PNG rendering (set CHROME=/path/to/chrome)")


def render(svg: Path, w: int, h: int, scale: int = 2) -> Image.Image:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.png"
        # Headless Chrome on macOS can write the screenshot and then never exit, so wait for the file.
        proc = subprocess.Popen([_chrome(), "--headless=new", "--disable-gpu", "--hide-scrollbars",
                                 "--no-first-run", "--no-default-browser-check",
                                 f"--user-data-dir={tmp}/profile", f"--window-size={w},{h}",
                                 f"--force-device-scale-factor={scale}", "--default-background-color=00000000",
                                 f"--screenshot={out}", svg.as_uri()],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in proc.stdout:
                if "bytes written to file" in line:
                    break
        finally:
            proc.kill()
            proc.wait()
        if not out.exists():
            raise SystemExit(f"Chrome did not render {svg}")
        return Image.open(out).convert("RGBA").copy()


def main() -> None:
    title, text = Font(_fetch(OUTFIT), 700), Font(_fetch(OUTFIT), 500)
    mono = Font(_fetch(MONO), 500)
    ASSETS.mkdir(parents=True, exist_ok=True)
    files = {ASSETS / "moley.svg": badge_svg(), VIZ / "moley.svg": badge_svg(), VIZ / "favicon.svg": favicon_svg(),
             ASSETS / "banner.svg": banner_svg(title, text), ASSETS / "social.svg": social_svg(title, text, mono)}
    for p, s in files.items():
        p.write_text(s, encoding="utf-8")

    badge = render(ASSETS / "moley.svg", 256, 256, scale=4)
    badge.resize((512, 512), Image.LANCZOS).save(ASSETS / "mascot.png", optimize=True)
    badge.resize((180, 180), Image.LANCZOS).save(VIZ / "mascot.png", optimize=True)
    render(VIZ / "favicon.svg", 256, 256).resize((64, 64), Image.LANCZOS).save(VIZ / "favicon.png", optimize=True)
    social = render(ASSETS / "social.svg", 1280, 640).resize((1280, 640), Image.LANCZOS).convert("RGB")
    social.save(ASSETS / "social.png", optimize=True)
    for p in [*files, ASSETS / "mascot.png", VIZ / "mascot.png", VIZ / "favicon.png", ASSETS / "social.png"]:
        print(f"{p.relative_to(ROOT)}  {p.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
