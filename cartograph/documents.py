"""Turn documents into citable text sections. Office formats (DOCX, PPTX, XLSX), EPUB and HTML are
parsed with the standard library; PDF uses PyMuPDF when installed (faster, optional OCR) else pypdf
(`pip install "cartograph__v1[docs]"`). Every section carries a locator — `p.12`, `slide 3`,
`sheet Revenue`, `§ Methods` — so a retrieved chunk can be cited back to where it came from."""
from __future__ import annotations

import html
import html.parser
import posixpath
import re
import zipfile
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

DOC_EXT = {".pdf", ".docx", ".pptx", ".xlsx", ".epub", ".html", ".htm"}
_MAX_SHEET_ROWS = 2000


@dataclass
class Section:
    """A run of text plus where it lives. `starts` are offsets where unit 1, 2, ... begin inside `text`
    (lines for source files, cells for notebooks); chunks inside it get `<unit> a-b` locators."""
    text: str
    locator: str = ""
    unit: str = ""
    starts: list[int] = field(default_factory=list)


def pdf_backend() -> str | None:
    for mod, name in (("pymupdf", "pymupdf"), ("fitz", "pymupdf"), ("pypdf", "pypdf")):
        try:
            __import__(mod)
            return name
        except Exception:
            continue
    return None


def supported(p: Path) -> bool:
    ext = p.suffix.lower()
    return ext in DOC_EXT and (ext != ".pdf" or pdf_backend() is not None)


# ------------------------------------------------------------------------------------------ helpers
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _xml(z: zipfile.ZipFile, name: str):
    try:
        return ET.fromstring(z.read(name))
    except (KeyError, ET.ParseError):
        return None


class _HTMLText(html.parser.HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "head"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
              "pre", "blockquote", "table", "ul", "ol", "dt", "dd"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in self._BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False
        if tag in self._BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.out.append(data)

    def text(self) -> str:
        t = "".join(self.out)
        t = re.sub(r"[ \t\r\f\v]+", " ", t)
        return re.sub(r"\n\s*\n+", "\n\n", t).strip()


def html_to_text(markup: str) -> tuple[str, str]:
    """(title, text) of an HTML/XHTML document."""
    p = _HTMLText()
    try:
        p.feed(markup)
        p.close()
    except Exception:
        return "", html.unescape(re.sub(r"<[^>]+>", " ", markup))
    return p.title.strip(), p.text()


# ------------------------------------------------------------------------------------------ formats
def _pdf(p: Path, ocr: bool = False) -> list[Section]:
    out: list[Section] = []
    backend = pdf_backend()
    if backend == "pymupdf":
        try:
            import pymupdf as fitz
        except Exception:
            import fitz  # type: ignore
        with fitz.open(str(p)) as doc:
            for i, page in enumerate(doc, 1):
                text = page.get_text("text") or ""
                if ocr and len(text.strip()) < 20:
                    text = _ocr_page(page) or text
                if text.strip():
                    out.append(Section(text, f"p.{i}"))
        return out
    if backend == "pypdf":
        import logging

        from pypdf import PdfReader
        logging.getLogger("pypdf").setLevel(logging.ERROR)
        reader = PdfReader(str(p))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                return []
        for i, page in enumerate(reader.pages, 1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                out.append(Section(text, f"p.{i}"))
    return out


def _ocr_page(page) -> str:
    try:
        import io

        import pytesseract
        from PIL import Image
        pix = page.get_pixmap(dpi=200)
        return pytesseract.image_to_string(Image.open(io.BytesIO(pix.tobytes("png"))))
    except Exception:
        return ""


def _docx(p: Path) -> list[Section]:
    with zipfile.ZipFile(p) as z:
        root = _xml(z, "word/document.xml")
    if root is None:
        return []
    sections: list[Section] = []
    heading, buf = "", []

    def flush():
        text = "\n".join(buf).strip()
        if text:
            sections.append(Section(text, f"§ {heading[:80]}" if heading else ""))

    for para in root.iter():
        if _local(para.tag) != "p":
            continue
        style = ""
        for el in para.iter():
            if _local(el.tag) == "pStyle":
                style = next((v for k, v in el.attrib.items() if _local(k) == "val"), "")
                break
        text = "".join(el.text or "" for el in para.iter() if _local(el.tag) == "t").strip()
        if not text:
            continue
        if re.match(r"(?i)^(heading\s?\d|title)$", style):
            flush()
            heading, buf = text, [text]
        else:
            buf.append(text)
    flush()
    return sections


def _slide_text(root) -> str:
    paras = []
    for para in root.iter():
        if _local(para.tag) == "p":
            t = "".join(el.text or "" for el in para.iter() if _local(el.tag) == "t").strip()
            if t:
                paras.append(t)
    return "\n".join(paras)


def _pptx(p: Path) -> list[Section]:
    out: list[Section] = []
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
        slides = sorted((n for n in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                        key=lambda n: int(re.findall(r"\d+", n)[-1]))
        for n in slides:
            num = int(re.findall(r"\d+", n)[-1])
            root = _xml(z, n)
            text = _slide_text(root) if root is not None else ""
            notes = _xml(z, f"ppt/notesSlides/notesSlide{num}.xml")
            if notes is not None:
                nt = _slide_text(notes)
                if nt:
                    text += "\n[notes] " + nt
            if text.strip():
                out.append(Section(text, f"slide {num}"))
    return out


def _xlsx(p: Path) -> list[Section]:
    out: list[Section] = []
    with zipfile.ZipFile(p) as z:
        shared: list[str] = []
        sroot = _xml(z, "xl/sharedStrings.xml")
        if sroot is not None:
            for si in sroot:
                shared.append("".join(el.text or "" for el in si.iter() if _local(el.tag) == "t"))
        wb = _xml(z, "xl/workbook.xml")
        rels = _xml(z, "xl/_rels/workbook.xml.rels")
        target = {}
        if rels is not None:
            for r in rels:
                target[r.attrib.get("Id")] = r.attrib.get("Target", "")
        sheets = []
        if wb is not None:
            for el in wb.iter():
                if _local(el.tag) == "sheet":
                    rid = next((v for k, v in el.attrib.items() if _local(k) == "id"), None)
                    t = target.get(rid, "")
                    path = t.lstrip("/") if t.startswith("/") else posixpath.normpath(posixpath.join("xl", t))
                    sheets.append((el.attrib.get("name", "sheet"), path))
        for name, path in sheets:
            root = _xml(z, path)
            if root is None:
                continue
            rows = []
            for row in root.iter():
                if _local(row.tag) != "row":
                    continue
                cells = []
                for c in row:
                    if _local(c.tag) != "c":
                        continue
                    kind = c.attrib.get("t", "")
                    val = ""
                    for el in c.iter():
                        tag = _local(el.tag)
                        if tag == "v" and el.text is not None:
                            val = el.text
                        elif tag == "t" and kind == "inlineStr":
                            val += el.text or ""
                    if kind == "s" and val.isdigit() and int(val) < len(shared):
                        val = shared[int(val)]
                    cells.append(val)
                if any(x.strip() for x in cells):
                    rows.append("\t".join(cells).rstrip())
                if len(rows) >= _MAX_SHEET_ROWS:
                    break
            if rows:
                out.append(Section("\n".join(rows), f"sheet {name}"))
    return out


def _epub(p: Path) -> list[Section]:
    out: list[Section] = []
    with zipfile.ZipFile(p) as z:
        cont = _xml(z, "META-INF/container.xml")
        opf_path = ""
        if cont is not None:
            for el in cont.iter():
                if _local(el.tag) == "rootfile":
                    opf_path = el.attrib.get("full-path", "")
                    break
        opf = _xml(z, opf_path) if opf_path else None
        order: list[str] = []
        if opf is not None:
            manifest = {el.attrib.get("id"): el.attrib.get("href", "") for el in opf.iter()
                        if _local(el.tag) == "item"}
            base = posixpath.dirname(opf_path)
            for el in opf.iter():
                if _local(el.tag) == "itemref":
                    href = manifest.get(el.attrib.get("idref"), "")
                    if href:
                        order.append(posixpath.normpath(posixpath.join(base, href)))
        if not order:
            order = sorted(n for n in z.namelist() if n.lower().endswith((".xhtml", ".html", ".htm")))
        for n in order:
            try:
                markup = z.read(n).decode("utf-8", errors="ignore")
            except KeyError:
                continue
            title, text = html_to_text(markup)
            if text:
                out.append(Section(text, f"§ {title[:80]}" if title else posixpath.basename(n)))
    return out


def _html(p: Path) -> list[Section]:
    title, text = html_to_text(p.read_text(encoding="utf-8", errors="ignore"))
    return [Section(text, f"§ {title[:80]}" if title else "")] if text else []


_EXTRACTORS = {".docx": _docx, ".pptx": _pptx, ".xlsx": _xlsx, ".epub": _epub, ".html": _html, ".htm": _html}


def extract(p: Path, *, ocr: bool = False) -> list[Section]:
    """Sections of a document; [] when it has no extractable text or can't be parsed."""
    ext = p.suffix.lower()
    try:
        if ext == ".pdf":
            return _pdf(p, ocr=ocr)
        fn = _EXTRACTORS.get(ext)
        return fn(p) if fn else []
    except (zipfile.BadZipFile, OSError, ValueError, KeyError, ET.ParseError):
        return []
    except Exception:
        return []


# ------------------------------------------------------------------------------------------ text + code
def text_section(text: str) -> Section:
    """A plain-text/source file as one section whose chunks get line-range locators (L12-48)."""
    starts = [0] + [m.end() for m in re.finditer("\n", text)]
    return Section(text, "", "L", starts)


def notebook_section(cells: list[str]) -> Section:
    starts, parts, pos = [], [], 0
    for c in cells:
        starts.append(pos)
        parts.append(c)
        pos += len(c) + 2
    return Section("\n\n".join(parts), "", "cell ", starts)


def span_locator(sec: Section, start: int, end: int) -> str:
    if not sec.unit or not sec.starts:
        return sec.locator
    a = bisect_right(sec.starts, start)
    b = bisect_right(sec.starts, max(start, end - 1))
    loc = f"{sec.unit}{a}" if a == b else f"{sec.unit}{a}-{b}"
    return f"{sec.locator} {loc}".strip()


_LEGACY_MAX = 4_000_000


def legacy_locator(file_path: str, chunk_text: str) -> str | None:
    """Line range of a chunk stored before locators existed, found by re-reading its source file.
    None when the file is gone (moved, or on an unplugged drive), is a binary document, or changed."""
    if not file_path or not chunk_text or Path(file_path).suffix.lower() in DOC_EXT | {".ipynb"}:
        return None
    try:
        p = Path(file_path)
        if not p.is_file() or p.stat().st_size > _LEGACY_MAX:
            return None
        text = p.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
    except OSError:
        return None
    i = text.find(chunk_text)
    if i < 0:
        return None
    return span_locator(text_section(text), i, i + len(chunk_text))


def cite(file_path: str, locator: str | None) -> str:
    """Human citation for a chunk: `src/train.py:L40-88`, `paper.pdf p.12`, `deck.pptx slide 3`."""
    if not locator:
        return file_path
    return f"{file_path}:{locator}" if re.match(r"^L\d", locator) else f"{file_path} {locator}"
