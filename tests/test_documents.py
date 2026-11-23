"""Document ingestion: every format yields text with a locator, chunks carry citations end to end,
and older graphs (no locator column) keep working."""
import sqlite3
import zipfile

import pytest

from cartograph import documents as D
from cartograph import ingest as ing
from cartograph.config import Config, db_path
from cartograph.retrieve import retrieve
from cartograph.storage import Store

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
S = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
R = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'


def _zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for name, body in files.items():
            z.writestr(name, body)
    return path


def make_pdf(path, pages):
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None]
    kids = []
    font = 3
    objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        content = len(objs)
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content} 0 R "
                    f"/Resources << /Font << /F1 {font} 0 R >> >> >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offs = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offs).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)
    return path


def make_docx(path):
    def p(text, style=None):
        ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        return f"<w:p>{ppr}<w:r><w:t>{text}</w:t></w:r></w:p>"
    body = (p("Intro paragraph about corneal permeability.") + p("Methods", "Heading1")
            + p("We used Franz diffusion cells.") + p("Results", "Heading1") + p("Timolol flux doubled."))
    return _zip(path, {"word/document.xml": f"<w:document {W}><w:body>{body}</w:body></w:document>"})


def make_pptx(path):
    slide = lambda t: f"<p:sld xmlns:p='p' {A}><a:p><a:r><a:t>{t}</a:t></a:r></a:p></p:sld>"  # noqa: E731
    return _zip(path, {
        "ppt/slides/slide1.xml": slide("Title slide"),
        "ppt/slides/slide2.xml": slide("Pharmacokinetic compartment model"),
        "ppt/slides/slide10.xml": slide("Thank you"),
        "ppt/notesSlides/notesSlide2.xml": slide("mention aqueous humor turnover"),
    })


def make_xlsx(path):
    return _zip(path, {
        "xl/workbook.xml": f'<workbook {S} {R}><sheets><sheet name="Doses" sheetId="1" r:id="rId1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                                      '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
        "xl/sharedStrings.xml": f"<sst {S}><si><t>drug</t></si><si><t>timolol</t></si></sst>",
        "xl/worksheets/sheet1.xml": f'<worksheet {S}><sheetData>'
                                    '<row><c t="s"><v>0</v></c><c t="inlineStr"><is><t>dose_mg</t></is></c></row>'
                                    '<row><c t="s"><v>1</v></c><c><v>0.5</v></c></row></sheetData></worksheet>',
    })


def make_epub(path):
    return _zip(path, {
        "META-INF/container.xml": '<container><rootfiles><rootfile full-path="OEBPS/book.opf"/></rootfiles></container>',
        "OEBPS/book.opf": '<package><manifest><item id="c2" href="ch2.xhtml"/><item id="c1" href="ch1.xhtml"/>'
                          '</manifest><spine><itemref idref="c1"/><itemref idref="c2"/></spine></package>',
        "OEBPS/ch1.xhtml": "<html><head><title>Diffusion</title></head><body><p>Fick's law.</p></body></html>",
        "OEBPS/ch2.xhtml": "<html><head><title>Clearance</title></head><body><p>Tear turnover.</p></body></html>",
    })


# ---------------------------------------------------------------------------------------------- formats
def test_docx_sections_follow_headings(tmp_path):
    secs = D.extract(make_docx(tmp_path / "paper.docx"))
    assert [s.locator for s in secs] == ["", "§ Methods", "§ Results"]
    assert "Franz diffusion" in secs[1].text


def test_pptx_slides_in_numeric_order_with_notes(tmp_path):
    secs = D.extract(make_pptx(tmp_path / "talk.pptx"))
    assert [s.locator for s in secs] == ["slide 1", "slide 2", "slide 10"]
    assert "aqueous humor" in secs[1].text


def test_xlsx_shared_and_inline_strings(tmp_path):
    (sec,) = D.extract(make_xlsx(tmp_path / "doses.xlsx"))
    assert sec.locator == "sheet Doses"
    assert sec.text.splitlines() == ["drug\tdose_mg", "timolol\t0.5"]


def test_epub_follows_spine_order(tmp_path):
    secs = D.extract(make_epub(tmp_path / "book.epub"))
    assert [s.locator for s in secs] == ["§ Diffusion", "§ Clearance"]


def test_html_drops_scripts_and_keeps_title(tmp_path):
    p = tmp_path / "page.html"
    p.write_text("<html><head><title>Notes</title><script>var x=1</script></head>"
                 "<body><h1>Ocular PK</h1><p>Cornea &amp; sclera.</p></body></html>")
    (sec,) = D.extract(p)
    assert sec.locator == "§ Notes" and "Cornea & sclera." in sec.text and "var x" not in sec.text


def test_pdf_pages(tmp_path):
    pytest.importorskip("pypdf")
    secs = D.extract(make_pdf(tmp_path / "paper.pdf", ["first page intro", "corneal epithelium barrier"]))
    assert [s.locator for s in secs] == ["p.1", "p.2"]
    assert "epithelium" in secs[1].text


def test_corrupt_documents_yield_nothing(tmp_path):
    for name in ("bad.docx", "bad.pptx", "bad.xlsx", "bad.epub", "bad.pdf"):
        (tmp_path / name).write_bytes(b"not a real file")
        assert D.extract(tmp_path / name) == []


# ---------------------------------------------------------------------------------------------- chunking
def _old_chunk_text(text, target=1200, overlap=150):
    text = text.replace("\r\n", "\n")
    if len(text) <= target:
        return [text.strip()] if text.strip() else []
    out, start = [], 0
    while start < len(text):
        end = min(start + target, len(text))
        nl = text.rfind("\n", start + target // 2, end)
        if nl != -1 and end < len(text):
            end = nl
        seg = text[start:end].strip()
        if seg:
            out.append(seg)
        start = max(end - overlap, end) if end >= len(text) else end - overlap
        if start <= 0:
            break
    return out


def test_chunking_is_byte_identical_to_v1():
    import random
    rng = random.Random(7)
    for _ in range(50):
        lines = ["  " * rng.randint(0, 3) + "x" * rng.randint(0, 140) for _ in range(rng.randint(0, 120))]
        text = "\r\n".join(lines) if rng.random() < 0.2 else "\n".join(lines)
        assert ing.chunk_text(text) == _old_chunk_text(text)


def test_line_locators_point_at_the_chunk():
    text = "\n".join(f"line {i}" for i in range(1, 501))
    sec = D.text_section(text)
    pairs = ing.chunk_sections([sec])
    assert pairs[0][1].startswith("L1-")
    for chunk, loc in pairs:
        a, b = map(int, loc[1:].split("-"))
        lines = chunk.splitlines()                       # overlap may start mid-line, as in v1
        assert f"line {a}".endswith(lines[0]) and f"line {b}".startswith(lines[-1])


def test_notebook_cells_locator():
    sec = D.notebook_section(["import numpy", "x = 1", "print(x)"])
    assert D.span_locator(sec, 0, len(sec.text)) == "cell 1-3"
    assert D.span_locator(sec, sec.text.index("x = 1"), sec.text.index("x = 1") + 5) == "cell 2"


def test_cite_format():
    assert D.cite("src/train.py", "L40-88") == "src/train.py:L40-88"
    assert D.cite("paper.pdf", "p.12") == "paper.pdf p.12"
    assert D.cite("notes.md", None) == "notes.md"


# ---------------------------------------------------------------------------------------------- end to end
def _proj(tmp_path):
    d = tmp_path / "ocular"
    d.mkdir()
    (d / ".git").mkdir()
    return d


def test_ingest_documents_and_retrieve_with_citations(tmp_path):
    d = _proj(tmp_path)
    make_docx(d / "paper.docx")
    make_pptx(d / "talk.pptx")
    (d / "model.py").write_text("\n".join(["import numpy as np"] * 3 + ["def clearance_rate(k): return k"]))
    store = Store(db_path())
    st = ing.ingest_path(d, store, Config())
    assert st.docs == 2
    res = retrieve("Franz diffusion cells", store, Config(), top_k=3)
    top = res.chunks[0]
    assert top["locator"] == "§ Methods" and top["cite"].endswith("paper.docx § Methods")
    res = retrieve("clearance_rate", store, Config(), top_k=3)
    assert res.chunks[0]["cite"].endswith("model.py:L1-4")


def test_pdf_ingest_cites_pages(tmp_path):
    pytest.importorskip("pypdf")
    d = _proj(tmp_path)
    make_pdf(d / "paper.pdf", ["introduction text", "corneal epithelium barrier"])
    store = Store(db_path())
    ing.ingest_path(d, store, Config())
    res = retrieve("epithelium", store, Config(), top_k=1)
    assert res.chunks[0]["cite"].endswith("paper.pdf p.2")


def test_office_lock_files_skipped(tmp_path):
    d = _proj(tmp_path)
    make_docx(d / "paper.docx")
    (d / "~$paper.docx").write_bytes(b"lock")
    (d / ".~paper.docx").write_bytes(b"lock")
    assert ing.ingest_path(d, Store(db_path()), Config()).files_seen == 1


def test_pdfs_counted_when_no_backend(tmp_path, monkeypatch):
    d = _proj(tmp_path)
    make_pdf(d / "paper.pdf", ["hello"])
    (d / "notes.md").write_text("notes")
    monkeypatch.setattr(D, "pdf_backend", lambda: None)
    st = ing.ingest_path(d, Store(db_path()), Config())
    assert st.docs_unsupported == 1 and st.docs == 0


def test_legacy_graph_without_locator_column(tmp_path):
    d = _proj(tmp_path)
    src = d / "train.py"
    src.write_text("\n".join(f"step {i}" for i in range(1, 40)) + "\nlearning_rate_schedule = cosine\n")
    db = tmp_path / "legacy.sqlite"
    c = sqlite3.connect(db)
    c.executescript("""
        CREATE TABLE projects (id INTEGER PRIMARY KEY, name TEXT UNIQUE, path TEXT, field TEXT, created_at TEXT);
        CREATE TABLE files (id INTEGER PRIMARY KEY, project_id INTEGER, path TEXT, ext TEXT, hash TEXT);
        CREATE TABLE chunks (id INTEGER PRIMARY KEY, file_id INTEGER, chunk_text TEXT, ordinal INTEGER);
        CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_text, content='chunks', content_rowid='id');
        CREATE TABLE edges (id INTEGER PRIMARY KEY, src_id INTEGER, relation TEXT, dst_id INTEGER, metadata TEXT);
        CREATE TABLE skills (id INTEGER PRIMARY KEY, name TEXT UNIQUE);
    """)
    c.execute("INSERT INTO projects VALUES (1, 'ocular', ?, 'general', '')", (str(d),))
    c.execute("INSERT INTO files VALUES (1, 1, ?, '.py', 'h')", (str(src),))
    c.execute("INSERT INTO chunks VALUES (1, 1, 'learning_rate_schedule = cosine', 0)")
    c.execute("INSERT INTO chunks_fts(rowid, chunk_text) VALUES (1, 'learning_rate_schedule = cosine')")
    c.commit()
    c.close()
    ro = Store(db, read_only=True)
    res = retrieve("learning_rate_schedule", ro, Config(), top_k=1)
    assert res.chunks[0]["cite"] == f"{src}:L40"
    Store(db)                                            # read-write open migrates in place
    assert Store(db, read_only=True).has_column("chunks", "locator")
