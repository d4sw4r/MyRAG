"""Find files and load them as text.

Supported: plain text and source code, Markdown, PDF, Word (.docx), PowerPoint
(.pptx), Excel (.xlsx), CSV, HTML, EPUB and e-mails (.eml).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

# Read as plain text.
TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".org", ".tex", ".log",
    # source code and config
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".go", ".rs", ".c", ".h",
    ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".scala", ".sh", ".bash", ".ps1",
    ".sql", ".r", ".lua", ".pl", ".css", ".scss", ".vue", ".svelte",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml",
}
HTML_EXTENSIONS = {".html", ".htm"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | HTML_EXTENSIONS | {
    ".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".epub", ".eml",
}
SKIP_DIRS = {".git", ".myrag", "__pycache__", "node_modules", ".venv", "venv"}


def iter_files(root: Path) -> Iterator[Path]:
    """All supported files under root, recursive and sorted."""
    for path in sorted(root.rglob("*")):
        rel_parts = path.relative_to(root).parts
        if any(p in SKIP_DIRS or p.startswith(".") for p in rel_parts):
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return _read_plain(path)
    reader = _READERS.get(suffix)
    if reader is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return reader(path)


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _read_plain(path: Path) -> str:
    return _decode(path.read_bytes())


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _read_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def html_to_text(html: str) -> str:
    """Visible text of an HTML document, one block per line."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    if soup.title:
        soup.title.decompose()
    lines = [line.strip() for line in soup.get_text("\n").splitlines()]
    body = "\n".join(line for line in lines if line)
    return f"{title}\n\n{body}" if title and not body.startswith(title) else body


def _read_html(path: Path) -> str:
    return html_to_text(_decode(path.read_bytes()))


def _read_csv(path: Path) -> str:
    """One line per row as "column: value; ..." so each chunk keeps the column names."""
    import csv
    import io

    text = _decode(path.read_bytes())
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in r)]
    if not rows:
        return ""
    header, data = rows[0], rows[1:]
    if not data:
        return " | ".join(header)
    return "\n".join(_row_line(header, r) for r in data)


def _row_line(header: list, row: list) -> str:
    cells = []
    for i, value in enumerate(row):
        value = "" if value is None else str(value).strip()
        if not value:
            continue
        name = str(header[i]).strip() if i < len(header) and header[i] is not None else ""
        cells.append(f"{name}: {value}" if name else value)
    return "; ".join(cells)


def _read_xlsx(path: Path) -> str:
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    parts = []
    try:
        for ws in wb.worksheets:
            rows = [r for r in ws.iter_rows(values_only=True) if any(c is not None and str(c).strip() for c in r)]
            if not rows:
                continue
            header, data = list(rows[0]), rows[1:]
            lines = [_row_line(header, list(r)) for r in data] if data else [" | ".join(str(c) for c in header if c is not None)]
            parts.append(f"Sheet: {ws.title}\n" + "\n".join(line for line in lines if line))
    finally:
        wb.close()
    return "\n\n".join(parts)


def _read_pptx(path: Path) -> str:
    from pptx import Presentation

    prs = Presentation(str(path))
    slides = []
    for number, slide in enumerate(prs.slides, 1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                texts.append(shape.text_frame.text)
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    texts.append(" | ".join(cell.text for cell in row.cells))
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text
            if notes.strip():
                texts.append(f"Notes: {notes}")
        body = "\n".join(t for t in texts if t.strip())
        if body:
            slides.append(f"Slide {number}\n{body}")
    return "\n\n".join(slides)


def _read_epub(path: Path) -> str:
    """Chapters in reading order (spine of the OPF package)."""
    import posixpath
    import xml.etree.ElementTree as ET
    import zipfile

    with zipfile.ZipFile(path) as z:
        container = ET.fromstring(z.read("META-INF/container.xml"))
        rootfile = next(el for el in container.iter() if el.tag.endswith("rootfile"))
        opf_path = rootfile.attrib["full-path"]
        opf = ET.fromstring(z.read(opf_path))
        base = posixpath.dirname(opf_path)
        manifest = {
            el.attrib["id"]: el.attrib["href"]
            for el in opf.iter() if el.tag.endswith("item") and "id" in el.attrib
        }
        order = [el.attrib["idref"] for el in opf.iter() if el.tag.endswith("itemref")]
        chapters = []
        for idref in order:
            href = manifest.get(idref)
            if not href:
                continue
            name = posixpath.normpath(posixpath.join(base, href))
            try:
                chapters.append(html_to_text(_decode(z.read(name))))
            except KeyError:
                continue
    return "\n\n".join(c for c in chapters if c.strip())


def _read_eml(path: Path) -> str:
    from email import policy
    from email.parser import BytesParser

    msg = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    header = "\n".join(
        f"{name}: {msg[name]}" for name in ("Subject", "From", "To", "Date") if msg[name]
    )
    body = msg.get_body(preferencelist=("plain", "html"))
    text = ""
    if body is not None:
        content = body.get_content()
        text = html_to_text(content) if body.get_content_type() == "text/html" else content
    attachments = [p.get_filename() for p in msg.iter_attachments() if p.get_filename()]
    if attachments:
        text += "\n\nAttachments: " + ", ".join(attachments)
    return f"{header}\n\n{text.strip()}".strip()


_READERS = {
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".pptx": _read_pptx,
    ".xlsx": _read_xlsx,
    ".csv": _read_csv,
    ".html": _read_html,
    ".htm": _read_html,
    ".epub": _read_epub,
    ".eml": _read_eml,
}
