import zipfile
from pathlib import Path

from myrag.loaders import iter_files, load_text


def test_html(tmp_path: Path):
    p = tmp_path / "page.html"
    p.write_text("<html><head><title>Titel</title><style>x{}</style></head>"
                 "<body><h1>Kopf</h1><p>Absatz <b>fett</b></p><script>alert(1)</script></body></html>")
    text = load_text(p)
    assert text.startswith("Titel") and "Kopf" in text and "fett" in text
    assert "alert" not in text and "x{}" not in text


def test_csv_keeps_column_names(tmp_path: Path):
    p = tmp_path / "data.csv"
    p.write_text("Name;Stadt\nAnna;Berlin\nBen;Köln\n", encoding="utf-8")
    assert load_text(p) == "Name: Anna; Stadt: Berlin\nName: Ben; Stadt: Köln"


def test_xlsx(tmp_path: Path):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Kosten"
    ws.append(["Posten", "Betrag"])
    ws.append(["Miete", 900])
    wb.save(tmp_path / "k.xlsx")
    assert load_text(tmp_path / "k.xlsx") == "Sheet: Kosten\nPosten: Miete; Betrag: 900"


def test_pptx(tmp_path: Path):
    from pptx import Presentation

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Agenda"
    slide.placeholders[1].text = "Punkt eins"
    slide.notes_slide.notes_text_frame.text = "Sprechernotiz"
    prs.save(tmp_path / "s.pptx")
    text = load_text(tmp_path / "s.pptx")
    assert text.startswith("Slide 1\nAgenda") and "Punkt eins" in text and "Notes: Sprechernotiz" in text


def test_epub_in_spine_order(tmp_path: Path):
    p = tmp_path / "book.epub"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml",
                   '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                   '<rootfile full-path="OEBPS/content.opf"/></rootfiles></container>')
        z.writestr("OEBPS/content.opf",
                   '<package xmlns="http://www.idpf.org/2007/opf"><manifest>'
                   '<item id="c1" href="one.xhtml"/><item id="c2" href="two.xhtml"/></manifest>'
                   '<spine><itemref idref="c2"/><itemref idref="c1"/></spine></package>')
        z.writestr("OEBPS/one.xhtml", "<html><body><p>Kapitel eins</p></body></html>")
        z.writestr("OEBPS/two.xhtml", "<html><body><p>Kapitel zwei</p></body></html>")
    assert load_text(p) == "Kapitel zwei\n\nKapitel eins"


def test_eml(tmp_path: Path):
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["Subject"] = "Angebot"
    msg["From"] = "a@example.com"
    msg["To"] = "b@example.com"
    msg.set_content("Hier ist das Angebot.")
    msg.add_attachment(b"data", maintype="application", subtype="pdf", filename="angebot.pdf")
    (tmp_path / "m.eml").write_bytes(bytes(msg))
    text = load_text(tmp_path / "m.eml")
    assert text.startswith("Subject: Angebot\nFrom: a@example.com")
    assert "Hier ist das Angebot." in text and "Attachments: angebot.pdf" in text


def test_code_files_are_found(tmp_path: Path):
    (tmp_path / "main.py").write_text("def hello():\n    return 1\n")
    (tmp_path / "config.yaml").write_text("key: value\n")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    assert [p.name for p in iter_files(tmp_path)] == ["config.yaml", "main.py"]
    assert "def hello" in load_text(tmp_path / "main.py")
