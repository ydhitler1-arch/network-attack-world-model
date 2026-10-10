"""Minimal Markdown -> PDF (headings, paragraphs, bullets, tables, images, fenced code) with reportlab.

    python docs/md2pdf.py docs/ARCHITECTURE.md docs/ARCHITECTURE.pdf [title]
"""
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, ListFlowable, ListItem, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ss = getSampleStyleSheet()
BODY = ParagraphStyle("b", parent=ss["Normal"], fontSize=8.8, leading=11.6, spaceAfter=3.5)
CELL = ParagraphStyle("c", parent=BODY, fontSize=7.6, leading=9.4, spaceAfter=0)
CELLH = ParagraphStyle("ch", parent=CELL, fontName="Helvetica-Bold")
H1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=16, leading=19, alignment=0, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11, leading=13, spaceBefore=6, spaceAfter=2)
H3 = ParagraphStyle("h3", parent=ss["Heading3"], fontSize=9.5, spaceBefore=4, spaceAfter=2)
CODE = ParagraphStyle("code", parent=BODY, fontName="Courier", fontSize=7.4, leading=9)


def inline(t):
    t = escape(t)
    t = re.sub(r"`([^`]+)`", r'<font face="Courier" size="8">\1</font>', t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])", r"<i>\1</i>", t)
    return t


def table(rows):
    rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    rows = [r for i, r in enumerate(rows) if not (i == 1 and all(re.fullmatch(r":?-+:?", c) for c in r))]
    data = [[Paragraph(inline(c), CELLH if i == 0 else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8ecf2")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b7c3")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return [t, Spacer(1, 4)]


def convert(src, out, title="Document", author="", margins=(16, 16, 14, 14)):
    base = Path(src).parent
    lines = Path(src).read_text(encoding="utf-8").split("\n")
    story, i = [], 0
    avail_w = A4[0] - (margins[0] + margins[1]) * mm
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
        elif ln.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            story += [Preformatted("\n".join(lines[i + 1:j]), CODE), Spacer(1, 3)]
            i = j + 1
        elif ln.startswith("# "):
            story.append(Paragraph(inline(ln[2:]), H1)); i += 1
        elif ln.startswith("## "):
            story.append(Paragraph(inline(ln[3:]), H2)); i += 1
        elif ln.startswith("### "):
            story.append(Paragraph(inline(ln[4:]), H3)); i += 1
        elif (m := re.match(r"!\[[^\]]*\]\(([^)]+)\)", ln)):
            from reportlab.lib.utils import ImageReader
            p = base / m.group(1)
            w, h = ImageReader(str(p)).getSize()
            story += [Image(str(p), width=avail_w, height=avail_w * h / w), Spacer(1, 3)]
            i += 1
        elif ln.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            story += table(lines[i:j]); i = j
        elif ln.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                txt = lines[i][2:]
                i += 1
                while i < len(lines) and lines[i].startswith("  ") and lines[i].strip():
                    txt += " " + lines[i].strip(); i += 1
                items.append(ListItem(Paragraph(inline(txt), BODY), leftIndent=10))
            story.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=10, bulletFontSize=7))
        else:
            txt = ln.strip(); i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"(#|\||- |!\[|```)", lines[i]):
                txt += " " + lines[i].strip(); i += 1
            story.append(Paragraph(inline(txt), BODY))

    def footer(c, d):
        c.setFont("Helvetica", 7.5); c.setFillColor(colors.grey)
        c.drawCentredString(A4[0] / 2, 8 * mm, f"{title}  |  page {d.page}")

    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=margins[0] * mm, rightMargin=margins[1] * mm,
                            topMargin=margins[2] * mm, bottomMargin=margins[3] * mm + 4 * mm, title=title, author=author)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return doc.page


if __name__ == "__main__":
    n = convert(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "Document")
    print(f"{sys.argv[2]}: {n} pages")
