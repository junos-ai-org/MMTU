#!/usr/bin/env python3
"""Convert markdown reports to styled .docx files with dark gray text and proper headings."""

import re
import sys
from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn


DARK_GRAY = RGBColor(0x33, 0x33, 0x33)
HEADING_COLOR = RGBColor(0x2B, 0x2B, 0x2B)
CODE_GRAY = RGBColor(0x44, 0x44, 0x44)
CODE_BG = "F5F5F5"


def set_run_style(run, color=DARK_GRAY, size=Pt(11), font_name="Calibri", bold=False, italic=False):
    run.font.color.rgb = color
    run.font.size = size
    run.font.name = font_name
    run.bold = bold
    run.italic = italic


def add_styled_heading(doc, text, level):
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        run.font.color.rgb = HEADING_COLOR
        run.font.name = "Calibri"
    return heading


def add_table(doc, header_row, data_rows):
    """Add a formatted table to the document."""
    ncols = len(header_row)
    table = doc.add_table(rows=1, cols=ncols, style="Table Grid")
    table.autofit = True

    # Header
    for i, cell_text in enumerate(header_row):
        cell = table.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        run = p.add_run(cell_text.strip())
        set_run_style(run, color=HEADING_COLOR, size=Pt(10), bold=True)
        # Light gray background for header
        shading = cell._element.get_or_add_tcPr()
        shd = shading.makeelement(qn("w:shd"), {
            qn("w:fill"): "E8E8E8",
            qn("w:val"): "clear",
        })
        shading.append(shd)

    # Data rows
    for row_data in data_rows:
        row = table.add_row()
        for i, cell_text in enumerate(row_data):
            if i < ncols:
                cell = row.cells[i]
                cell.text = ""
                p = cell.paragraphs[0]
                text = cell_text.strip()
                # Handle bold markers
                if text.startswith("**") and text.endswith("**"):
                    run = p.add_run(text.strip("*"))
                    set_run_style(run, size=Pt(10), bold=True)
                else:
                    run = p.add_run(text)
                    set_run_style(run, size=Pt(10))

    return table


def parse_inline(paragraph, text, base_size=Pt(11)):
    """Parse inline markdown (bold, italic, code) and add runs to paragraph."""
    # Split on bold, italic, and code patterns
    parts = re.split(r'(\*\*.*?\*\*|`[^`]+`|\*[^*]+\*)', text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            set_run_style(run, size=base_size, bold=True)
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            set_run_style(run, color=CODE_GRAY, size=Pt(10), font_name="Consolas")
        elif part.startswith("*") and part.endswith("*") and not part.startswith("**"):
            run = paragraph.add_run(part[1:-1])
            set_run_style(run, size=base_size, italic=True)
        else:
            run = paragraph.add_run(part)
            set_run_style(run, size=base_size)


def md_to_docx(md_path: Path, docx_path: Path):
    """Convert a markdown file to a styled .docx document."""
    doc = Document()

    # Set default font for the document
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Calibri"
    font.size = Pt(11)
    font.color.rgb = DARK_GRAY

    # Also style heading defaults
    for level in range(1, 4):
        h_style = doc.styles[f"Heading {level}"]
        h_style.font.name = "Calibri"
        h_style.font.color.rgb = HEADING_COLOR

    lines = md_path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # Skip horizontal rules
        if re.match(r'^-{3,}$', line.strip()):
            i += 1
            continue

        # Headings
        heading_match = re.match(r'^(#{1,3})\s+(.*)', line)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            add_styled_heading(doc, text, level)
            i += 1
            continue

        # Code blocks
        if line.strip().startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            p = doc.add_paragraph()
            code_text = "\n".join(code_lines)
            run = p.add_run(code_text)
            set_run_style(run, color=CODE_GRAY, size=Pt(9), font_name="Consolas")
            # Add background shading
            pPr = p._element.get_or_add_pPr()
            shd = pPr.makeelement(qn("w:shd"), {
                qn("w:fill"): CODE_BG,
                qn("w:val"): "clear",
            })
            pPr.append(shd)
            continue

        # Tables
        if "|" in line and i + 1 < len(lines) and re.match(r'^\|[-|: ]+\|$', lines[i + 1].strip()):
            header_cells = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2  # skip header and separator
            data_rows = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip().startswith("|"):
                row_cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                data_rows.append(row_cells)
                i += 1
            add_table(doc, header_cells, data_rows)
            doc.add_paragraph()  # spacing after table
            continue

        # Bullet points
        bullet_match = re.match(r'^(\s*)[-*]\s+(.*)', line)
        if bullet_match:
            indent = len(bullet_match.group(1))
            text = bullet_match.group(2)
            p = doc.add_paragraph(style="List Bullet")
            if indent >= 2:
                p.style = doc.styles["List Bullet 2"] if "List Bullet 2" in [s.name for s in doc.styles] else doc.styles["List Bullet"]
                p.paragraph_format.left_indent = Inches(0.5)
            parse_inline(p, text)
            i += 1
            continue

        # Numbered list
        numbered_match = re.match(r'^(\d+)\.\s+(.*)', line)
        if numbered_match:
            text = numbered_match.group(2)
            p = doc.add_paragraph(style="List Number")
            parse_inline(p, text)
            i += 1
            continue

        # Empty lines
        if not line.strip():
            i += 1
            continue

        # Regular paragraph
        p = doc.add_paragraph()
        parse_inline(p, line)
        i += 1

    doc.save(str(docx_path))
    print(f"Wrote {docx_path}")


def main():
    if len(sys.argv) > 1:
        md_files = [Path(f) for f in sys.argv[1:]]
    else:
        # Default: convert all report .md files in this directory
        report_dir = Path(__file__).parent
        md_files = sorted(report_dir.glob("*_report.md")) + [report_dir / "synthesis.md"]
        md_files = [f for f in md_files if f.exists()]

    for md_file in md_files:
        docx_file = md_file.with_suffix(".docx")
        md_to_docx(md_file, docx_file)


if __name__ == "__main__":
    main()
