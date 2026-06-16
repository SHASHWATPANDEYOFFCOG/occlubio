"""Render REPORT.md to REPORT.pdf (pure-Python: markdown + xhtml2pdf, no system binaries).

    pip install -e ".[report]"
    python scripts/build_report.py            # REPORT.md -> REPORT.pdf
    python scripts/build_report.py IN.md OUT.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import markdown
from xhtml2pdf import pisa

CSS = """
@page { size: a4; margin: 1.8cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10pt; color: #111; line-height: 1.4; }
h1 { font-size: 19pt; color: #13315c; margin-bottom: 2px; }
h2 { font-size: 13.5pt; color: #1769ff; border-bottom: 1px solid #ccc; padding-bottom: 2px; margin-top: 16px; }
h3 { font-size: 11pt; color: #333; }
p, li { font-size: 10pt; }
code { background: #f2f2f2; font-family: Courier, monospace; font-size: 8.5pt; }
pre { background: #f5f5f5; padding: 6px; border: 1px solid #ddd; font-size: 8pt; }
table { border-collapse: collapse; width: 100%; margin: 6px 0; }
th, td { border: 1px solid #999; padding: 3px 5px; font-size: 8.5pt; text-align: left; }
th { background: #eef2ff; }
a { color: #1769ff; text-decoration: none; }
"""


def main():
    md_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("REPORT.md")
    pdf_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("REPORT.pdf")
    body = markdown.markdown(
        md_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    with open(pdf_path, "wb") as f:
        result = pisa.CreatePDF(html, dest=f, encoding="utf-8")
    if result.err:
        raise SystemExit(f"PDF generation failed with {result.err} error(s)")
    print(f"wrote {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
