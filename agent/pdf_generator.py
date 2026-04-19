
from io import BytesIO
from datetime import datetime
from typing import Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)


# ── Color Palette ─────────────────────────────────────────────────────────────
NAVY       = colors.HexColor("#0f172a")
DARK_BLUE  = colors.HexColor("#1e3a5f")
ACCENT     = colors.HexColor("#3b82f6")
LIGHT_GRAY = colors.HexColor("#f1f5f9")
MID_GRAY   = colors.HexColor("#94a3b8")
DARK_GRAY  = colors.HexColor("#334155")
WHITE      = colors.white
GREEN      = colors.HexColor("#16a34a")
AMBER      = colors.HexColor("#d97706")
RED        = colors.HexColor("#dc2626")
GREEN_LIGHT = colors.HexColor("#dcfce7")
AMBER_LIGHT = colors.HexColor("#fef3c7")
RED_LIGHT   = colors.HexColor("#fee2e2")


# ── Style Builder ─────────────────────────────────────────────────────────────

def _build_styles():
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "title",
            fontName="Helvetica-Bold",
            fontSize=26,
            textColor=WHITE,
            leading=32,
            alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName="Helvetica",
            fontSize=11,
            textColor=colors.HexColor("#93c5fd"),
            leading=16,
            alignment=TA_LEFT,
        ),
        "section_header": ParagraphStyle(
            "section_header",
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=ACCENT,
            leading=18,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=10,
            textColor=DARK_GRAY,
            leading=15,
            alignment=TA_JUSTIFY,
            spaceAfter=4,
        ),
        "body_small": ParagraphStyle(
            "body_small",
            fontName="Helvetica",
            fontSize=9,
            textColor=MID_GRAY,
            leading=13,
        ),
        "label": ParagraphStyle(
            "label",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=MID_GRAY,
            leading=12,
        ),
        "value": ParagraphStyle(
            "value",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=DARK_GRAY,
            leading=14,
        ),
        "price_big": ParagraphStyle(
            "price_big",
            fontName="Helvetica-Bold",
            fontSize=28,
            textColor=ACCENT,
            leading=34,
            alignment=TA_CENTER,
        ),
        "price_range": ParagraphStyle(
            "price_range",
            fontName="Helvetica",
            fontSize=10,
            textColor=MID_GRAY,
            leading=14,
            alignment=TA_CENTER,
        ),
        "rec_text": ParagraphStyle(
            "rec_text",
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
        ),
        "disclaimer": ParagraphStyle(
            "disclaimer",
            fontName="Helvetica-Oblique",
            fontSize=8,
            textColor=RED,
            leading=12,
            alignment=TA_JUSTIFY,
        ),
        "footer": ParagraphStyle(
            "footer",
            fontName="Helvetica",
            fontSize=8,
            textColor=MID_GRAY,
            alignment=TA_CENTER,
        ),
        "comp_header": ParagraphStyle(
            "comp_header",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=WHITE,
            leading=12,
            alignment=TA_CENTER,
        ),
        "comp_cell": ParagraphStyle(
            "comp_cell",
            fontName="Helvetica",
            fontSize=9,
            textColor=DARK_GRAY,
            leading=12,
            alignment=TA_CENTER,
        ),
    }
    return styles


# ── Helper Flowables ──────────────────────────────────────────────────────────

def _divider(color=ACCENT, thickness=0.5, space=8):
    return [
        Spacer(1, space),
        HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=space),
    ]


def _kv_table(pairs: list[tuple], styles: dict) -> Table:

    data = [[Paragraph(k, styles["label"]), Paragraph(v, styles["value"])] for k, v in pairs]
    t = Table(data, colWidths=[2.2 * inch, 4.3 * inch])
    t.setStyle(TableStyle([
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT_GRAY, WHITE]),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return t


# ── Colored Box ───────────────────────────────────────────────────────────────

def _colored_box(content_elements, bg_color, border_color, margin=0.1):

    t = Table([[content_elements]], colWidths=[6.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), bg_color),
        ("BOX",          (0, 0), (-1, -1), 1.5, border_color),
        ("LEFTPADDING",  (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
        ("TOPPADDING",   (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 14),
        ("ROUNDEDCORNERS", [6]),
    ]))
    return t


# ── Main Generator ────────────────────────────────────────────────────────────

def generate_pdf_report(
    report_data: dict,
    property_features: Optional[dict] = None,
) -> bytes:

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.75 * inch,
        title="AI Real Estate Advisory Report",
        author="AI Real Estate Advisor",
    )

    styles = _build_styles()
    story = []
    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    # ── HEADER BANNER ─────────────────────────────────────────────────────────
    header_data = [[
        Paragraph("🏡 AI Real Estate Advisory Report", styles["title"]),
        Paragraph(f"Generated: {now}", styles["subtitle"]),
    ]]
    header_table = Table(header_data, colWidths=[6.5 * inch])
    header_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), NAVY),
        ("LEFTPADDING",  (0, 0), (-1, -1), 24),
        ("RIGHTPADDING", (0, 0), (-1, -1), 24),
        ("TOPPADDING",   (0, 0), (-1, -1), 20),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 20),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 16))

    # ── SECTION 1: VALUATION ──────────────────────────────────────────────────
    story.append(Paragraph("1. Property Valuation", styles["section_header"]))
    story += _divider()

    val_raw = report_data.get("valuation_summary", "")
    lines = val_raw.split("\n")
    price_str  = lines[0].replace("Estimated Market Value:", "").replace("**", "").strip() if lines else "—"
    range_str  = lines[1].strip() if len(lines) > 1 else ""
    conf_str   = lines[2].strip() if len(lines) > 2 else ""

    val_inner = [
        Paragraph(price_str, styles["price_big"]),
        Paragraph(range_str, styles["price_range"]),
        Spacer(1, 4),
        Paragraph(conf_str, styles["price_range"]),
    ]
    story.append(_colored_box(val_inner, colors.HexColor("#eff6ff"), ACCENT))
    story.append(Spacer(1, 14))

    # Property details table
    if property_features:
        roof_names = {0: 'Flat', 1: 'Gable', 2: 'Gambrel', 3: 'Hip', 4: 'Mansard', 5: 'Shed'}
        pairs = [
            ("Overall Quality",      f"{property_features.get('overall_qual', '—')}/10"),
            ("Living Area",          f"{property_features.get('gr_liv_area', 0):,.0f} sq ft"),
            ("Lot Area",             f"{property_features.get('lot_area', 0):,.0f} sq ft"),
            ("Year Built",           str(property_features.get('year_built', '—'))),
            ("Remodel Year",         str(property_features.get('year_remod', '—'))),
            ("Garage Area",          f"{property_features.get('garage_area', 0):,.0f} sq ft"),
            ("Full Bathrooms",       str(property_features.get('full_bath', '—'))),
            ("Fireplaces",           str(property_features.get('fireplaces', '—'))),
            ("Central Air",          "Yes" if property_features.get('central_air', 1) == 1 else "No"),
            ("Finished Basement",    f"{property_features.get('bsmt_fin_sf1', 0):,.0f} sq ft"),
            ("Roof Style",           roof_names.get(property_features.get('roof_style', 1), 'Gable')),
        ]
        story.append(Paragraph("Property Details", styles["body"]))
        story.append(Spacer(1, 4))
        story.append(_kv_table(pairs, styles))
        story.append(Spacer(1, 12))

    # ── SECTION 2: MARKET VIEW ────────────────────────────────────────────────
    story.append(Paragraph("2. Market Analysis", styles["section_header"]))
    story += _divider()
    market_text = report_data.get("market_view", "No market data available.")
    story.append(Paragraph(market_text, styles["body"]))
    story.append(Spacer(1, 14))

    # ── SECTION 3: COMPARABLE SALES ───────────────────────────────────────────
    story.append(Paragraph("3. Comparable Sales", styles["section_header"]))
    story += _divider()

    comps_raw = report_data.get("comparable_sales", "")
    comp_lines = [l for l in comps_raw.split("\n") if l.strip() and l.strip().startswith("•")]

    if comp_lines:
        # Build a proper table
        table_data = [[
            Paragraph("Sale Price",   styles["comp_header"]),
            Paragraph("Area (sqft)",  styles["comp_header"]),
            Paragraph("Quality",      styles["comp_header"]),
            Paragraph("Year Built",   styles["comp_header"]),
            Paragraph("Garage (sqft)",styles["comp_header"]),
            Paragraph("Match %",      styles["comp_header"]),
        ]]
        for line in comp_lines:

            parts = [p.strip() for p in line.lstrip("• ").split("|")]
            if len(parts) >= 5:
                row = [Paragraph(p, styles["comp_cell"]) for p in parts[:6]]
                while len(row) < 6:
                    row.append(Paragraph("—", styles["comp_cell"]))
                table_data.append(row)

        comp_table = Table(
            table_data,
            colWidths=[1.1*inch, 0.95*inch, 0.75*inch, 0.85*inch, 1.0*inch, 0.85*inch],
        )
        comp_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), DARK_BLUE),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT_GRAY, WHITE]),
            ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ]))
        story.append(comp_table)
    else:
        story.append(Paragraph(comps_raw or "No comparable data available.", styles["body"]))

    story.append(Spacer(1, 14))

    # ── SECTION 4: RECOMMENDATION ─────────────────────────────────────────────
    story.append(Paragraph("4. Investment Recommendation", styles["section_header"]))
    story += _divider()

    rec = report_data.get("recommendation", "HOLD").upper()
    score = float(report_data.get("investment_score", 5.0))
    reasoning = report_data.get("advisory_reasoning", "—")

    rec_configs = {
        "BUY":   ("BUY",   GREEN, GREEN_LIGHT),
        "AVOID": ("AVOID", RED,   RED_LIGHT),
        "HOLD":  ("HOLD",  AMBER, AMBER_LIGHT),
    }
    rec_label, rec_color, rec_bg = rec_configs.get(rec, rec_configs["HOLD"])

    rec_style = ParagraphStyle(
        "rec_dyn",
        parent=styles["rec_text"],
        textColor=rec_color,
    )
    rec_inner = [
        Paragraph(rec_label, rec_style),
        Spacer(1, 6),
        Paragraph(f"Investment Score: {score}/10", styles["price_range"]),
    ]
    story.append(KeepTogether([_colored_box(rec_inner, rec_bg, rec_color)]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Advisory Reasoning:", styles["label"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(reasoning, styles["body"]))
    story.append(Spacer(1, 10))

    # Section 5: Disclaimer
    story += _divider(color=RED, thickness=0.3)
    story.append(Paragraph("Legal Disclaimer", styles["section_header"]))
    disclaimer = report_data.get(
        "disclaimer",
        "This report is generated by an AI system for informational purposes only. "
        "It does not constitute a licensed real estate appraisal, financial advice, "
        "or a solicitation to buy or sell property. Always consult a licensed "
        "professional before making investment decisions.",
    )
    story.append(Paragraph(disclaimer, styles["disclaimer"]))
    story.append(Spacer(1, 16))

    # ── FOOTER ────────────────────────────────────────────────────────────────
    story += _divider(color=MID_GRAY, thickness=0.3, space=4)
    story.append(Paragraph(
        f"AI Real Estate Advisor · Milestone 2 · LangGraph · ChromaDB RAG · GPT-4o-mini  |  {now}",
        styles["footer"],
    ))

    doc.build(story)
    return buffer.getvalue()
