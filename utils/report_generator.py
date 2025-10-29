import io
from datetime import datetime
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # Non-GUI backend
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle


def generate_pdf_report(
    original_path: str,
    enhanced_path: str,
    metrics: Dict[str, float],
    parameters: Dict[str, object],
    output_path: str
) -> str:
    """
    Generate a professional PDF report summarizing image enhancement results.

    Parameters
    ----------
    original_path : str
        Path to the original image file.
    enhanced_path : str
        Path to the enhanced image file.
    metrics : Dict[str, float]
        Dictionary containing 'psnr', 'mse', 'ssim'.
    parameters : Dict[str, object]
        Parameters used during enhancement (mode, clip_limit, etc.).
    output_path : str
        Output path for the PDF file.

    Returns
    -------
    str
        Path to the saved PDF file.
    """
    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    title = Paragraph("<b>Image Enhancement Report</b>", styles["Title"])
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Metadata
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_text = f"Generated on: {timestamp}<br/>Method: {parameters.get('method_used', parameters.get('mode', 'N/A'))}<br/>Use Case: {parameters.get('use_case', 'N/A')}"
    elements.append(Paragraph(meta_text, styles["Normal"]))
    elements.append(Spacer(1, 12))

    # Parameters table
    param_items = sorted(parameters.items(), key=lambda kv: kv[0])
    param_data = [["Parameter", "Value"]] + [[str(k), str(v)] for k, v in param_items]
    param_table = Table(param_data, hAlign="LEFT")
    param_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263238")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    elements.append(Paragraph("<b>Parameters</b>", styles["Heading2"]))
    elements.append(param_table)
    elements.append(Spacer(1, 12))

    # Images: side-by-side
    elements.append(Paragraph("<b>Original vs Enhanced</b>", styles["Heading2"]))
    # Resize to fit width (A4 width ~ 595 points, margins considered)
    img_width = (A4[0] - 72) / 2 - 12  # two images per row with spacing

    img1 = Image(original_path, width=img_width, height=img_width * 0.75)  # approximate aspect
    img2 = Image(enhanced_path, width=img_width, height=img_width * 0.75)

    img_table = Table([[img1, img2]], hAlign="CENTER", colWidths=[img_width, img_width])
    elements.append(img_table)
    elements.append(Spacer(1, 12))

    # Metrics table
    elements.append(Paragraph("<b>Metrics</b>", styles["Heading2"]))
    metrics_data = [
        ["Metric", "Value"],
        ["PSNR (dB)", f"{metrics.get('psnr', 0):.3f}"],
        ["MSE", f"{metrics.get('mse', 0):.6f}"],
        ["SSIM", f"{metrics.get('ssim', 0):.4f}"],
    ]
    metrics_table = Table(metrics_data, hAlign="LEFT")
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#00695C")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#E0F2F1")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    elements.append(metrics_table)
    elements.append(Spacer(1, 12))

    # Metrics bar chart (matplotlib)
    labels = ["PSNR", "MSE", "SSIM"]
    values = [metrics.get("psnr", 0.0), metrics.get("mse", 0.0), metrics.get("ssim", 0.0)]
    colors_bar = ["#1976D2", "#EF5350", "#43A047"]

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(labels, values, color=colors_bar)
    ax.set_title("Metric Performance")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval, f"{yval:.3f}", va="bottom", ha="center", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", dpi=150)
    plt.close(fig)
    buf.seek(0)
    chart_img = Image(buf, width=A4[0] - 72, height=(A4[0] - 72) * 0.4)
    elements.append(Paragraph("<b>Metrics Chart</b>", styles["Heading2"]))
    elements.append(chart_img)

    # Build document
    doc.build(elements)
    buf.close()

    return output_path