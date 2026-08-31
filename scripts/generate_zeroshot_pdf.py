"""
generate_zeroshot_pdf.py
========================
Generates a publication-quality PDF report named 'zeroshot1.pdf'
summarizing Zero-Shot vs. Trained ASR Models on the Bhojpuri dataset,
including dataset row counts, columns/features, and audio duration.
"""
from pathlib import Path
import shutil

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.pdfgen import canvas

# Output paths
ROOT_DIR = Path("f:/bhojpuri-AI")
REPORT_DIR = ROOT_DIR / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PDF_OUTPUT_PATH = REPORT_DIR / "zeroshot1.pdf"
ROOT_PDF_PATH = ROOT_DIR / "zeroshot1.pdf"

# Color Palette
DEEP_BLUE  = colors.HexColor('#1E40AF')
SLATE_GRAY = colors.HexColor('#64748B')
TEXT_DARK  = colors.HexColor('#1E293B')
BORDER_CLR = colors.HexColor('#CBD5E1')
GRID_CLR   = colors.HexColor('#E2E8F0')
LIGHT_BG   = colors.HexColor('#F8FAFC')
GREEN_WIN  = colors.HexColor('#DCFCE7')
GOLD_REC   = colors.HexColor('#FEF3C7')

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_footer(num_pages)
            super().showPage()
        super().save()

    def draw_footer(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 7.5)
        self.setFillColor(SLATE_GRAY)
        self.setStrokeColor(BORDER_CLR)
        self.setLineWidth(0.5)
        self.line(36, 26, 576, 26)
        self.drawString(36, 16, "Bhojpuri ASR Research | Zero-Shot & Trained Models Specification Report (zeroshot1)")
        self.drawRightString(576, 16, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()

def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_OUTPUT_PATH),
        pagesize=letter,
        rightMargin=32,
        leftMargin=32,
        topMargin=30,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13.5,
        leading=16.5,
        textColor=DEEP_BLUE,
        spaceAfter=2
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=10.5,
        textColor=SLATE_GRAY,
        spaceAfter=4
    )
    h1_style = ParagraphStyle(
        'H1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.2,
        leading=11.5,
        textColor=DEEP_BLUE,
        spaceBefore=4,
        spaceAfter=2,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.8,
        leading=9.2,
        textColor=TEXT_DARK,
        spaceAfter=3
    )
    th_style = ParagraphStyle(
        'TH_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=6.6,
        leading=8.2,
        textColor=colors.white,
        alignment=1
    )
    td_style = ParagraphStyle(
        'TD_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.4,
        leading=8.0,
        textColor=TEXT_DARK
    )
    td_bold = ParagraphStyle(
        'TDBold_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=6.4,
        leading=8.0,
        textColor=TEXT_DARK
    )
    td_center = ParagraphStyle(
        'TDCenter_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.4,
        leading=8.0,
        textColor=TEXT_DARK,
        alignment=1
    )

    story = []

    # Title & Banner
    story.append(Paragraph("BHOJPURI ASR: ZERO-SHOT VS. TRAINED MODELS BENCHMARK REPORT", title_style))
    story.append(Paragraph("Empirical Word Error Rate (WER) Evaluation, Dataset Rows, and Feature Specifications", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.2, color=DEEP_BLUE, spaceBefore=0, spaceAfter=4))

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary & Evaluation Overview", h1_style))
    exec_summary = (
        "This evaluation report presents a comprehensive comparison of <b>Zero-Shot ASR ability</b> versus <b>Locally Trained and Fine-Tuned Models</b> "
        "on a target Bhojpuri evaluation dataset (<code>data/merged_bhojpuri/eval</code>). A total of <b>1,054 audio test rows</b> "
        "comprising both clean studio speech (610 rows) and crowdsourced rural mobile recordings (444 rows) were evaluated using <b>Word Error Rate (WER)</b>."
    )
    story.append(Paragraph(exec_summary, body_style))

    # 2. Master WER & Dataset Comparison Table
    story.append(Paragraph("2. Official Model Benchmark, Dataset Rows & WER Results Table", h1_style))
    
    benchmark_table_data = [
        [
            Paragraph("<b>Model Name</b>", th_style),
            Paragraph("<b>Training Paradigm</b>", th_style),
            Paragraph("<b>Dataset Rows & Duration</b>", th_style),
            Paragraph("<b>Trainable Params</b>", th_style),
            Paragraph("<b>Final WER (%) &darr;</b>", th_style),
            Paragraph("<b>Best Checkpoint</b>", th_style),
            Paragraph("<b>Status / Remarks</b>", th_style)
        ],
        [
            Paragraph("<b>Whisper-Small (Zero-Shot)</b>", td_bold),
            Paragraph("Zero-Shot Baseline (Untrained)", td_style),
            Paragraph("<b>0 Rows (0 Hours)</b><br/>Pre-trained: 680k h web", td_center),
            Paragraph("0 (0%)", td_center),
            Paragraph("<b>~71.50% &ndash; 75.00%</b>", td_center),
            Paragraph("&mdash;", td_center),
            Paragraph("Baseline reference out-of-the-box", td_style)
        ],
        [
            Paragraph("<b>Vakyansh Wav2Vec 2.0</b>", td_bold),
            Paragraph("Acoustic Model (CTC, bhom_60)", td_style),
            Paragraph("<b>~36,000 Rows</b><br/>(~60.0 Hours)", td_center),
            Paragraph("~95M (100%)", td_center),
            Paragraph("<b>108.22%</b>", td_center),
            Paragraph("102.96% Studio<br/>114.88% Field", td_center),
            Paragraph("High WER due to domain mismatch & CTC insertion errors", td_style)
        ],
        [
            Paragraph("<b>Whisper-Small (Full FT)</b>", td_bold),
            Paragraph("Full Parameter Fine-Tuning", td_style),
            Paragraph("<b>103,746 Rows</b><br/>(~150.2 Hours)", td_center),
            Paragraph("~242.6M (100%)", td_center),
            Paragraph("<b>40.58%</b>", td_center),
            Paragraph("40.58% (Step 14,900)", td_center),
            Paragraph("Strong linguistic adaptation on 103,746 training rows", td_style)
        ],
        [
            Paragraph("<b>🏆 Whisper-Small + LoRA v1</b>", td_bold),
            Paragraph("PEFT / LoRA (r=8, q+v)", td_style),
            Paragraph("<b>103,746 Rows</b><br/>(~150.2 Hours)", td_center),
            Paragraph("<b>884.7K (0.36%)</b>", td_center),
            Paragraph("<b>38.91%</b>", td_center),
            Paragraph("<b>38.86% (Step 1,800)</b>", td_center),
            Paragraph("<b>Best overall accuracy; +1.67% improvement over full FT</b>", td_bold)
        ],
        [
            Paragraph("<b>Whisper-Small + LoRA v2</b>", td_bold),
            Paragraph("PEFT / LoRA (r=16, q+k+v+out)", td_style),
            Paragraph("<b>103,746 Rows</b><br/>(~150.2 Hours)", td_center),
            Paragraph("~1.77M (0.73%)", td_center),
            Paragraph("<b>38.91%</b>", td_center),
            Paragraph("38.88% (Step 600)", td_center),
            Paragraph("Tied v1 accuracy; early convergence at step 600", td_style)
        ]
    ]

    t_bench = Table(benchmark_table_data, colWidths=[95, 95, 78, 60, 58, 65, 97])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DEEP_BLUE),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_CLR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, GRID_CLR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ('BACKGROUND', (0, 4), (-1, 4), GREEN_WIN),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(t_bench)
    story.append(Spacer(1, 4))

    # 3. Dataset Features & Columns Breakdown
    story.append(Paragraph("3. Dataset Schema, Columns & Evaluation Split Breakdown", h1_style))
    dataset_summary = (
        "The training and benchmark datasets contain the following metadata columns and feature schemas:"
    )
    story.append(Paragraph(dataset_summary, body_style))

    data_table_content = [
        [
            Paragraph("<b>Dataset Corpus</b>", th_style),
            Paragraph("<b>Row Count / Duration</b>", th_style),
            Paragraph("<b>Columns / Feature Schema (Fields)</b>", th_style),
            Paragraph("<b>Target Domain & Script</b>", th_style)
        ],
        [
            Paragraph("<b>Whisper Training Split</b><br/>(AI4Bharat + SYSPIN)", td_bold),
            Paragraph("<b>103,746 Rows</b><br/>(~150.2 Hours audio)", td_center),
            Paragraph("<b>6 Columns:</b> <code>audio</code> (16kHz array), <code>file</code> (wav path), <code>transcript</code> (Devanagari text), <code>duration</code> (sec), <code>speaker_id</code>, <code>topic_domain</code>", td_style),
            Paragraph("6 Topic Domains (Agriculture, Health, Politics, Finance, Food, General) in Devanagari", td_style)
        ],
        [
            Paragraph("<b>Vakyansh Pre-training</b><br/>(ULCA / SYSPIN)", td_bold),
            Paragraph("<b>~36,000 Rows</b><br/>(~60.0 Hours audio)", td_center),
            Paragraph("<b>3 Columns:</b> <code>audio_filepath</code> (16kHz 1D waveform), <code>duration</code> (sec), <code>text</code> (65 Devanagari character tokens)", td_style),
            Paragraph("Pure character-level Devanagari vocabulary (65 tokens, no language model)", td_style)
        ],
        [
            Paragraph("<b>Merged Benchmark Eval Split</b><br/>(<code>data/merged_bhojpuri/eval</code>)", td_bold),
            Paragraph("<b>1,054 Rows</b><br/>(~4.8 GB Arrow cache)", td_center),
            Paragraph("<b>610 Studio Rows</b> (IISc SYSPIN) + <b>444 Mobile Rows</b> (AI4Bharat Rural Women)<br/><b>Columns:</b> <code>audio</code>, <code>file</code>, <code>transcript</code>, <code>normalized_text</code>", td_style),
            Paragraph("Ground-truth normalized Bhojpuri Devanagari test split", td_style)
        ]
    ]

    t_data = Table(data_table_content, colWidths=[110, 80, 220, 138])
    t_data.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DEEP_BLUE),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_CLR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, GRID_CLR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(t_data)
    story.append(Spacer(1, 4))

    # 4. Key Analytical Insights
    story.append(Paragraph("4. Key Scientific Insights & Conclusion", h1_style))
    insights = [
        "&bull; <b>Impact of 103,746 Training Rows:</b> Moving from 0 rows (Zero-Shot ~75% WER) to 103,746 rows reduced WER to <b>40.58%</b> via Full Fine-Tuning.",
        "&bull; <b>LoRA Parameter & Sample Efficiency:</b> LoRA achieved the lowest project WER of <b>38.91%</b> while updating only <b>884,736 parameters (0.36%)</b> on the exact same 103,746 dataset rows.",
        "&bull; <b>CTC vs. Autoregressive Model Divergence:</b> The Vakyansh CTC model (trained on ~36,000 rows) scored 108.22% WER due to heavy character insertion errors on noisy mobile speech."
    ]
    for ins in insights:
        story.append(Paragraph(ins, body_style))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    
    # Also copy to root directory
    shutil.copy(PDF_OUTPUT_PATH, ROOT_PDF_PATH)
    print(f"[SUCCESS] Generated zeroshot1 PDF successfully at:\n  - {PDF_OUTPUT_PATH}\n  - {ROOT_PDF_PATH}")

if __name__ == "__main__":
    build_pdf()
