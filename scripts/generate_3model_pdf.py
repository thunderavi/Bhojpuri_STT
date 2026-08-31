"""
generate_3model_pdf.py
======================
Generates a publication-quality PDF report named '3MODEL.pdf'
containing 3 separate dedicated tables for each local Bhojpuri ASR model,
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

# Paths
ROOT_DIR = Path("f:/bhojpuri-AI")
REPORT_DIR = ROOT_DIR / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PDF_OUTPUT_PATH = REPORT_DIR / "3MODEL.pdf"
ROOT_PDF_PATH = ROOT_DIR / "3MODEL.pdf"

# Palette
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
        self.drawString(36, 16, "Bhojpuri ASR Research | 3 Local Models & Dataset Specification Report (3MODEL)")
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
    meta_box_style = ParagraphStyle(
        'MetaBox',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.8,
        leading=9.2,
        textColor=TEXT_DARK,
        spaceAfter=2
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

    # Title Banner
    story.append(Paragraph("BHOJPURI ASR: 3 LOCAL MODELS & DATASET SPECIFICATION REPORT", title_style))
    story.append(Paragraph("Detailed Checkpoint Progression, Dataset Rows, Column Features & Word Error Rate (WER) Analysis", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.2, color=DEEP_BLUE, spaceBefore=0, spaceAfter=4))

    # TABLE 1: Whisper Full Fine-Tuned
    story.append(Paragraph("Table 1: Model 1 &mdash; Whisper-Small (Full Fine-Tuned)", h1_style))
    meta_m1 = (
        "<b>Dataset Rows / Hours:</b> <b>103,746 Training Rows</b> (~150.2 Hours) | <b>Eval:</b> 1,054 Rows (~4.8 GB Arrow cache)<br/>"
        "<b>Dataset Columns (6 Features):</b> <code>audio</code> (16kHz waveform array), <code>file</code> (wav path), "
        "<code>transcript</code> (Devanagari text), <code>duration</code> (sec), <code>speaker_id</code>, <code>topic_domain</code><br/>"
        "<b>Architecture & Params:</b> Seq2Seq Transformer (242.6M Total, <b>242.6M / 100% Trainable</b>) | "
        "<b>Saved Path:</b> <code>models/bhojpuri-whisper-small-full</code>"
    )
    story.append(Paragraph(meta_m1, meta_box_style))
    
    t1_data = [
        [
            Paragraph("<b>Checkpoint (Step)</b>", th_style),
            Paragraph("<b>Epoch</b>", th_style),
            Paragraph("<b>Eval Loss</b>", th_style),
            Paragraph("<b>WER (%) &darr;</b>", th_style),
            Paragraph("<b>Status & Dataset Progression Notes</b>", th_style)
        ],
        [Paragraph("Step 100", td_bold), Paragraph("0.010", td_center), Paragraph("0.6940", td_center), Paragraph("71.50%", td_center), Paragraph("Initial adaptation starting point on 103,746 rows", td_style)],
        [Paragraph("Step 900", td_bold), Paragraph("0.086", td_center), Paragraph("0.3724", td_center), Paragraph("53.00%", td_center), Paragraph("Rapid early phonetic learning across Devanagari tokens", td_style)],
        [Paragraph("Step 1,700", td_bold), Paragraph("0.162", td_center), Paragraph("0.3420", td_center), Paragraph("49.19%", td_center), Paragraph("Broke sub-50% WER threshold", td_style)],
        [Paragraph("Step 3,000", td_bold), Paragraph("0.286", td_center), Paragraph("0.3301", td_center), Paragraph("49.46%", td_center), Paragraph("Stable convergence across multi-speaker acoustic rows", td_style)],
        [Paragraph("Step 5,200", td_bold), Paragraph("0.496", td_center), Paragraph("0.2985", td_center), Paragraph("45.28%", td_center), Paragraph("Significant vocabulary alignment on 6 topic domains", td_style)],
        [Paragraph("Step 7,700", td_bold), Paragraph("0.735", td_center), Paragraph("0.2813", td_center), Paragraph("43.48%", td_center), Paragraph("Steady loss reduction on telephony & studio speech", td_style)],
        [Paragraph("Step 8,700", td_bold), Paragraph("0.830", td_center), Paragraph("0.2776", td_center), Paragraph("42.74%", td_center), Paragraph("Improved complex phrasing and compound word accuracy", td_style)],
        [Paragraph("Step 10,500", td_bold), Paragraph("1.002", td_center), Paragraph("0.2668", td_center), Paragraph("41.66%", td_center), Paragraph("Passed all 103,746 training rows (1.0 full epoch)", td_style)],
        [Paragraph("Step 10,800", td_bold), Paragraph("1.030", td_center), Paragraph("0.2712", td_center), Paragraph("41.56%", td_center), Paragraph("Checkpoint-10800 saved on local disk", td_style)],
        [Paragraph("Step 13,600", td_bold), Paragraph("1.298", td_center), Paragraph("0.2633", td_center), Paragraph("41.65%", td_center), Paragraph("Minor loss fluctuations near global minimum", td_style)],
        [Paragraph("Step 14,500", td_bold), Paragraph("1.383", td_center), Paragraph("0.2604", td_center), Paragraph("41.06%", td_center), Paragraph("Approaching optimal cross-entropy loss", td_style)],
        [Paragraph("<b>🏆 Step 14,900</b>", td_bold), Paragraph("<b>1.422</b>", td_center), Paragraph("<b>0.2655</b>", td_center), Paragraph("<b>40.58%</b>", td_center), Paragraph("<b>All-Time Best Full Fine-Tuning Checkpoint (Saved on disk)</b>", td_bold)]
    ]
    t1 = Table(t1_data, colWidths=[80, 40, 50, 55, 323])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DEEP_BLUE),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_CLR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, GRID_CLR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ('BACKGROUND', (0, -1), (-1, -1), GOLD_REC),
        ('TOPPADDING', (0, 0), (-1, -1), 1.8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.8),
    ]))
    story.append(t1)
    story.append(Spacer(1, 4))

    # TABLE 2: Whisper + LoRA
    story.append(Paragraph("Table 2: Model 2 &mdash; Whisper-Small + LoRA (Parameter-Efficient PEFT)", h1_style))
    meta_m2 = (
        "<b>Dataset Rows / Hours:</b> Trained on the same <b>103,746 Rows</b> (~150h) starting from base Checkpoint-14900 | <b>Eval:</b> 1,054 Rows<br/>"
        "<b>Dataset Columns / Input Tensors:</b> <code>input_features</code> (80-channel log-Mel spectrogram), <code>labels</code> (BPE token IDs)<br/>"
        "<b>LoRA Config & Trainable Params:</b> Rank $r=8, \\alpha=16$, modules: <code>q_proj, v_proj</code> | <b>884,736 / 242.6M (0.36%) Trainable</b><br/>"
        "<b>Saved Path:</b> <code>models/LORAmodel/lora-merged-final</code>"
    )
    story.append(Paragraph(meta_m2, meta_box_style))
    
    t2_data = [
        [
            Paragraph("<b>Checkpoint (Step)</b>", th_style),
            Paragraph("<b>Epoch</b>", th_style),
            Paragraph("<b>Eval Loss</b>", th_style),
            Paragraph("<b>WER (%) &darr;</b>", th_style),
            Paragraph("<b>Status & LoRA Progression Notes</b>", th_style)
        ],
        [Paragraph("Step 100", td_bold), Paragraph("0.019", td_center), Paragraph("0.2591", td_center), Paragraph("40.59%", td_center), Paragraph("Initial LoRA baseline (inherited from Checkpoint-14900)", td_style)],
        [Paragraph("Step 300", td_bold), Paragraph("0.057", td_center), Paragraph("0.2546", td_center), Paragraph("39.30%", td_center), Paragraph("Rapid early adapter improvement across attention matrices", td_style)],
        [Paragraph("Step 700", td_bold), Paragraph("0.134", td_center), Paragraph("0.2530", td_center), Paragraph("38.99%", td_center), Paragraph("First time breaking sub-39% WER barrier in project", td_style)],
        [Paragraph("Step 800", td_bold), Paragraph("0.153", td_center), Paragraph("0.2521", td_center), Paragraph("39.33%", td_center), Paragraph("Stable validation checkpoint", td_style)],
        [Paragraph("Step 1,000", td_bold), Paragraph("0.191", td_center), Paragraph("0.2514", td_center), Paragraph("39.81%", td_center), Paragraph("Temporary greedy decoding variance", td_style)],
        [Paragraph("Step 1,400", td_bold), Paragraph("0.267", td_center), Paragraph("0.2502", td_center), Paragraph("38.89%", td_center), Paragraph("Approaching global peak accuracy", td_style)],
        [Paragraph("<b>🏆 Step 1,800</b>", td_bold), Paragraph("<b>0.343</b>", td_center), Paragraph("<b>0.2465</b>", td_center), Paragraph("<b>38.86%</b>", td_center), Paragraph("<b>All-Time Lowest WER across entire Bhojpuri project (Best Step)</b>", td_bold)],
        [Paragraph("Step 2,100", td_bold), Paragraph("0.401", td_center), Paragraph("0.2463", td_center), Paragraph("39.39%", td_center), Paragraph("Cosine learning rate scheduler entered decay phase", td_style)],
        [Paragraph("Step 2,700", td_bold), Paragraph("0.515", td_center), Paragraph("0.2465", td_center), Paragraph("39.13%", td_center), Paragraph("Sustained high accuracy plateau on test split", td_style)],
        [Paragraph("Step 3,500", td_bold), Paragraph("0.668", td_center), Paragraph("0.2471", td_center), Paragraph("40.17%", td_center), Paragraph("Completed 3,500 steps (lr decayed to 5.88e-08)", td_style)],
        [Paragraph("<b>🏆 Final Merged</b>", td_bold), Paragraph("<b>&mdash;</b>", td_center), Paragraph("<b>0.2465</b>", td_center), Paragraph("<b>38.91%</b>", td_center), Paragraph("<b>Final standalone fused model (+1.67% improvement over full FT)</b>", td_bold)]
    ]
    t2 = Table(t2_data, colWidths=[80, 40, 50, 55, 323])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DEEP_BLUE),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_CLR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, GRID_CLR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ('BACKGROUND', (0, 6), (-1, 6), GREEN_WIN),
        ('BACKGROUND', (0, -1), (-1, -1), GREEN_WIN),
        ('TOPPADDING', (0, 0), (-1, -1), 1.8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.8),
    ]))
    story.append(t2)
    story.append(Spacer(1, 4))

    # TABLE 3: Vakyansh Wav2Vec 2.0
    story.append(Paragraph("Table 3: Model 3 &mdash; Vakyansh Wav2Vec 2.0 (bhom_60 CTC)", h1_style))
    meta_m3 = (
        "<b>Dataset Rows / Hours:</b> Pre-trained on <b>~36,000 Rows (~60.0 Hours)</b> Bhojpuri speech (ULCA / IISc RESPINS & SYSPIN)<br/>"
        "<b>Dataset Columns (3 Features):</b> <code>audio_filepath</code> (16kHz 1D waveform), <code>duration</code> (sec), <code>text</code> (65 Devanagari character tokens)<br/>"
        "<b>Architecture & Params:</b> Wav2Vec 2.0 Base + CTC Head (~95M Total, <b>~95M Trainable</b>) | "
        "<b>Saved Path:</b> <code>models/vakyansh-bhojpuri</code>"
    )
    story.append(Paragraph(meta_m3, meta_box_style))
    
    t3_data = [
        [
            Paragraph("<b>Benchmark Test Split</b>", th_style),
            Paragraph("<b>Samples / Rows</b>", th_style),
            Paragraph("<b>WER (%) &darr;</b>", th_style),
            Paragraph("<b>Inference Speed</b>", th_style),
            Paragraph("<b>Acoustic & Error Characteristics</b>", th_style)
        ],
        [
            Paragraph("Studio Speech Split (IISc SYSPIN)", td_bold),
            Paragraph("610 Audio Rows (57.9%)", td_center),
            Paragraph("102.96%", td_center),
            Paragraph("~25 files/sec", td_center),
            Paragraph("Clean acoustic room; high character substitution & token insertion", td_style)
        ],
        [
            Paragraph("Field Mobile Split (AI4Bharat)", td_bold),
            Paragraph("444 Audio Rows (42.1%)", td_center),
            Paragraph("114.88%", td_center),
            Paragraph("~25 files/sec", td_center),
            Paragraph("High error rate on ambient village background & telephone bandwidth", td_style)
        ],
        [
            Paragraph("<b>Total Benchmark Eval Corpus</b>", td_bold),
            Paragraph("<b>1,054 Audio Rows</b>", td_center),
            Paragraph("<b>108.22%</b>", td_center),
            Paragraph("<b>⚡ ~25 files/sec</b>", td_center),
            Paragraph("<b>Evaluated on <code>data/merged_bhojpuri/eval</code> (~5x faster CTC throughput)</b>", td_bold)
        ]
    ]
    t3 = Table(t3_data, colWidths=[120, 80, 50, 70, 228])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DEEP_BLUE),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_CLR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, GRID_CLR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 1.8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.8),
    ]))
    story.append(t3)
    story.append(Spacer(1, 4))

    # Executive Conclusions
    story.append(Paragraph("Executive Summary & Dataset Insights", h1_style))
    summary_bullets = [
        "&bull; <b>Dataset Volume Impact:</b> Training on <b>103,746 rows (150h)</b> enabled Whisper-Small to reduce WER from 75% down to <b>40.58%</b>.",
        "&bull; <b>LoRA Parameter & Sample Efficiency:</b> LoRA achieved the lowest project WER (<b>38.91% final / 38.86% peak</b>) updating only <b>884K parameters (0.36%)</b> on the exact same 103,746 rows.",
        "&bull; <b>CTC vs Autoregressive Trade-Off:</b> Vakyansh provides ~5x faster throughput (~25 audio files/sec) but suffers from high WER (108.22%) due to character insertion errors and lack of an autoregressive language model decoder."
    ]
    for b in summary_bullets:
        story.append(Paragraph(b, meta_box_style))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    
    # Copy to project root
    shutil.copy(PDF_OUTPUT_PATH, ROOT_PDF_PATH)
    print(f"[SUCCESS] Generated 3MODEL PDF successfully at:\n  - {PDF_OUTPUT_PATH}\n  - {ROOT_PDF_PATH}")

if __name__ == "__main__":
    build_pdf()
