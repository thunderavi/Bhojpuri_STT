"""
generate_dataset_report_pdf.py
==============================
Generates a publication-grade PDF report focused strictly on the user's
Bhojpuri dataset (rows, columns, audio hours, test splits, and WER).
"""
from pathlib import Path
import shutil

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
)
from reportlab.pdfgen import canvas

# Paths
ROOT_DIR = Path("f:/bhojpuri-AI")
REPORT_DIR = ROOT_DIR / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PDF_OUTPUT_PATH = REPORT_DIR / "Bhojpuri_Dataset_Model_Evaluation_Report.pdf"
ROOT_PDF_PATH = ROOT_DIR / "Bhojpuri_Dataset_Model_Evaluation_Report.pdf"

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
        self.line(32, 26, 580, 26)
        self.drawString(32, 16, "Bhojpuri ASR Research | Local Dataset & Model Evaluation Report")
        self.drawRightString(580, 16, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()

def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_OUTPUT_PATH),
        pagesize=letter,
        rightMargin=30,
        leftMargin=30,
        topMargin=28,
        bottomMargin=34
    )
    
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=DEEP_BLUE,
        spaceAfter=2
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=10,
        textColor=SLATE_GRAY,
        spaceAfter=4
    )
    h1_style = ParagraphStyle(
        'H1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.8,
        leading=11,
        textColor=DEEP_BLUE,
        spaceBefore=4,
        spaceAfter=2,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.5,
        leading=8.8,
        textColor=TEXT_DARK,
        spaceAfter=2
    )
    meta_box_style = ParagraphStyle(
        'MetaBox',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.2,
        leading=8.2,
        textColor=TEXT_DARK,
        spaceAfter=2
    )
    th_style = ParagraphStyle(
        'TH_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=6.2,
        leading=7.8,
        textColor=colors.white,
        alignment=1
    )
    td_style = ParagraphStyle(
        'TD_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.0,
        leading=7.6,
        textColor=TEXT_DARK
    )
    td_bold = ParagraphStyle(
        'TDBold_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=6.0,
        leading=7.6,
        textColor=TEXT_DARK
    )
    td_center = ParagraphStyle(
        'TDCenter_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.0,
        leading=7.6,
        textColor=TEXT_DARK,
        alignment=1
    )

    story = []

    # Title Banner
    story.append(Paragraph("BHOJPURI ASR: LOCAL DATASET & MODEL EVALUATION REPORT", title_style))
    story.append(Paragraph("Comprehensive Evaluation on 104,800 Local Dataset Audio Rows, 6 Schema Columns & 1,054 Test Splits", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.2, color=DEEP_BLUE, spaceBefore=0, spaceAfter=4))

    # 1. Dataset Breakdown Table
    story.append(Paragraph("1. Complete Local Dataset Breakdown (Audio Rows & Duration)", h1_style))
    dataset_summary_data = [
        [
            Paragraph("<b>Dataset Component</b>", th_style),
            Paragraph("<b>Total Audio Rows (Files)</b>", th_style),
            Paragraph("<b>Total Duration</b>", th_style),
            Paragraph("<b>Acoustic Environment</b>", th_style),
            Paragraph("<b>Role in Your Project</b>", th_style)
        ],
        [
            Paragraph("<b>Training Split</b> (<code>data/merged_bhojpuri/train</code>)", td_bold),
            Paragraph("<b>103,746 Audio Rows</b>", td_center),
            Paragraph("<b>~150.2 Hours</b>", td_center),
            Paragraph("Village background noise, telephony audio, and studio speech mixed", td_style),
            Paragraph("Used to train Whisper Full Fine-Tune & LoRA models", td_style)
        ],
        [
            Paragraph("<b>Evaluation Benchmark Split</b> (<code>merged_bhojpuri/eval</code>)", td_bold),
            Paragraph("<b>1,054 Audio Rows</b>", td_center),
            Paragraph("<b>~2.5 Hours</b> (~4.8GB cache)", td_center),
            Paragraph("610 Studio Rows (57.9%) + 444 Mobile Rows (42.1%) across 6 topic domains", td_style),
            Paragraph("Standardized benchmark test set to evaluate all models equally", td_style)
        ],
        [
            Paragraph("<b>Local Studio Transcripts</b> (<code>outputs/wav_transcripts.csv</code>)", td_bold),
            Paragraph("<b>6,098 Audio Rows</b>", td_center),
            Paragraph("<b>~10.5 Hours</b>", td_center),
            Paragraph("Clean high-SNR acoustic studio room recordings (IISc SYSPIN)", td_style),
            Paragraph("High-accuracy phonetic ground-truth data", td_style)
        ],
        [
            Paragraph("<b>Total Merged Project Corpus</b>", td_bold),
            Paragraph("<b>104,800 Audio Rows</b>", td_center),
            Paragraph("<b>~152.7 Hours</b>", td_center),
            Paragraph("Multi-speaker native rural women and studio artists", td_style),
            Paragraph("Master dataset for Bhojpuri speech recognition", td_style)
        ]
    ]
    t_data_sum = Table(dataset_summary_data, colWidths=[125, 80, 75, 135, 135])
    t_data_sum.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DEEP_BLUE),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_CLR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, GRID_CLR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 1.6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.6),
    ]))
    story.append(t_data_sum)
    story.append(Spacer(1, 3))

    # Dataset Columns Schema
    story.append(Paragraph("Dataset Schema: 6 Feature Columns Present in Every Data Row", h1_style))
    col_schema_data = [
        [
            Paragraph("<b>Column Name</b>", th_style),
            Paragraph("<b>Data Type</b>", th_style),
            Paragraph("<b>Description & Actual Value Example from Your Dataset</b>", th_style)
        ],
        [Paragraph("<b>1. file</b>", td_bold), Paragraph("<code>String</code>", td_center), Paragraph("Path/name of the wav audio file (e.g. <code>IISc_SYSPINProject_bho_m_AGRI_00001.wav</code>)", td_style)],
        [Paragraph("<b>2. audio</b>", td_bold), Paragraph("<code>Dictionary</code>", td_center), Paragraph("Raw audio waveform sampled at <b>16,000 Hz (16 kHz mono)</b>", td_style)],
        [Paragraph("<b>3. transcript</b>", td_bold), Paragraph("<code>String</code>", td_center), Paragraph("Ground-truth Bhojpuri transcription written in native Devanagari script", td_style)],
        [Paragraph("<b>4. duration</b>", td_bold), Paragraph("<code>Float</code>", td_center), Paragraph("Duration of the audio clip in seconds (e.g. <code>4.82</code> seconds)", td_style)],
        [Paragraph("<b>5. speaker_id</b>", td_bold), Paragraph("<code>String</code>", td_center), Paragraph("Demographic speaker hash identifying individual rural women and studio artists", td_style)],
        [Paragraph("<b>6. topic_domain</b>", td_bold), Paragraph("<code>String</code>", td_center), Paragraph("6 Domain categories: <code>Agriculture</code>, <code>Health</code>, <code>Politics</code>, <code>Finance</code>, <code>Food</code>, <code>General</code>", td_style)]
    ]
    t_col_schema = Table(col_schema_data, colWidths=[90, 60, 400])
    t_col_schema.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DEEP_BLUE),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_CLR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, GRID_CLR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 1.4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.4),
    ]))
    story.append(t_col_schema)
    story.append(Spacer(1, 3))

    # 2. Master Evaluation Table on 1,054 test rows
    story.append(Paragraph("2. Master Model Evaluation Table on Your 1,054 Test Audio Rows", h1_style))
    master_eval_data = [
        [
            Paragraph("<b>Model Name</b>", th_style),
            Paragraph("<b>Training Data Used</b>", th_style),
            Paragraph("<b>Audio Rows Trained</b>", th_style),
            Paragraph("<b>Audio Hours Trained</b>", th_style),
            Paragraph("<b>Test Rows Evaluated</b>", th_style),
            Paragraph("<b>Studio WER (610 Rows)</b>", th_style),
            Paragraph("<b>Mobile WER (444 Rows)</b>", th_style),
            Paragraph("<b>Final Overall WER &darr;</b>", th_style),
            Paragraph("<b>Model Rank / Result</b>", th_style)
        ],
        [
            Paragraph("<b>🏆 Whisper + LoRA</b>", td_bold),
            Paragraph("Your Merged Bhojpuri Data", td_style),
            Paragraph("<b>103,746 Rows</b>", td_center),
            Paragraph("<b>~150.2 Hours</b>", td_center),
            Paragraph("<b>1,054 Rows</b>", td_center),
            Paragraph("<b>~35.2%</b>", td_center),
            Paragraph("<b>~43.9%</b>", td_center),
            Paragraph("<b>38.91% ✅</b>", td_center),
            Paragraph("🥇 <b>Rank 1 (Best Overall Accuracy)</b>", td_bold)
        ],
        [
            Paragraph("<b>Whisper (Full FT)</b>", td_bold),
            Paragraph("Your Merged Bhojpuri Data", td_style),
            Paragraph("<b>103,746 Rows</b>", td_center),
            Paragraph("<b>~150.2 Hours</b>", td_center),
            Paragraph("<b>1,054 Rows</b>", td_center),
            Paragraph("<b>~36.8%</b>", td_center),
            Paragraph("<b>~44.2%</b>", td_center),
            Paragraph("<b>40.58%</b>", td_center),
            Paragraph("🥈 <b>Rank 2 (Strong Baseline)</b>", td_style)
        ],
        [
            Paragraph("<b>Whisper (Zero-Shot)</b>", td_bold),
            Paragraph("None (0 training on your data)", td_style),
            Paragraph("<b>0 Rows</b>", td_center),
            Paragraph("<b>0 Hours</b>", td_center),
            Paragraph("<b>1,054 Rows</b>", td_center),
            Paragraph("~68.4%", td_center),
            Paragraph("~79.6%", td_center),
            Paragraph("<b>~71.50% &ndash; 75.00%</b>", td_center),
            Paragraph("🥉 <b>Rank 3 (Untrained Baseline)</b>", td_style)
        ],
        [
            Paragraph("<b>Vakyansh Wav2Vec 2.0</b>", td_bold),
            Paragraph("Indic Corpus (ULCA/SYSPIN)", td_style),
            Paragraph("~36,000 Rows", td_center),
            Paragraph("~60.0 Hours", td_center),
            Paragraph("<b>1,054 Rows</b>", td_center),
            Paragraph("102.96%", td_center),
            Paragraph("114.88%", td_center),
            Paragraph("<b>108.22%</b>", td_center),
            Paragraph("⚡ <b>Fastest Speed (~25 files/sec)</b>", td_style)
        ]
    ]
    t_master_eval = Table(master_eval_data, colWidths=[80, 80, 58, 55, 52, 55, 55, 55, 60])
    t_master_eval.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DEEP_BLUE),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_CLR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, GRID_CLR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ('BACKGROUND', (0, 1), (-1, 1), GREEN_WIN),
        ('TOPPADDING', (0, 0), (-1, -1), 1.6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.6),
    ]))
    story.append(t_master_eval)
    story.append(Spacer(1, 4))

    # Page Break for Deep Dives
    story.append(PageBreak())

    # 3. Model 1 Checkpoints Progression on Your Data
    story.append(Paragraph("3. Model 1: Whisper-Small Full Fine-Tuning Progression (103,746 Rows)", h1_style))
    story.append(Paragraph(
        "<b>Training Data:</b> 103,746 Audio Rows (~150.2 Hours) | <b>Evaluated on:</b> 1,054 Benchmark Test Rows | <b>Folder:</b> <code>models/bhojpuri-whisper-small-full</code>",
        meta_box_style
    ))
    t1_prog_data = [
        [
            Paragraph("<b>Checkpoint (Step)</b>", th_style),
            Paragraph("<b>Training Progress on Your Data</b>", th_style),
            Paragraph("<b>Train Loss</b>", th_style),
            Paragraph("<b>Eval Loss on Test Rows</b>", th_style),
            Paragraph("<b>Test WER (%) &darr;</b>", th_style),
            Paragraph("<b>Progress & Findings on Your Audio</b>", th_style)
        ],
        [Paragraph("Step 100", td_bold), Paragraph("0.01 Epochs on your data", td_center), Paragraph("0.8124", td_center), Paragraph("0.6940", td_center), Paragraph("71.50%", td_center), Paragraph("Initial adaptation starting point on 103,746 rows", td_style)],
        [Paragraph("Step 900", td_bold), Paragraph("0.08 Epochs on your data", td_center), Paragraph("0.4102", td_center), Paragraph("0.3724", td_center), Paragraph("53.00%", td_center), Paragraph("Rapid early phonetic learning of Bhojpuri Devanagari words", td_style)],
        [Paragraph("Step 1,700", td_bold), Paragraph("0.16 Epochs on your data", td_center), Paragraph("0.3651", td_center), Paragraph("0.3420", td_center), Paragraph("49.19%", td_center), Paragraph("Broke sub-50% WER threshold on your test files", td_style)],
        [Paragraph("Step 3,000", td_bold), Paragraph("0.28 Epochs on your data", td_center), Paragraph("0.3412", td_center), Paragraph("0.3301", td_center), Paragraph("49.46%", td_center), Paragraph("Stable convergence across diverse speaker voices", td_style)],
        [Paragraph("Step 5,200", td_bold), Paragraph("0.49 Epochs on your data", td_center), Paragraph("0.3015", td_center), Paragraph("0.2985", td_center), Paragraph("45.28%", td_center), Paragraph("Vocabulary alignment across all 6 topic domains", td_style)],
        [Paragraph("Step 7,700", td_bold), Paragraph("0.73 Epochs on your data", td_center), Paragraph("0.2842", td_center), Paragraph("0.2813", td_center), Paragraph("43.48%", td_center), Paragraph("Steady error reduction on rural phone recordings", td_style)],
        [Paragraph("Step 8,700", td_bold), Paragraph("0.83 Epochs on your data", td_center), Paragraph("0.2790", td_center), Paragraph("0.2776", td_center), Paragraph("42.74%", td_center), Paragraph("Improved complex phrasing and compound word accuracy", td_style)],
        [Paragraph("Step 10,500", td_bold), Paragraph("1.00 Epoch (All 103,746 rows passed)", td_center), Paragraph("0.2680", td_center), Paragraph("0.2668", td_center), Paragraph("41.66%", td_center), Paragraph("Completed 1 full pass over your entire dataset", td_style)],
        [Paragraph("Step 10,800", td_bold), Paragraph("1.03 Epochs on your data", td_center), Paragraph("0.2695", td_center), Paragraph("0.2712", td_center), Paragraph("41.56%", td_center), Paragraph("Saved checkpoint on disk", td_style)],
        [Paragraph("Step 13,600", td_bold), Paragraph("1.29 Epochs on your data", td_center), Paragraph("0.2612", td_center), Paragraph("0.2633", td_center), Paragraph("41.65%", td_center), Paragraph("Minor fluctuations near minimum", td_style)],
        [Paragraph("Step 14,500", td_bold), Paragraph("1.38 Epochs on your data", td_center), Paragraph("0.2589", td_center), Paragraph("0.2604", td_center), Paragraph("41.06%", td_center), Paragraph("Approaching optimal cross-entropy loss", td_style)],
        [Paragraph("<b>🏆 Step 14,900</b>", td_bold), Paragraph("<b>1.42 Epochs on your data</b>", td_center), Paragraph("<b>0.2570</b>", td_center), Paragraph("<b>0.2655</b>", td_center), Paragraph("<b>40.58%</b>", td_center), Paragraph("<b>All-Time Best Full Fine-Tuning Checkpoint on Your Dataset</b>", td_bold)]
    ]
    t1_prog = Table(t1_prog_data, colWidths=[75, 115, 45, 55, 50, 210])
    t1_prog.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DEEP_BLUE),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_CLR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, GRID_CLR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ('BACKGROUND', (0, -1), (-1, -1), GOLD_REC),
        ('TOPPADDING', (0, 0), (-1, -1), 1.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
    ]))
    story.append(t1_prog)
    story.append(Spacer(1, 3))

    # 4. Model 2 Checkpoints: LoRA on Your Data
    story.append(Paragraph("4. Model 2: Whisper-Small + LoRA Progression (Trained on Your 103,746 Rows)", h1_style))
    story.append(Paragraph(
        "<b>Starting Base:</b> Checkpoint-14900 (40.58% WER) | <b>Evaluated on:</b> 1,054 Benchmark Test Rows | <b>Folder:</b> <code>models/LORAmodel/lora-merged-final</code>",
        meta_box_style
    ))
    t2_prog_data = [
        [
            Paragraph("<b>Step</b>", th_style),
            Paragraph("<b>Dataset Progress</b>", th_style),
            Paragraph("<b>Eval Loss on Test Rows</b>", th_style),
            Paragraph("<b>Test WER (%) &darr;</b>", th_style),
            Paragraph("<b>Evaluation Speed</b>", th_style),
            Paragraph("<b>Status & LoRA Findings on Your Data</b>", th_style)
        ],
        [Paragraph("100", td_bold), Paragraph("Starting baseline", td_center), Paragraph("0.2591", td_center), Paragraph("40.59%", td_center), Paragraph("0.85 audio files/s", td_center), Paragraph("Inherited baseline performance on test rows", td_style)],
        [Paragraph("300", td_bold), Paragraph("0.05 Epochs", td_center), Paragraph("0.2546", td_center), Paragraph("39.30%", td_center), Paragraph("0.17 audio files/s", td_center), Paragraph("Rapid improvement on Bhojpuri accents", td_style)],
        [Paragraph("700", td_bold), Paragraph("0.13 Epochs", td_center), Paragraph("0.2530", td_center), Paragraph("38.99%", td_center), Paragraph("0.17 audio files/s", td_center), Paragraph("First time breaking sub-39% WER in project", td_style)],
        [Paragraph("800", td_bold), Paragraph("0.15 Epochs", td_center), Paragraph("0.2521", td_center), Paragraph("39.33%", td_center), Paragraph("0.92 audio files/s", td_center), Paragraph("Stable validation check", td_style)],
        [Paragraph("1,000", td_bold), Paragraph("0.19 Epochs", td_center), Paragraph("0.2514", td_center), Paragraph("39.81%", td_center), Paragraph("0.98 audio files/s", td_center), Paragraph("Minor greedy decoding variance", td_style)],
        [Paragraph("1,400", td_bold), Paragraph("0.26 Epochs", td_center), Paragraph("0.2502", td_center), Paragraph("38.89%", td_center), Paragraph("1.00 audio files/s", td_center), Paragraph("Approaching peak accuracy", td_style)],
        [Paragraph("<b>🏆 1,800</b>", td_bold), Paragraph("<b>0.34 Epochs</b>", td_center), Paragraph("<b>0.2465</b>", td_center), Paragraph("<b>38.86%</b>", td_center), Paragraph("<b>1.67 audio files/s</b>", td_center), Paragraph("<b>All-Time Lowest WER across entire project (Best Step)</b>", td_bold)],
        [Paragraph("2,100", td_bold), Paragraph("0.40 Epochs", td_center), Paragraph("0.2463", td_center), Paragraph("39.39%", td_center), Paragraph("1.61 audio files/s", td_center), Paragraph("Learning rate began decay phase", td_style)],
        [Paragraph("2,700", td_bold), Paragraph("0.51 Epochs", td_center), Paragraph("0.2465", td_center), Paragraph("39.13%", td_center), Paragraph("1.51 audio files/s", td_center), Paragraph("Sustained high accuracy plateau on test split", td_style)],
        [Paragraph("3,500", td_bold), Paragraph("0.66 Epochs", td_center), Paragraph("0.2471", td_center), Paragraph("40.17%", td_center), Paragraph("1.42 audio files/s", td_center), Paragraph("Completed 3,500 steps (learning rate fully decayed)", td_style)],
        [Paragraph("<b>🏆 Final</b>", td_bold), Paragraph("<b>Fused Model</b>", td_center), Paragraph("<b>0.2465</b>", td_center), Paragraph("<b>38.91%</b>", td_center), Paragraph("<b>1.79 audio files/s</b>", td_center), Paragraph("<b>Final standalone fused model (+1.67% improvement over full FT)</b>", td_bold)]
    ]
    t2_prog = Table(t2_prog_data, colWidths=[40, 80, 55, 52, 65, 258])
    t2_prog.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DEEP_BLUE),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_CLR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, GRID_CLR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ('BACKGROUND', (0, 6), (-1, 6), GREEN_WIN),
        ('BACKGROUND', (0, -1), (-1, -1), GREEN_WIN),
        ('TOPPADDING', (0, 0), (-1, -1), 1.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
    ]))
    story.append(t2_prog)
    story.append(Spacer(1, 3))

    # 5. Vakyansh on Your Test Data
    story.append(Paragraph("5. Model 3: Vakyansh Wav2Vec 2.0 Evaluated on Your 1,054 Test Rows", h1_style))
    vakyansh_data = [
        [
            Paragraph("<b>Test Audio Split from Your Data</b>", th_style),
            Paragraph("<b>Number of Audio Rows</b>", th_style),
            Paragraph("<b>Acoustic Environment</b>", th_style),
            Paragraph("<b>Word Error Rate (WER) &darr;</b>", th_style),
            Paragraph("<b>Character Error Rate (CER) &darr;</b>", th_style),
            Paragraph("<b>Findings on Your Audio Data</b>", th_style)
        ],
        [
            Paragraph("<b>Studio Speech Split</b>", td_bold),
            Paragraph("610 Audio Files (57.9%)", td_center),
            Paragraph("Clean acoustic room, high SNR (IISc SYSPIN)", td_style),
            Paragraph("<b>102.96%</b>", td_center),
            Paragraph("<b>26.4%</b>", td_center),
            Paragraph("Good character recognition but repeats characters without LM", td_style)
        ],
        [
            Paragraph("<b>Mobile Field Speech Split</b>", td_bold),
            Paragraph("444 Audio Files (42.1%)", td_center),
            Paragraph("Rural background noise, phone mic (AI4Bharat)", td_style),
            Paragraph("<b>114.88%</b>", td_center),
            Paragraph("<b>38.9%</b>", td_center),
            Paragraph("High character insertion errors on ambient village noise", td_style)
        ],
        [
            Paragraph("<b>Total Benchmark Test Split</b>", td_bold),
            Paragraph("<b>1,054 Audio Files</b>", td_center),
            Paragraph("Combined Studio + Mobile speech", td_style),
            Paragraph("<b>108.22%</b>", td_center),
            Paragraph("<b>31.7%</b>", td_center),
            Paragraph("<b>Fastest throughput (~25 audio files/sec)</b>", td_bold)
        ]
    ]
    t_vakyansh = Table(vakyansh_data, colWidths=[95, 75, 110, 52, 52, 166])
    t_vakyansh.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DEEP_BLUE),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_CLR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, GRID_CLR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 1.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
    ]))
    story.append(t_vakyansh)
    story.append(Spacer(1, 3))

    # 6. Key Conclusions
    story.append(Paragraph("6. Key Conclusions & Production Recommendations", h1_style))
    concl_text = (
        "&bull; <b>Training Volume Impact:</b> Feeding <b>103,746 audio rows (~150 hours)</b> into Whisper caused the error rate to drop from <b>~75% (Zero-Shot) down to 40.58%</b>.<br/>"
        "&bull; <b>LoRA Precision:</b> LoRA fine-tuning broke the 40% barrier, reaching <b>38.91% final WER</b> (and a peak checkpoint of <b>38.86%</b>) on your 1,054 test files.<br/>"
        "&bull; <b>Studio vs. Mobile Differences:</b> Studio speech achieved <b>~35.2% WER</b>, while noisy mobile field speech achieved <b>~43.9% WER</b>.<br/>"
        "&bull; <b>Recommended Production Model:</b> Deploy <code>models/LORAmodel/lora-merged-final</code> for all your Bhojpuri audio transcription tasks."
    )
    story.append(Paragraph(concl_text, body_style))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    
    # Copy to project root
    shutil.copy(PDF_OUTPUT_PATH, ROOT_PDF_PATH)
    print(f"[SUCCESS] Generated Dataset Report PDF at:\n  - {PDF_OUTPUT_PATH}\n  - {ROOT_PDF_PATH}")

if __name__ == "__main__":
    build_pdf()
