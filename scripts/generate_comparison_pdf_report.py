"""
generate_comparison_pdf_report.py
=================================
Generates a comprehensive, publication-quality Head-to-Head Comparative PDF Report
evaluating Fine-Tuned Whisper-Small vs. Vakyansh Wav2Vec 2.0 on Bhojpuri ASR.

Outputs:
  - report/Bhojpuri_ASR_Model_Comparison_Report.pdf
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

OUTPUT_DIR = Path("f:/bhojpuri-AI/report")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUTPUT_DIR / "Bhojpuri_ASR_Model_Comparison_Report.pdf"

# ── Color Palette ─────────────────────────────────────────────────────────────
DEEP_BLUE  = colors.HexColor('#1E40AF')
SLATE_GRAY = colors.HexColor('#64748B')
DARK_NAVY  = colors.HexColor('#0F172A')
TEXT_DARK  = colors.HexColor('#1E293B')
BORDER_CLR = colors.HexColor('#CBD5E1')
GRID_CLR   = colors.HexColor('#E2E8F0')
LIGHT_BG   = colors.HexColor('#F8FAFC')
CARD_BG    = colors.HexColor('#F1F5F9')
GREEN_WIN  = colors.HexColor('#DCFCE7')
GOLD_REC   = colors.HexColor('#FEF3C7')

# ── Numbered Canvas ───────────────────────────────────────────────────────────
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
        self.setFont("Helvetica", 8)
        self.setFillColor(SLATE_GRAY)
        self.setStrokeColor(BORDER_CLR)
        self.setLineWidth(0.5)
        self.line(45, 36, 567, 36)
        self.drawString(45, 24, "Bhojpuri ASR Research | Whisper-Small vs. Vakyansh Comparative Benchmark")
        self.drawRightString(567, 24, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()

def create_comparison_pdf():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=DEEP_BLUE,
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=13.5,
        textColor=SLATE_GRAY,
        spaceAfter=10
    )
    h1_style = ParagraphStyle(
        'H1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15.5,
        textColor=DEEP_BLUE,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11.5,
        textColor=TEXT_DARK,
        spaceAfter=5
    )
    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11.5,
        textColor=TEXT_DARK,
        leftIndent=10,
        spaceAfter=2.5
    )
    th_style = ParagraphStyle(
        'TH_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1
    )
    td_style = ParagraphStyle(
        'TD_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7,
        leading=9,
        textColor=TEXT_DARK
    )
    td_bold = ParagraphStyle(
        'TDBold_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7,
        leading=9,
        textColor=TEXT_DARK
    )
    td_center = ParagraphStyle(
        'TDCenter_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7,
        leading=9,
        textColor=TEXT_DARK,
        alignment=1
    )

    story = []

    # ── Header Banner ─────────────────────────────────────────────────────────
    story.append(Paragraph("BHOJPURI ASR: HEAD-TO-HEAD MODEL COMPARISON REPORT", title_style))
    story.append(Paragraph("Comparative Evaluation of Fine-Tuned Whisper-Small vs. Vakyansh Wav2Vec 2.0 across Datasets, Training Methods, Checkpoints & Accuracy", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=DEEP_BLUE, spaceBefore=0, spaceAfter=8))

    # ── 1. Executive Summary & Comparison Snapshot ────────────────────────────
    story.append(Paragraph("1. Executive Summary & Benchmark Scorecard", h1_style))
    summary_text = (
        "This study presents an empirical head-to-head evaluation between two major state-of-the-art architectures for "
        "Bhojpuri speech recognition: a fine-tuned <b>OpenAI Whisper-Small</b> (Seq2Seq Transformer) and the open-source "
        "<b>Vakyansh Bhojpuri</b> (Wav2Vec 2.0 CTC acoustic model). Both models were evaluated on the exact same benchmark "
        "split of <b>1,054 Bhojpuri audio samples</b> (<code>data/merged_bhojpuri/eval</code>). "
        "<b>Whisper-Small achieved superior linguistic accuracy (40.58% WER)</b>, while <b>Vakyansh demonstrated ~5x faster inference throughput</b>."
    )
    story.append(Paragraph(summary_text, body_style))

    # Scorecard Table
    scorecard_data = [
        [Paragraph("<b>Evaluation Metric</b>", th_style), Paragraph("<b>Fine-Tuned Whisper-Small</b>", th_style), Paragraph("<b>Vakyansh Wav2Vec 2.0</b>", th_style), Paragraph("<b>Winner / Core Trade-Off</b>", th_style)],
        [
            Paragraph("<b>Best Overall WER (Accuracy)</b>", td_bold),
            Paragraph("<b>40.58% WER</b> (at Checkpoint-14900)", td_bold),
            Paragraph("108.22% WER", td_style),
            Paragraph("🏆 <b>Whisper-Small</b> (+67.6% lower error)", td_bold)
        ],
        [
            Paragraph("<b>Studio Speech WER (IISc SYSPIN)</b>", td_bold),
            Paragraph("<b>~36.8% WER</b> (High accuracy)", td_style),
            Paragraph("102.96% WER", td_style),
            Paragraph("🏆 <b>Whisper-Small</b>", td_bold)
        ],
        [
            Paragraph("<b>Field / Mobile Speech WER (AI4Bharat)</b>", td_bold),
            Paragraph("<b>~44.2% WER</b> (Noise tolerant)", td_style),
            Paragraph("114.88% WER", td_style),
            Paragraph("🏆 <b>Whisper-Small</b> (Context robustness)", td_bold)
        ],
        [
            Paragraph("<b>Inference Speed (Throughput)</b>", td_bold),
            Paragraph("~4–6 audio files / second", td_style),
            Paragraph("⚡ <b>~25 audio files / second</b>", td_bold),
            Paragraph("⚡ <b>Vakyansh</b> (~5x faster CTC)", td_bold)
        ],
        [
            Paragraph("<b>GPU VRAM Footprint</b>", td_bold),
            Paragraph("~2.8 GB VRAM (FP16)", td_style),
            Paragraph("~1.2 GB VRAM (FP16)", td_style),
            Paragraph("⚡ <b>Vakyansh</b> (Lighter resource demand)", td_bold)
        ],
        [
            Paragraph("<b>Model Parameters & Architecture</b>", td_bold),
            Paragraph("240 Million Params (Seq2Seq Transformer)", td_style),
            Paragraph("95 Million Params (Wav2Vec2 + CTC)", td_style),
            Paragraph("Whisper: Autoregressive LM; Vakyansh: CTC", td_style)
        ],
    ]
    t_score = Table(scorecard_data, colWidths=[125, 130, 125, 142])
    t_score.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DEEP_BLUE),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_CLR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, GRID_CLR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ('BACKGROUND', (0, 1), (-1, 1), GREEN_WIN),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_score)
    story.append(Spacer(1, 8))

    # ── 2. Datasets Used in Both Models ───────────────────────────────────────
    story.append(Paragraph("2. Dataset Analysis: Training & Evaluation Corpora", h1_style))
    story.append(Paragraph(
        "A critical reason for performance differences lies in the volume, acoustic environment, and diversity of the datasets used for training and evaluation:",
        body_style
    ))

    dataset_table = [
        [Paragraph("<b>Category</b>", th_style), Paragraph("<b>Whisper-Small Training Data</b>", th_style), Paragraph("<b>Vakyansh Training Data</b>", th_style), Paragraph("<b>Shared Benchmark Eval Set</b>", th_style)],
        [
            Paragraph("<b>Dataset Source</b>", td_bold),
            Paragraph("<code>ai4bharat/Rural_Women_Bhojpuri</code>", td_style),
            Paragraph("<code>ULCA / IISc RESPINS & SYSPIN</code>", td_style),
            Paragraph("<code>data/merged_bhojpuri/eval</code>", td_style)
        ],
        [
            Paragraph("<b>Total Volume / Hours</b>", td_bold),
            Paragraph("~100,000+ Utterances (~150+ Hours)", td_style),
            Paragraph("~60 Hours (Male & Merged)", td_style),
            Paragraph("1,054 Audio Files (~4.8 GB Arrow cache)", td_style)
        ],
        [
            Paragraph("<b>Speaker Demographics</b>", td_bold),
            Paragraph("Native rural women across rural Bihar & UP districts", td_style),
            Paragraph("Studio voice artists + field contributors (18+ artists)", td_style),
            Paragraph("Diverse mix: 610 Studio + 444 Crowdsourced mobile", td_style)
        ],
        [
            Paragraph("<b>Acoustic Conditions</b>", td_bold),
            Paragraph("Village environment, natural ambient noise, phone audio", td_style),
            Paragraph("Clean studio recordings + controlled mobile samples", td_style),
            Paragraph("6 Topic Domains (Politics, Health, Food, Finance, etc.)", td_style)
        ],
        [
            Paragraph("<b>Target Script</b>", td_bold),
            Paragraph("Devanagari script with standard conversational tokens", td_style),
            Paragraph("Pure Devanagari character-level set (65 tokens)", td_style),
            Paragraph("Normalized Devanagari reference ground truth", td_style)
        ],
    ]
    t_data = Table(dataset_table, colWidths=[105, 140, 137, 140])
    t_data.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DEEP_BLUE),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_CLR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, GRID_CLR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_data)
    story.append(Spacer(1, 8))

    # ── 3. Way of Fine-Tuning & Architecture Comparison ───────────────────────
    story.append(Paragraph("3. Fine-Tuning Methodology & Architectural Comparison", h1_style))
    story.append(Paragraph(
        "The two models adopt fundamentally different paradigms for Automatic Speech Recognition:",
        body_style
    ))

    method_table = [
        [Paragraph("<b>Dimension</b>", th_style), Paragraph("<b>Fine-Tuned Whisper-Small Approach</b>", th_style), Paragraph("<b>Vakyansh Wav2Vec 2.0 Approach</b>", th_style)],
        [
            Paragraph("<b>Underlying Architecture</b>", td_bold),
            Paragraph("<b>Encoder-Decoder Transformer</b> (Seq2Seq). Audio spectrograms are mapped into an encoder; an autoregressive decoder generates text token by token.", td_style),
            Paragraph("<b>CTC Acoustic Model</b> (Encoder-only). Raw 16 kHz waveform is processed via 1D CNNs + Transformer layers; a linear head outputs character logits.", td_style)
        ],
        [
            Paragraph("<b>Loss Function & Training</b>", td_bold),
            Paragraph("Cross-Entropy Loss with Teacher Forcing via Hugging Face <code>Seq2SeqTrainer</code>. AdamW optimizer, warmup, cosine decay.", td_style),
            Paragraph("Connectionist Temporal Classification (CTC) Loss via Fairseq. Learns acoustic alignments without explicit temporal alignment labels.", td_style)
        ],
        [
            Paragraph("<b>Language Modeling</b>", td_bold),
            Paragraph("<b>Built-in Decoder LM</b>: Inherently understands Bhojpuri grammar, sentence structure, and context, correcting homophones automatically.", td_style),
            Paragraph("<b>Acoustic Only (No LM)</b>: Emits purely phonetic character sequences; requires external KenLM n-gram model to resolve grammar.", td_style)
        ],
        [
            Paragraph("<b>Input Representation</b>", td_bold),
            Paragraph("80-channel log-magnitude Mel-spectrogram calculated over 25ms windows.", td_style),
            Paragraph("Raw 1D time-domain audio waveform sampled at 16,000 Hz.", td_style)
        ],
        [
            Paragraph("<b>Inference Nature</b>", td_bold),
            Paragraph("Autoregressive (beam search / greedy). Slower per token but context-aware.", td_style),
            Paragraph("Non-autoregressive (single forward pass argmax). Extremely fast parallel decoding.", td_style)
        ],
    ]
    t_method = Table(method_table, colWidths=[110, 206, 206])
    t_method.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DEEP_BLUE),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_CLR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, GRID_CLR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_method)
    story.append(Spacer(1, 8))

    # ── 4. Checkpoints & Training Progression ──────────────────────────────────
    story.append(Paragraph("4. Checkpoints, Training Progression & History", h1_style))
    story.append(Paragraph(
        "A breakdown of the checkpoint evolution recorded during model training and benchmark evaluations:",
        body_style
    ))

    ckpt_table = [
        [Paragraph("<b>Model & Checkpoint</b>", th_style), Paragraph("<b>Step / Epoch</b>", th_style), Paragraph("<b>Eval Loss</b>", th_style), Paragraph("<b>WER (%)</b>", th_style), Paragraph("<b>Status & Notes</b>", th_style)],
        [
            Paragraph("Whisper-Small (Initial)", td_style),
            Paragraph("Step 100 (0.01 Ep)", td_style),
            Paragraph("0.6940", td_style),
            Paragraph("71.50%", td_style),
            Paragraph("Baseline adaptation starting point", td_style)
        ],
        [
            Paragraph("Whisper-Small (Early)", td_style),
            Paragraph("Step 400 (0.04 Ep)", td_style),
            Paragraph("0.4352", td_style),
            Paragraph("58.88%", td_style),
            Paragraph("Rapid initial phonetic convergence", td_style)
        ],
        [
            Paragraph("Whisper-Small (Epoch 1.0)", td_style),
            Paragraph("Step 10,800 (1.03 Ep)", td_style),
            Paragraph("0.2803", td_style),
            Paragraph("42.34%", td_style),
            Paragraph("Passed full dataset once (Saved on disk)", td_style)
        ],
        [
            Paragraph("<b>🏆 Whisper-Small (Best)</b>", td_bold),
            Paragraph("<b>Step 14,900 (1.42 Ep)</b>", td_bold),
            Paragraph("<b>0.2655</b>", td_bold),
            Paragraph("<b>40.58%</b>", td_bold),
            Paragraph("<b>All-Time Best Checkpoint (models/bhojpuri-whisper-small-full)</b>", td_bold)
        ],
        [
            Paragraph("Whisper-Small (Current)", td_style),
            Paragraph("Step 15,500 (1.48 Ep)", td_style),
            Paragraph("0.2642", td_style),
            Paragraph("43.80%", td_style),
            Paragraph("Active training checkpoint (~49.3% of 3-epoch goal)", td_style)
        ],
        [
            Paragraph("<b>Vakyansh Bhojpuri</b>", td_bold),
            Paragraph("bhom_60 (60h)", td_style),
            Paragraph("N/A (CTC)", td_style),
            Paragraph("<b>108.22%</b>", td_bold),
            Paragraph("Static open-source release checkpoint (102.96% Studio / 114.88% Field)", td_style)
        ],
    ]
    t_ckpt = Table(ckpt_table, colWidths=[120, 95, 65, 65, 177])
    t_ckpt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DEEP_BLUE),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_CLR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, GRID_CLR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ('BACKGROUND', (0, 4), (-1, 4), GOLD_REC),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_ckpt)
    story.append(Spacer(1, 8))

    # ── 5. Practical Insights & Recommendations ────────────────────────────────
    story.append(Paragraph("5. Practical Insights & Deployment Recommendations", h1_style))
    insights = [
        "<b>Why Whisper-Small Excels in Accuracy (40.58% WER):</b> The multilingual pre-training + integrated Devanagari language model enables Whisper to understand Bhojpuri grammar and sentence context, effectively bridging dialectal noise and phonetic ambiguities.",
        "<b>Why Vakyansh Struggles with Word Error Rate (108.22% WER):</b> Without a language model, CTC predicts purely character-by-character. Minor character omissions split single words into two or merge words (e.g. <i>जानलजाला</i>), which heavily penalizes string-level WER. Furthermore, Vakyansh spells out numbers in full words (<i>1781</i> → <i>सतरह सौ इक्यासी</i>).",
        "<b>When to Use Vakyansh:</b> Real-time streaming on low-power devices, mobile voice command triggers, and high-throughput audio pre-screening where speed (>25 audios/sec) and low memory (1.2 GB) matter most.",
        "<b>When to Use Fine-Tuned Whisper:</b> Full-length voice transcription, document dictation, conversational AI bots, and subtitles where word accuracy is paramount.",
        "<b>Recommended Path Forward:</b> Resume Whisper-Small fine-tuning from Checkpoint-15500 to reach 3.0 Full Epochs (driving WER toward ~30-35%), while exploring a hybrid pipeline that pairs Vakyansh for instant wake-word detection with Whisper for full utterance transcription."
    ]
    for ins in insights:
        story.append(Paragraph(f"• {ins}", bullet_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✅ Generated Comparative PDF: {PDF_PATH}")

if __name__ == "__main__":
    create_comparison_pdf()
