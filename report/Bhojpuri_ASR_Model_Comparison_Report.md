# Bhojpuri ASR Model Comparison Report

This report summarizes the performance evaluation of various Automatic Speech Recognition (ASR) models on the Bhojpuri language dataset. The primary metric for evaluation is **Word Error Rate (WER)**, where a lower percentage indicates better accuracy.

---

## 1. Whisper Base (Full Fine-Tuning Progression)

The standard fine-tuning process of the Whisper model on the Bhojpuri dataset yielded significant improvements. Over the course of 14,900 steps, the model successfully adapted to the language's specific phonetic and structural nuances.

### Training Progress (Key Milestones)

| Step | Epoch | Training Loss | Word Error Rate (WER) |
| :--- | :--- | :--- | :--- |
| **900** | 0.086 | 0.3724 | 53.00% |
| **3,000** | 0.286 | 0.3301 | 49.46% |
| **6,000** | 0.572 | 0.3012 | 47.04% |
| **9,000** | 0.859 | 0.2709 | 43.77% |
| **12,000** | 1.145 | 0.2692 | 43.02% |
| **14,900** | 1.422 | 0.2655 | **40.58%** |

*Note: The standard fine-tuning reached its optimal point at checkpoint 14,900 with a WER of 40.58%.*

---

## 2. Whisper + LoRA Fine-Tuning (Parameter-Efficient)

To push the accuracy even further without retraining the entire model, **Low-Rank Adaptation (LoRA)** was applied to the best checkpoint (14,900). This process fine-tuned only a small subset of the model's parameters (specifically the `q_proj` and `v_proj` attention modules).

- **Starting Baseline WER:** 40.58%
- **Total LoRA Training Steps:** 3,500
- **Final LoRA WER:** **38.91%**

The LoRA fine-tuning successfully broke the plateau, yielding an absolute improvement of **+1.67%** in just a few hours.

---

## 3. Overall Model Comparison (Master Table)

The table below provides a holistic comparison across different architectural approaches (Zero-shot vs. Custom-trained).

| Model Name | Training Type | Word Error Rate (WER) |
| :--- | :--- | :--- |
| **m1 (Whisper Base)** | Zero-Shot (No Bhojpuri training) | *[Need Your Number]* |
| **m2** | Zero-Shot | *[Need Your Number]* |
| **m3** | Zero-Shot | *[Need Your Number]* |
| **Vakyansh** | Pre-trained for Indic Languages | **108.22%** |
| **Whisper Fine-Tuned** | Full Fine-Tuning (checkpoint-14900) | **40.58%** |
| **Whisper + LoRA v1** | PEFT, r=8, q+v only, 3,500 steps | **38.91%** |
| **Whisper + LoRA v2** | PEFT, r=16, q+k+v+out, 2,000 steps | **38.91%** (tied v1) |

> [!NOTE]
> *Zero-Shot and Vakyansh model WER scores for m1, m2, m3 are pending. Whisper and LoRA numbers are confirmed from actual training runs.*

---

## 4. LoRA v2 Deep Dive

| Metric | LoRA v1 | LoRA v2 |
| :--- | :--- | :--- |
| Rank | 8 | **16** |
| Target Modules | q, v | **q, k, v, out** |
| Learning Rate | 1e-4 | **5e-5** |
| Steps | 3,500 | 2,000 |
| Best Step WER | 38.86% (step 1800) | **38.88%** (step 600) |
| Final WER | **38.91%** | **38.91%** |
| Runtime | ~5.3h | ~8.8h |
| Eval time/checkpoint | ~12 min | ~96 min |

**Verdict:** LoRA v2 tied v1 exactly at 38.91%. The larger rank and extra modules improved early convergence (best at step 600 vs step 1,800), but the cosine scheduler decayed too quickly over 2,000 steps. Eval speed was 8x slower due to the expanded model size.

---

## 4. Conclusion & Recommendations

Based on the current data:
1. **The Whisper + LoRA model** is currently the best-performing model, achieving a WER of **38.91%**. Breaking the 40% barrier is a strong milestone for a low-resource language.
2. The standard fine-tuning provided the massive heavy lifting (improving from ~53% to 40%), while the LoRA stage provided critical, precision refinements (+1.67%).
3. **Recommendation:** Deploy the merged `lora-merged-final` standalone model for transcription inferences, as it contains all base weights and LoRA weights fused together efficiently.
