# ChroniclingAmericaQA — Phi-3.5 Zero-Shot Baseline

> **Course:** Georgia Tech CS 8803: Machine Learning for History and Humanities
> **Model:** `microsoft/phi-3.5-mini-instruct`
> **Dataset:** [ChroniclingAmericaQA](https://huggingface.co/datasets/Bhawna/ChroniclingAmericaQA)

---

## Overview

This project explores question answering over historical American newspaper text from the [Chronicling America](https://chroniclingamerica.loc.gov/) archive. Newspapers span the 1770s–1960s and present unique challenges:

- **OCR noise** from digitized historical print
- **Archaic language** and spelling variations
- **Long, dense contexts** with limited answer spans

The current implementation establishes a **zero-shot baseline** using `phi-3.5-mini-instruct` with instruction prompting. QLoRA fine-tuning is planned as the next phase.

---

## Current Progress (Sections 1–8)

| Section | Title | Description |
|---------|-------|-------------|
| 1 | Setup and Imports | Dependencies; pip install commands |
| 2 | Configuration | All hyperparameters in one place |
| 3 | Load Dataset | Load `Bhawna/ChroniclingAmericaQA` from HF Hub |
| 4 | Exploratory Data Inspection | Field overview, word-count statistics |
| 5 | Preprocessing | Whitespace normalization, truncation, OCR toggle |
| 6 | Prompt Formatting | Phi-3.5 instruction-style chat prompt template |
| 7 | Zero-Shot Baseline Inference | Batched greedy generation on dev subset |
| 8 | Baseline Evaluation | Exact Match, Token F1, ROUGE-L |

---

## Repository Structure

```
.
├── phi35_chronicling_america_qa.ipynb   # Main notebook
├── outputs/                             # Created at runtime
│   └── dev_predictions_baseline.csv    # Zero-shot predictions
└── README.md
```

---

## Dataset

**[Bhawna/ChroniclingAmericaQA](https://huggingface.co/datasets/Bhawna/ChroniclingAmericaQA)**

| Split | Examples |
|-------|----------|
| Train | 439,302 |
| Validation | 24,111 |
| Test | 24,084 |

**Key fields:**

| Field | Description |
|-------|-------------|
| `question` | Natural language question |
| `answer` | Ground-truth answer string |
| `context` | Cleaned/gold newspaper passage |
| `raw_ocr` | Raw OCR text of the same passage |
| `publication_date` | Date of the newspaper issue |

---

## Prompt Template

```
<|system|>
You are answering questions about historical American newspapers.
Use the provided context to answer the question briefly and accurately.
If the answer is not supported by the context, say "Not enough information."<|end|>
<|user|>
Context:
{context}

Question:
{question}

Answer:<|end|>
<|assistant|>
```

---

## Evaluation Metrics

- **Exact Match (EM):** Normalized string equality (lowercase, strip punctuation, remove articles)
- **Token F1:** SQuAD-style token overlap between prediction and gold answer
- **ROUGE-L:** Longest common subsequence recall via `evaluate` library

---

## Requirements

### Hardware

- Single GPU with ≥ 16 GB VRAM recommended (e.g., NVIDIA A100)
- Google Colab A100 runtime works well
- CPU-only: all cells except model loading run fine

### Software

```
torch>=2.0
transformers==4.44.2
datasets>=2.18
evaluate>=0.4
rouge_score
sentencepiece
pandas
numpy
tqdm
```

> **Important:** pin `transformers==4.44.2` — the model's bundled `modeling_phi3.py`
> uses `DynamicCache.from_legacy_cache`, which was removed in transformers ≥ 4.46.

```bash
pip install "transformers==4.44.2" torch datasets evaluate rouge_score \
            sentencepiece pandas numpy tqdm
```

---

## Quickstart

1. Upload `phi35_chronicling_america_qa.ipynb` to Google Colab
2. Set runtime to **GPU → A100**
3. Uncomment and run the pip install cell (Section 1)
4. Run all cells top-to-bottom

---

## License

For academic use only (CS 8803 course project).
