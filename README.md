# Efficient Adaptation of Large Language Models for Question Answering on Historical American Newspapers using LoRA

> **Course:** Georgia Tech CS 8803
> **Model:** `microsoft/phi-3.5-mini-instruct`
> **Dataset:** [ChroniclingAmericaQA](https://huggingface.co/datasets/Bhawna/ChroniclingAmericaQA)
> **Method:** QLoRA — 4-bit quantization + LoRA adapters via PEFT

---

## Overview

This project explores parameter-efficient fine-tuning of a small language model for extractive question answering over historical American newspaper text from the [Chronicling America](https://chroniclingamerica.loc.gov/) archive. The newspaper articles span the 1770s–1960s and present unique challenges:

- **OCR noise** from digitized historical print
- **Archaic language** and spelling variations
- **Long, dense contexts** with limited answer spans

We use **QLoRA** (Quantized Low-Rank Adaptation) to fine-tune `phi-3.5-mini-instruct` on a fraction of trainable parameters, making the approach accessible on a single consumer GPU.

---

## Repository Structure

```
.
├── phi35_chronicling_america_qa.ipynb   # Main end-to-end notebook
├── outputs/                             # Created at runtime
│   ├── checkpoints/                     # Training checkpoints
│   ├── lora_adapter/                    # Saved LoRA weights
│   ├── tokenizer/                       # Saved tokenizer
│   ├── dev_predictions_baseline.csv     # Zero-shot predictions
│   ├── dev_predictions_finetuned.csv    # Fine-tuned predictions (dev)
│   ├── test_predictions_finetuned.csv   # Fine-tuned predictions (test)
│   ├── metrics_summary.csv             # All metrics in CSV
│   ├── metrics_summary.json            # All metrics in JSON
│   └── metrics_comparison.png          # Visualization chart
└── README.md
```

---

## Notebook Structure

| Section | Title | Description |
|---------|-------|-------------|
| 1 | Setup and Imports | All dependencies; pip install commands |
| 2 | Configuration | All hyperparameters in one place |
| 3 | Load Dataset | Load `Bhawna/ChroniclingAmericaQA` from HF Hub |
| 4 | Exploratory Data Inspection | Field overview, length statistics |
| 5 | Preprocessing | Whitespace normalization, truncation, OCR toggle |
| 6 | Prompt Formatting | Phi-3.5 instruction-style chat prompt template |
| 7 | Zero-Shot Baseline Inference | Batched generation without fine-tuning |
| 8 | Baseline Evaluation | Exact Match, Token F1, ROUGE-L |
| 9 | Prepare Training Data for QLoRA | SFT format with `text` field |
| 10 | Load Quantized Model and Tokenizer | 4-bit NF4 via BitsAndBytesConfig |
| 11 | Attach LoRA Adapters | PEFT LoraConfig, trainable parameter count |
| 12 | Fine-Tuning Setup | `SFTConfig` with completion-only loss |
| 13 | Train Model | `SFTTrainer.train()` |
| 14 | Validation / Dev Evaluation | Post-training generation + metrics |
| 15 | Test Set Inference | Final held-out predictions |
| 16 | Final Evaluation Metrics | Comparison table + bar chart visualization |
| 17 | Error Analysis | Correct examples, failures, hallucinations, OCR noise |
| 18 | Save Model, Predictions, and Metrics | Persist all outputs |
| 19 | Optional Experiments | LoRA rank sweep, OCR vs. clean, year split, RAG |
| 20 | Conclusions / Notes | Observations and next steps |

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
| `url` | Source URL on Chronicling America |

The dataset contains both **cleaned context** and **raw OCR** text for each example, enabling controlled experiments on the effect of OCR noise.

---

## Method

### QLoRA Setup

| Component | Choice |
|-----------|--------|
| Base model | `microsoft/phi-3.5-mini-instruct` (3.8B params) |
| Quantization | 4-bit NF4 via `bitsandbytes` |
| LoRA rank `r` | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Trainable params | ~1% of total |

### Prompt Template

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
{answer}<|end|>
```

### Training

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | `paged_adamw_8bit` |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Batch size (effective) | 16 (2 × 8 grad accum) |
| Max sequence length | 1024 tokens |
| Loss | Completion-only (answer tokens only) |

---

## Evaluation Metrics

- **Exact Match (EM):** Normalized string equality (lowercase, strip punctuation, remove articles)
- **Token F1:** SQuAD-style token overlap between prediction and gold answer
- **ROUGE-L:** Longest common subsequence recall, using `evaluate` library

---

## Requirements

### Hardware

- **Minimum:** Single GPU with ≥ 16 GB VRAM (e.g., NVIDIA A100, RTX 3090/4090)
- **Recommended:** Google Colab A100 runtime (free tier T4 may work with smaller subsets)
- CPU-only: imports, preprocessing, and evaluation cells run fine; model loading and training require CUDA

### Software

```
torch>=2.0
transformers>=4.40
peft>=0.10
trl>=0.9
bitsandbytes>=0.43
accelerate>=0.27
datasets>=2.18
evaluate>=0.4
rouge_score
sentencepiece
pandas
numpy
tqdm
matplotlib
```

Install all at once:

```bash
pip install torch transformers peft trl bitsandbytes accelerate \
            datasets evaluate rouge_score sentencepiece \
            pandas numpy tqdm matplotlib
```

---

## Quickstart

### Option A — Google Colab (recommended)

1. Upload `phi35_chronicling_america_qa.ipynb` to Colab
2. Set runtime to **GPU → A100** (or T4 for smaller subsets)
3. Run **Section 1** (pip installs, uncomment the install cell)
4. Run all cells top-to-bottom

### Option B — Local

```bash
# Clone the repo
git clone git@github.com:ericsheng495/chronicling-america-qa-lora.git
cd chronicling-america-qa-lora

# Install dependencies
pip install -r requirements.txt   # or use the pip command above

# Launch Jupyter
jupyter notebook phi35_chronicling_america_qa.ipynb
```

---

## Configuration

All experiment knobs live in **Section 2** of the notebook:

```python
MAX_TRAIN_SAMPLES = 500    # None = full dataset (439k examples)
MAX_DEV_SAMPLES   = 100
USE_RAW_OCR       = False  # True = use raw OCR instead of cleaned context
LORA_R            = 16     # LoRA rank (sweep: 4, 8, 16, 32, 64)
LEARNING_RATE     = 2e-4
NUM_EPOCHS        = 3
```

Start small (`MAX_TRAIN_SAMPLES = 200`) to verify the pipeline end-to-end before scaling up.

---

## Optional Experiments (Section 19)

The notebook includes template cells for:

| Experiment | Description |
|------------|-------------|
| **LoRA Rank Sweep** | Compare EM/F1 across r = 4, 8, 16, 32, 64 |
| **OCR vs. Clean Context** | Toggle `USE_RAW_OCR` and retrain/re-evaluate |
| **Year-Based Partitioning** | Train pre-1910, test post-1910 for temporal generalization |
| **Retrieval-Augmented Baseline** | Replace gold context with BM25-retrieved passages |

---

## Known Issues / TODOs

- [ ] Verify `TARGET_MODULES` names match phi-3.5-mini-instruct architecture (run `for name, _ in model.named_modules(): print(name)` to confirm)
- [ ] Add HF token if dataset access is rate-limited (`from huggingface_hub import login`)
- [ ] `paged_adamw_8bit` requires CUDA; change to `adamw_torch` for CPU-only debugging
- [ ] Fill in Section 20 (Conclusions) after running experiments

---

## License

For academic use only (CS 8803 course project).
