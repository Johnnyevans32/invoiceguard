# InvoiceGuard

Catching tampered invoices before an AI agent pays them.

AI agents are starting to read invoices and pay bills on their own. InvoiceGuard looks at an invoice image first and says whether it is safe to pay. It gives one of four answers:

| Label | Meaning |
|---|---|
| `clean` | nothing wrong |
| `hidden_instruction` | faint or tiny text tells the AI agent to pay someone else |
| `total_mismatch` | the subtotal or total does not match the line items |
| `edited_amount_bank` | an amount or the bank account was pasted over, and the numbers still add up |

**Research question.** Does a model that looks at the invoice image and its text together catch tampered invoices better than a check that only reads the text? And how much does it lower the number of times a VLM payment agent pays the wrong amount or account?

Workshop project for Artificial Intelligence and Neural Networks, UrFU, autumn 2026.

## What is here

| File | What it does |
|---|---|
| `invoicegen.py` | Makes synthetic invoices from 12 templates and applies the three kinds of tampering and scan effects |
| `baseline.py` | Text-only baseline: Tesseract OCR, an arithmetic rule, TF-IDF + logistic regression |
| `InvoiceGuard_data_and_baseline.ipynb` | Walks through the data and runs the baseline (opens in Colab) |
| `DATASET_CARD.md` | How the dataset was made, the split, limits and ethics |
| `results/` | Baseline results on the test set |

## Data

Dataset v0.1: 1,200 invoices, 300 per class, split by template (800 train, 200 val, 200 test). Download it from the [v0.1-data release](https://github.com/Johnnyevans32/invoiceguard/releases/tag/v0.1-data) or rebuild it:

```bash
pip install -r requirements.txt   # and install Tesseract (brew install tesseract / apt-get install tesseract-ocr)
python invoicegen.py --out data --n 1200 --seed 7
python baseline.py --data data
```

## Baseline results (test set, layouts never seen in training)

| Model | Macro F1 |
|---|---|
| Arithmetic rule only | 0.354 |
| TF-IDF + logistic regression only | 0.185 |
| TF-IDF + rule features (baseline) | 0.427 |

The rule catches every total mismatch (F1 0.98). Tesseract does not read the faint hidden instructions at all, and the edited amounts still add up, so both classes stay close to chance. That is the gap the image model has to close.

## Plan

| Week | Stage | Result |
|---|---|---|
| 3-4 | Data and baseline | Generator, dataset v0.1, text-only baseline (this release) |
| 5 | Model development | Real test set (SROIE and sample invoices, edited by hand), DocTamper, LayoutLMv3 fine-tuned |
| 6 | Model development | Agent test with Qwen2.5-VL, 3 seed runs, error analysis, experiment log |
| 7 | Paper | Draft, peer review, fixes |
| 8 | Presentation | Poster and demo |

## Ethics

All invoices are fake. The hidden instruction texts describe a known attack in plain words. The project builds a defence.
