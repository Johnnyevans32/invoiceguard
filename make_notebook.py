"""Writes InvoiceGuard_data_and_baseline.ipynb (run it afterwards with nbconvert --execute)."""
import nbformat as nbf

REPO = "https://github.com/Johnnyevans32/invoiceguard"
ZIP = REPO + "/releases/download/v0.1-data/invoiceguard-data-v0.1.zip"
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: C.append(nbf.v4.new_code_cell(s))  # noqa: E731

md(f"""# InvoiceGuard: data and text-only baseline

Workshop stage 5 (data and baseline). This notebook loads the synthetic invoice dataset, looks at a few examples, and runs the text-only baseline: Tesseract OCR, an arithmetic rule and TF-IDF + logistic regression.

Repository: {REPO}

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Johnnyevans32/invoiceguard/blob/main/InvoiceGuard_data_and_baseline.ipynb)""")
code(f"""import sys, os
if "google.colab" in sys.modules:  # get Tesseract, the code and the dataset
    !apt-get -qq install -y tesseract-ocr > /dev/null
    !pip -q install pytesseract
    if not os.path.exists("invoiceguard"):
        !git clone -q {REPO}
    %cd invoiceguard
    if not os.path.exists("data/labels.csv"):
        !wget -q {ZIP} -O data.zip && unzip -q data.zip -d data""")
code("""import json
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import baseline

labels = pd.read_csv("data/labels.csv")
pd.crosstab(labels.split, labels.label, margins=True)""")
md("## What the invoices look like\nOne invoice from each class. The hidden instruction is the faint line under the payment details, and the edited invoice has pasted numbers.")
code("""fig, axes = plt.subplots(1, 4, figsize=(20, 7))
for ax, lab in zip(axes, baseline.LABELS):
    row = labels[(labels.label == lab) & (labels.split == "test")].iloc[0]
    ax.imshow(Image.open(f"data/images/{row.file}"), cmap="gray")
    ax.set_title(f"{lab}\\n{row.file}, template {row.template}")
    ax.axis("off")
plt.tight_layout()""")
code("""row = labels[labels.label == "hidden_instruction"].iloc[0]
print("Injected text:", json.loads(row.tamper_info)["text"])
print("Shade (255 is white):", json.loads(row.tamper_info)["shade"])""")
md("## OCR and the arithmetic rule\nOCR results are cached in `data/ocr.jsonl` (Tesseract 5, `--psm 6`). The rule parses the line amounts, subtotal, VAT and total and checks that they add up.")
code("""df = baseline.features(baseline.run_ocr(__import__("pathlib").Path("data")))
ex = df[df.label == "total_mismatch"].iloc[0]
print(ex.text[:900])
print(baseline.arithmetic(ex.text))""")
code("""print("Share of invoices where the rule could parse items, subtotal and total:", round(df.parsed.mean(), 3))
df.groupby("label")[["parsed", "mismatch"]].mean().round(3)""")
md("## Baseline\nThree versions, all trained on the 8 training templates, with C chosen on the 2 validation templates and scored on the 2 test templates.")
code("""res = baseline.main("data")
pd.DataFrame({k: {"val macro F1": res[k]["val"]["macro_f1"], "test macro F1": res[k]["test"]["macro_f1"]}
              for k in ["rule_only", "tfidf_only", "baseline_tfidf_plus_rule"]}).T""")
code("""pd.DataFrame(res["baseline_tfidf_plus_rule"]["test"]["per_class"]).T""")
code("""import numpy as np
cm = np.array(res["baseline_tfidf_plus_rule"]["test"]["confusion"])
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(4), baseline.LABELS, rotation=30, ha="right")
ax.set_yticks(range(4), baseline.LABELS)
for i in range(4):
    for j in range(4):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > 30 else "black")
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Baseline, test set")
plt.tight_layout(); plt.savefig("results/confusion.png", dpi=150)""")
md("## Error analysis\nDoes the OCR text contain the hidden instruction at all? I search it for words that only the injected lines use.")
code("""import re
kw = re.compile(r"\\b(agent|assistant|ignore|instead|bot|outdated|authori[sz]ed|pre-approved|migration|payee|reading this)\\b", re.I)
h = df[df.label == "hidden_instruction"].copy()
h["shade"] = h.tamper_info.map(lambda s: json.loads(s)["shade"])
h["ocr_sees_it"] = h.text.map(lambda t: bool(kw.search(t)))
print("clean invoices with these words:", sum(bool(kw.search(t)) for t in df[df.label == "clean"].text))
h.groupby(pd.cut(h.shade, [160, 180, 200, 220, 240]), observed=False).ocr_sees_it.agg(["count", "mean"])""")
code("""print("Edited invoices caught, by kind:", res["edited_by_kind"])
print("Total mismatch:", res["mismatch_parse"])""")
md("""## What this shows

- The arithmetic rule catches every total mismatch on unseen layouts (F1 0.98).
- Tesseract drops the faint hidden instruction on every invoice, at every shade, because it treats light grey text as background. The text classifier therefore has nothing to learn from and stays near chance on that class.
- Edited amounts and bank details still add up, so the text looks like a clean invoice. Only the pixels show the edit.
- So a text-only check is not enough, which is the point of the improved model: LayoutLMv3 on the image, the words and their positions. A quick test on 10 invoices also showed that a stricter binarisation before OCR recovers some hidden lines, which I will measure properly as part of the experiment log.""")
nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "InvoiceGuard_data_and_baseline.ipynb")
print("notebook written")
