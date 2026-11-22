"""InvoiceGuard text-only baseline: Tesseract OCR, an arithmetic rule, and TF-IDF + logistic regression.

  python baseline.py --data data            # OCR (cached in data/ocr.jsonl), train, evaluate
Outputs results.json, per-class tables and a confusion matrix in data/results/.
"""
import argparse
import json
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

LABELS = ["clean", "hidden_instruction", "total_mismatch", "edited_amount_bank"]
MONEY = re.compile(r"(?<![\d.])\d{1,3}(?:[,.\s]\d{3})*[.,]\d{2}(?!\d)")
TOTAL_WORDS = ("amount due", "balance due", "amount payable", "total payable", "grand total", "total due", "total")


def ocr_one(path):
    import pytesseract
    from PIL import Image
    return pytesseract.image_to_string(Image.open(path), config="--psm 6")


def run_ocr(data, workers=6):
    cache = data / "ocr.jsonl"
    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            r = json.loads(line)
            done[r["file"]] = r["text"]
    labels = pd.read_csv(data / "labels.csv")
    todo = [f for f in labels.file if f not in done]
    if todo:
        with Pool(workers) as pool, open(cache, "a") as fh:
            for f, text in zip(todo, pool.imap(ocr_one, [str(data / "images" / f) for f in todo], chunksize=4)):
                done[f] = text
                fh.write(json.dumps({"file": f, "text": text}) + "\n")
    labels["text"] = labels.file.map(done)
    return labels


def to_num(s):
    s = s.replace(" ", "")
    if re.search(r",\d{2}$", s) and "." in s[:-3]:  # 1.234,56 style
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s.replace(",", ""))
    except ValueError:  # OCR merged two numbers, e.g. 17.417274.72
        return None


def arithmetic(text):
    """Parse line amounts, subtotal, tax and total from OCR text and check that they add up."""
    items, sub, tax, tot = [], None, None, None
    for line in text.lower().splitlines():
        nums = [x for x in (to_num(m) for m in MONEY.findall(line)) if x is not None]
        if not nums:
            continue
        if "subtotal" in line or "sub total" in line:
            sub = nums[-1]
        elif "vat" in line or "tax" in line:
            tax = nums[-1]
        elif any(w in line for w in TOTAL_WORDS):
            tot = nums[-1]
        elif len(nums) >= 2 and sub is None:
            items.append(nums[-1])
    parsed = bool(items) and sub is not None and tot is not None
    if not parsed:
        return {"parsed": 0, "sum_ok": 1, "total_ok": 1, "mismatch": 0}
    sum_ok = abs(sum(items) - sub) <= 0.05
    total_ok = abs(sub + (tax or 0) - tot) <= 0.05
    return {"parsed": 1, "sum_ok": int(sum_ok), "total_ok": int(total_ok), "mismatch": int(not (sum_ok and total_ok))}


def features(df):
    rules = pd.DataFrame([arithmetic(t) for t in df.text])
    return pd.concat([df.reset_index(drop=True), rules], axis=1)


def report(y_true, y_pred):
    rep = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return {"macro_f1": round(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0), 3),
            "per_class": {k: {m: round(rep[k][m], 3) for m in ("precision", "recall", "f1-score", "support")} for k in LABELS},
            "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist()}


def main(data):
    data = Path(data)
    df = features(run_ocr(data))
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    out = {"sizes": {s: int((df.split == s).sum()) for s in ("train", "val", "test")},
           "parse_rate": round(df.parsed.mean(), 3)}

    # 1. rule only: flag total_mismatch when the numbers do not add up, otherwise say clean
    rule = lambda d: np.where(d.mismatch == 1, "total_mismatch", "clean")  # noqa: E731
    out["rule_only"] = {"val": report(va.label, rule(va)), "test": report(te.label, rule(te))}

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    Xtr_t, Xva_t, Xte_t = vec.fit_transform(tr.text), vec.transform(va.text), vec.transform(te.text)
    rule_cols = ["parsed", "sum_ok", "total_ok", "mismatch"]
    R = lambda d: csr_matrix(d[rule_cols].values.astype(float) * 3.0)  # noqa: E731

    def fit_eval(Xtr, Xva, Xte, name):
        best = None
        for C in (0.3, 1, 3, 10, 30):
            m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, tr.label)
            f = f1_score(va.label, m.predict(Xva), average="macro")
            if best is None or f > best[0]:
                best = (f, C, m)
        _, C, m = best
        pv, pt = m.predict(Xva), m.predict(Xte)
        out[name] = {"C": C, "val": report(va.label, pv), "test": report(te.label, pt)}
        return m, pt

    fit_eval(Xtr_t, Xva_t, Xte_t, "tfidf_only")
    model, pred = fit_eval(hstack([Xtr_t, R(tr)]).tocsr(), hstack([Xva_t, R(va)]).tocsr(),
                           hstack([Xte_t, R(te)]).tocsr(), "baseline_tfidf_plus_rule")

    # error analysis on the test set
    te = te.assign(pred=pred)
    hid = te[te.label == "hidden_instruction"].copy()
    hid["shade"] = hid.tamper_info.map(lambda s: json.loads(s)["shade"])
    hid["caught"] = hid.pred == "hidden_instruction"
    hid["band"] = pd.cut(hid.shade, [160, 200, 220, 240], labels=["165-200 (visible grey)", "201-220 (light)", "221-238 (near white)"])
    out["hidden_by_shade"] = {str(k): {"n": int(len(g)), "caught": int(g.caught.sum())} for k, g in hid.groupby("band", observed=False)}
    ed = te[te.label == "edited_amount_bank"].copy()
    ed["kind"] = ed.tamper_info.map(lambda s: json.loads(s)["kind"])
    out["edited_by_kind"] = {k: {"n": int(len(g)), "caught": int((g.pred == "edited_amount_bank").sum())} for k, g in ed.groupby("kind")}
    tm = te[te.label == "total_mismatch"]
    out["mismatch_parse"] = {"n": int(len(tm)), "parsed": int(tm.parsed.sum()), "rule_flagged": int(tm.mismatch.sum())}
    names = np.array(list(vec.get_feature_names_out()) + rule_cols)
    ci = list(model.classes_).index("hidden_instruction")
    out["top_hidden_terms"] = list(names[np.argsort(model.coef_[ci])[::-1][:15]])
    res = data / "results"
    res.mkdir(exist_ok=True)
    (res / "results.json").write_text(json.dumps(out, indent=2))
    te[["file", "template", "label", "pred", "parsed", "mismatch"]].to_csv(res / "test_predictions.csv", index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    o = main(ap.parse_args().data)
    print(json.dumps({k: v["test"]["macro_f1"] for k, v in o.items() if isinstance(v, dict) and isinstance(v.get("test"), dict)}, indent=1))
