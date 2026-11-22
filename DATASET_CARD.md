# Dataset card: InvoiceGuard synthetic invoices v0.1

## Summary
1,200 synthetic invoice images (A4 at 150 dpi, JPEG) for detecting invoices that were changed to trick a person or an AI payment agent. Every image has one of four labels. The data was made by `invoicegen.py` with seed 7, so it can be rebuilt exactly on the same machine.

| Label | What was done to the invoice | Count |
|---|---|---|
| `clean` | nothing | 300 |
| `hidden_instruction` | a faint, small line of text tells an AI agent to pay another account (grey 165 to near white 238, 9 to 14 px) | 300 |
| `total_mismatch` | the printed subtotal or total is 4% to 35% off the sum of the items | 300 |
| `edited_amount_bank` | numbers pasted over the clean invoice: the bank account (half) or one line amount with subtotal, tax and total changed so that everything still adds up (half). Half the pastes are careful (same font), half are sloppy (other font, size or offset) | 300 |

## How it was made
- 12 templates with different fonts, colours, titles, words for the total and the bank field, currencies (NGN, USD, EUR, GBP, KES, ZAR, GHS) and VAT rates (0% to 20%).
- Fake seller and buyer names, addresses, invoice numbers, 3 to 7 line items, bank names and account numbers (10 digit or IBAN style). No real people or companies.
- Scan effects on every page: rotation up to 0.8 degrees, Gaussian blur on 70% of pages, JPEG quality 60 to 90, and 30% saved in grey.

## Split
By template, so the validation and test sets only contain layouts the model never saw.

| Split | Templates | Invoices |
|---|---|---|
| train | T00 to T07 | 800 |
| val | T08, T09 | 200 |
| test | T10, T11 | 200 |

Each split has the same number of invoices in every class.

## Files
- `images/inv_00000.jpg` to `images/inv_01199.jpg`
- `labels.csv`: file, template, split, label, the real payee account, total, currency, JPEG quality, and `tamper_info` (JSON: the injected text and its shade and size, the edited boxes, the attacker's account)
- `ocr.jsonl`: Tesseract 5 output for every image, so the baseline can run without OCR

## Intended use and limits
- For research on catching tampered invoices before an AI agent pays them.
- The invoices are cleaner and more regular than real ones. Results on this set will be higher than on real documents, which is why a small real test set (public sample invoices and SROIE receipts, edited by hand) and DocTamper are planned for week 5.
- Fonts come from the operating system. Rebuilding on Linux or Colab uses Liberation and DejaVu fonts, so the images differ slightly from the released set.

## Ethics
All data is fake. The hidden instruction texts describe a known attack in plain words and contain no working exploit against a real system. The dataset is meant to train a defence.
