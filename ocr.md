# OCR Pipeline — Implementation Guide

PDF → per-page text files using a vision LLM. Use this pattern when standard
OCR tools (Tesseract, Google Document AI, Cloud Vision) produce poor output on
your documents — degraded scans, historical typefaces, handwriting, or complex
layouts.

**Why a vision LLM over specialized OCR services:**
- Specialized OCR services do character recognition; vision LLMs do reading —
  they understand context, correct obvious transcription errors, and handle
  unusual fonts and layouts
- On heavily degraded source material, vision LLMs consistently return far more
  usable text than specialized services
- Slower (15–30s/page vs. <1s) and more expensive, but accuracy is not
  comparable on hard documents

**Which vision LLM to use:** Claude, Gemini, and GPT-4o all share this
advantage over specialized OCR — any capable vision LLM will perform similarly.
Choose based on cost, rate limits, and your existing API setup. The code below
uses Claude as the reference implementation; swapping the API call is the only
change needed.

---

## Stack

- **PyMuPDF** (`fitz`) — PDF → PIL image per page
- **Pillow** — image preprocessing
- **Anthropic SDK** — Claude vision API
- **`ocr/index.json`** — per-page state: done/not-done, quality scores, used_vision flag

## Dependencies

```bash
pip install pymupdf pillow anthropic
```

---

## 1. PDF → Image

```python
import fitz
from PIL import Image

def page_to_image(pdf_path: Path, page_num: int, dpi: int = 300) -> Image.Image:
    doc = fitz.open(str(pdf_path))
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img
```

300 DPI is the sweet spot — higher doesn't improve Claude's accuracy but increases
payload size. Use RGB (not grayscale) even though the source is B&W; some services
reject single-channel images.

---

## 2. Claude Vision OCR

```python
import base64, io, anthropic

# Tailor this prompt to your document type. Key elements:
#   1. Describe the source material (era, medium, condition)
#   2. Specify what to preserve (line breaks, formatting, spelling)
#   3. Suppress commentary so the output is clean text only
VISION_PROMPT = (
    "This is a scanned page of [document type, e.g. 'typewritten text from the 1950s']. "
    "Transcribe the text exactly as written, preserving line breaks. "
    "Output only the transcribed text with no commentary."
)

def claude_vision_ocr(img: Image.Image, client: anthropic.Anthropic, model="claude-haiku-4-5-20251001") -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode()

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": VISION_PROMPT},
            ],
        }],
    )
    return response.content[0].text.strip()
```

Haiku is fast and cheap for transcription. Upgrade to Sonnet if the document is
highly degraded or contains complex layout (tables, columns, handwriting).

---

## 3. Quality Scoring

Detect bad OCR output before storing it. Two signals work better than one:

```python
import re

# Tune these thresholds for your document type.
# Dense prose: use values below. Numeric/tabular content: raise digit threshold.
DIGIT_THRESHOLD = 0.06       # >6% words are pure digits → noise
SUSPICIOUS_THRESHOLD = 0.07  # >7% words contain special chars → noise

def compute_scores(text: str) -> dict:
    words = text.split()
    if not words:
        return {"digit_ratio": 0.0, "suspicious_ratio": 0.0, "word_count": 0}
    digit_ratio = sum(1 for w in words if re.fullmatch(r"\d+", w)) / len(words)
    suspicious_ratio = sum(
        1 for w in words if re.search(r'[^a-zA-Z\s\'\-\.,;:!?"\(\)]', w)
    ) / len(words)
    return {
        "digit_ratio": round(digit_ratio, 4),
        "suspicious_ratio": round(suspicious_ratio, 4),
        "word_count": len(words),
    }

def is_bad_ocr(scores: dict) -> bool:
    if scores["word_count"] < 20:
        return False  # blank page, don't bother retrying
    return scores["digit_ratio"] > DIGIT_THRESHOLD or scores["suspicious_ratio"] > SUSPICIOUS_THRESHOLD
```

---

## 4. Resume-Safe Index

Store per-page state so any step can be interrupted and restarted:

```python
import json
from pathlib import Path

INDEX_PATH = Path("ocr/index.json")

def load_index() -> dict:
    if INDEX_PATH.exists():
        return {f"{e['doc_id']}_{e['page']:04d}": e
                for e in json.loads(INDEX_PATH.read_text())}
    return {}

def save_index(index: dict):
    entries = sorted(index.values(), key=lambda e: (e["doc_id"], e["page"]))
    INDEX_PATH.write_text(json.dumps(entries, indent=2))

# Per-page entry shape:
{
    "doc_id": "my-document",
    "page": 0,
    "text_file": "my-document_p0000.txt",
    "char_count": 1423,
    "flagged_blank": False,
    "used_vision": True,
    "scores": {"digit_ratio": 0.02, "suspicious_ratio": 0.01, "word_count": 241},
}
```

Key: write the index after every page, not at the end. A crash mid-run should
lose at most one page.

---

## 5. Full Page Loop

```python
def process_page(doc_id, page_num, pdf_path, index, client):
    key = f"{doc_id}_{page_num:04d}"
    if key in index:
        return  # already done

    txt_path = Path("ocr") / f"{doc_id}_p{page_num:04d}.txt"
    img = page_to_image(pdf_path, page_num)
    text = claude_vision_ocr(img, client)
    scores = compute_scores(text)

    txt_path.write_text(text, encoding="utf-8")
    index[key] = {
        "doc_id": doc_id,
        "page": page_num,
        "text_file": txt_path.name,
        "char_count": len(text),
        "flagged_blank": len(text) < 50,
        "used_vision": True,
        "scores": scores,
    }
    save_index(index)
```

---

## 6. Upgrading Tesseract Pages to Vision

If you initially ran Tesseract and want to upgrade all non-vision pages to Claude
after the fact, the index makes this straightforward:

```python
candidates = [
    entry for entry in index.values()
    if not entry.get("used_vision") and not entry.get("flagged_blank")
]
for entry in candidates:
    img = page_to_image(pdf_path, entry["page"])
    text = claude_vision_ocr(img, client)
    key = f"{entry['doc_id']}_{entry['page']:04d}"
    (Path("ocr") / entry["text_file"]).write_text(text, encoding="utf-8")
    index[key]["used_vision"] = True
    index[key]["scores"] = compute_scores(text)
    index[key]["char_count"] = len(text)
    save_index(index)
```

This is why tracking `used_vision` per page matters — it makes selective reprocessing
trivial without re-running everything.

---

## Known Gotchas

**Always send RGB images.** Even for B&W source documents, convert to RGB before
encoding. In testing, Google Vision and Document AI returned near-blank output
(7 words per page) when sent grayscale PNG — the same pages Claude handled correctly.
`img.convert("RGB")` before encoding costs nothing and avoids silent failures.

**Quality scoring catches Tesseract noise, not vision LLM noise.** The digit/suspicious
thresholds were designed to detect Tesseract garbage output. Vision LLM output rarely
triggers them — if a vision LLM page looks wrong, the problem is usually the prompt
or a genuinely blank/illegible page, not a threshold issue.

**Single-digit word counts mean silent failure.** A working service on a dense prose
page returns 150–300 words. If you see <20 words on a page that should have content,
the service failed without raising an error (wrong image format, API misconfiguration,
disabled API). Check the raw output before assuming the page is blank.

---

## Running a Bake-off

Before committing to any OCR service on a new document type, compare them on
your hardest page. Run Tesseract, Claude, and any cloud services side-by-side
and produce a quality/timing/cost comparison table.

The key metric is word count on pages that are known to have dense text — a
working service should return 150–300 words per full page of prose. Single-digit
word counts mean the service failed silently.
