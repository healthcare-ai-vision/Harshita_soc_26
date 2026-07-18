# AI Health Assistant — setup guide

## What this app does
1. You pick "Skin" or "Throat" and upload a photo.
2. Your own trained models run inference (classification, and for skin,
   segmentation too).
3. The structured results (predicted class, confidence, affected-area %)
   are sent to the Gemini API, which turns them into a plain-language
   assessment: what it likely is, how serious, and whether to see a doctor.

Gemini never sees the raw image or makes the diagnosis itself — your models
do the actual detection. Gemini's only job is to explain the numbers in
plain English. This keeps the medical judgment tied to your trained models'
actual confidence scores, not to Gemini guessing from a photo.

## Files
- `app.py` — the Streamlit UI (upload, buttons, layout). You shouldn't need
  to change this much.
- `model_utils.py` — loads your 3 Ultralytics YOLOv8 models (already matched
  to your training scripts: skin classifier at imgsz=224, segmentation and
  throat classifier at imgsz=640). You just need to supply the weight files.
- `gemini_report.py` — calls the Gemini API with the model results.
- `requirements.txt` — Python packages to install.

## Setup (one-time)

1. **Get your trained weights out of Colab.** After each training cell
   finishes, Ultralytics saves the best checkpoint at:
   ```
   runs/classify/throat disease classification/weights/best.pt
   runs/segment/affected skin segmentation/weights/best.pt
   runs/classify/skin classification/weights/best.pt
   ```
   Download those 3 `best.pt` files from Colab (right-click in the file
   browser panel -> Download, or `from google.colab import files; files.download(path)`).

2. **Rename and place them** in a `models/` folder next to `app.py`:
   ```
   healthbot/
     app.py
     model_utils.py
     gemini_report.py
     models/
       skin_classifier.pt        <- from "skin classification" run
       skin_segmentation.pt      <- from "affected skin segmentation" run
       throat_classifier.pt      <- from "throat disease classification" run
   ```
   No need to edit class labels or preprocessing — Ultralytics stores the
   class names and handles resizing internally, since `model_utils.py`
   already uses the same `imgsz` you trained each model with.

3. **Get a Gemini API key:** https://aistudio.google.com/apikey (free tier
   is enough for this).

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set your API key** (don't hardcode it in the file):
   - macOS/Linux: `export GEMINI_API_KEY="your-key-here"`
   - Windows (PowerShell): `$env:GEMINI_API_KEY="your-key-here"`

6. **Run it:**
   ```bash
   streamlit run app.py
   ```
   This opens the app in your browser automatically (usually
   `http://localhost:8501`).

## Common issues
- **`FileNotFoundError: models/skin_classifier.pt`** — you haven't downloaded/
  renamed the weights from Colab yet, or put them in the wrong folder (see
  step 1-2 above).
- **Segmentation mask always empty / 0% affected area** — the model found no
  detections above its default confidence threshold. You can lower it by
  passing `conf=0.25` (or lower) into `skin_seg_model.predict(...)` in
  `model_utils.py`.
- **`GEMINI_API_KEY environment variable is not set`** — you skipped the API
  key step, or set it in a different terminal window than the one running
  Streamlit.
- **Slow first run** — Ultralytics downloads some small support files on
  first use; subsequent runs are fast.

## Security note
Your training scripts had a Roboflow API key hardcoded in plain text. Rotate
that key in your Roboflow account settings, and avoid committing API keys
directly in scripts going forward — use environment variables instead (as
this app already does for the Gemini key).
