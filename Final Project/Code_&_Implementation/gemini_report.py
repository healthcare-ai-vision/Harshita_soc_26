import os
import json
from google import genai

# gemini-2.5-flash is the current stable, low-latency multimodal model as of
# mid-2026. If it's ever retired, swap this string for whatever "flash" model
# ai.google.dev/gemini-api/docs/models currently lists.
MODEL_NAME = "gemini-2.5-flash"


def _get_client():
    api_key = os.environ.get("################################################")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
        )
    return genai.Client(api_key=api_key)


SYSTEM_INSTRUCTIONS = """
You are a cautious medical-screening assistant inside a student research project.
You are given STRUCTURED OUTPUT from the student's own trained image classification
and segmentation models — not the raw image. Only reason about what's given; never
invent a diagnosis the models didn't produce.

Do the following, in plain language:
1. State the predicted condition and confidence level in one sentence.
2. Judge severity as Low / Moderate / High, using the confidence score and,
   for skin cases, the affected-area percentage together. Briefly say why.
3. Give one clear recommendation: "no action needed", "monitor and recheck in
   a few days", or "see a doctor soon/promptly" — pick the strongest one that
   fits low confidence, ambiguous results, or a concerning affected area.
4. End with a one-line reminder that this is not a medical diagnosis.

Keep the whole reply under 150 words, no markdown headers or bullet lists.
"""


def generate_medical_report(results: dict) -> str:
    client = _get_client()
    prompt = SYSTEM_INSTRUCTIONS + "\n\nModel output:\n" + json.dumps(results, indent=2)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    return response.text
