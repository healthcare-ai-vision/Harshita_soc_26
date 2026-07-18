import streamlit as st
from PIL import Image

from model_utils import (
    predict_skin_classification,
    predict_skin_segmentation,
    predict_throat_classification,
)
from gemini_report import generate_medical_report

st.set_page_config(page_title="AI Health Assistant", layout="centered")
st.title("AI Health Assistant")
st.caption(
    "Skin & throat condition screening demo — for educational/project purposes only, "
    "not a substitute for professional medical advice."
)

domain = st.radio("What are you uploading a photo of?", ["Skin", "Throat"], horizontal=True)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Analyze", type="primary"):
        results = {}

        with st.spinner("Running your models..."):
            if domain == "Skin":
                label, confidence, all_scores = predict_skin_classification(image)
                results["classification"] = {
                    "domain": "skin",
                    "predicted_label": label,
                    "confidence": round(confidence, 4),
                    "all_scores": {k: round(v, 4) for k, v in all_scores.items()},
                }

                mask, affected_pct = predict_skin_segmentation(image)
                results["segmentation"] = {"affected_area_percent": round(affected_pct, 2)}

                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original")
                with col2:
                    st.image(mask, caption=f"Affected area (~{affected_pct:.1f}%)")
            else:
                label, confidence, all_scores = predict_throat_classification(image)
                results["classification"] = {
                    "domain": "throat",
                    "predicted_label": label,
                    "confidence": round(confidence, 4),
                    "all_scores": {k: round(v, 4) for k, v in all_scores.items()},
                }

        st.subheader("Model output")
        st.write(f"**Predicted condition:** {results['classification']['predicted_label']}")
        st.write(f"**Confidence:** {results['classification']['confidence'] * 100:.1f}%")
        if domain == "Skin":
            st.write(
                f"**Affected area:** {results['segmentation']['affected_area_percent']:.1f}% "
                "of the visible skin region"
            )
        with st.expander("Full raw scores"):
            st.json(results)

        with st.spinner("Generating explanation with Gemini..."):
            try:
                report = generate_medical_report(results)
            except Exception as e:
                report = f"(Could not reach Gemini API: {e})"

        st.subheader("Assistant's assessment")
        st.write(report)

        st.warning(
            "This tool is a student project, not a medical device. If symptoms persist, "
            "worsen, or you're worried, please see a doctor regardless of this result."
        )
