import base64
import os

import requests
import streamlit as st

from app.knowledge import LANGS

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Plant Disease AI", page_icon="🌿", layout="wide")
st.title("🌿 Plant Disease Detection")

lang_name = st.sidebar.selectbox("Language / భాష", list(LANGS))
src = st.radio("Input", ["Upload", "Camera"], horizontal=True)
file = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png", "webp"]) if src == "Upload" \
    else st.camera_input("Take a photo")

if file and st.button("Analyze", type="primary"):
    with st.spinner("Analyzing..."):
        try:
            r = requests.post(f"{API}/predict",
                              files={"file": (file.name, file.getvalue(), file.type)},
                              data={"lang": LANGS[lang_name]}, timeout=60)
            r.raise_for_status()
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()
    res = r.json()
    c1, c2 = st.columns(2)
    with c1:
        st.image(base64.b64decode(res["annotated_image"]), caption="Processed image")
    with c2:
        p = res["prediction"]
        st.subheader(p["label"].replace("___", " — ").replace("_", " "))
        st.progress(min(p["confidence"], 1.0), text=f'Confidence {p["confidence"]:.1%}')
        st.metric("Severity", f'{res["severity"]["percent"]}%', res["severity"]["level"])
        if res["uncertain"]:
            st.warning("Low confidence — retake a clear, well-lit, close-up photo of one leaf.")
        elif res["advice"]:
            st.markdown("### " + res["advice"]["summary"])
            for t in res["advice"]["treatment"]:
                st.markdown(f"- {t}")
            if res["advice"]["source"] == "machine-translated":
                st.caption("Machine-translated text.")
        with st.expander("Top-3 predictions"):
            st.json(res["top3"])
    st.caption(res["disclaimer"])
