import streamlit as st
from PIL import Image
from dummy_model import dummy_predict


st.set_page_config(page_title="Infant Skin Classifier", layout="centered")
st.title("🧒 Infant Skin Condition Classifier")

st.markdown("""
Upload a photo of the affected skin area. The model will try to classify the skin condition (dummy version).
""")

uploaded_file = st.file_uploader("📸 Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("🧠 Analyze"):
        label, confidence = dummy_predict(img)
        st.success(f"Predicted condition: **{label}** ({confidence * 100:.1f}% confidence)")