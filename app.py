import streamlit as st
from PIL import Image, ImageOps
from pathlib import Path
from src.infer import predict_pil
from src.config import PRED_THRESHOLD
import pandas as pd

st.set_page_config(page_title="Skin Condition Classifier", layout="centered")
st.title("Skin Condition Classifier")

st.caption(
    "This is a research prototype, not a medical device. "
    "For concerning or worsening symptoms, consult a clinician."
)

# --- Init session state ---
if "input_image_pil" not in st.session_state:
    st.session_state.input_image_pil = None  # currently selected image (PIL)
if "input_caption" not in st.session_state:
    st.session_state.input_caption = None    # caption for the preview

# --- Sidebar: filter by class and choose a demo sample ---
st.sidebar.header("Try sample images")
sample_dir = Path("demo_samples")
class_to_files = {}
thr = st.sidebar.slider("Decision threshold", 0.50, 0.95, PRED_THRESHOLD, 0.01)


if sample_dir.exists():
    # Build a dict: class -> list of files (search recursively)
    for cls_dir in sorted([p for p in sample_dir.iterdir() if p.is_dir()]):
        files = sorted([p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png"}])
        if files:
            class_to_files[cls_dir.name] = files

if class_to_files:
    classes = sorted(class_to_files.keys())
    sel_cls = st.sidebar.selectbox("Class", classes, key="demo_class")

    files = class_to_files.get(sel_cls, [])
    file_labels = [f.relative_to(sample_dir).as_posix() for f in files]
    idx = st.sidebar.selectbox("Image", list(range(len(files))), format_func=lambda i: file_labels[i], key="demo_image_idx")

    if st.sidebar.button("Use selected sample"):
        # Load the chosen sample and store in session state
        im = Image.open(files[idx]).convert("RGB")
        im = ImageOps.exif_transpose(im)
        st.session_state.input_image_pil = im
        st.session_state.input_caption = f"Sample: {files[idx].name}"
else:
    st.sidebar.info("Put some images into 'demo_samples/<class>/' to enable samples.")

# --- Sidebar tips  ---
with st.sidebar.expander("📸 How to take a good photo", expanded=False):
    st.markdown(
        """
- Use bright, even **natural light** (no flash glare).
- Hold the camera **close and steady**; keep the area **in focus**.
- Fill the frame with the **skin area**, avoid backgrounds.
- **No filters**; avoid heavy makeup/creams right before the photo.
        """
    )

# "About this app" in sidebar
with st.sidebar.expander("ℹ️ About this app", expanded=False):
    st.markdown(
        """
This research MVP predicts common skin conditions from photos and **abstains** when uncertain.
It is **not a medical device**. If symptoms are concerning or persistent, consult a clinician.

**Why I built it:** As a parent, I wanted a cautious tool to triage rashes and common conditions
without overconfident guesses.
        """
    )


# --- Main uploader (overrides the current sample if provided) ---
uploaded = st.file_uploader("Upload a close, well-lit photo (JPG/PNG)", type=["jpg","jpeg","png"])
if uploaded is not None:
    im = Image.open(uploaded).convert("RGB")
    im = ImageOps.exif_transpose(im)
    st.session_state.input_image_pil = im
    st.session_state.input_caption = "Uploaded image"

# --- Preview current image ---
img_to_analyze = st.session_state.input_image_pil
if img_to_analyze is not None:
    st.image(img_to_analyze, caption=st.session_state.input_caption or "Image", use_container_width=True)

col1, col2 = st.columns([1,1])
with col1:
    analyze_clicked = st.button("Analyze")
with col2:
    if st.button("Clear"):
        st.session_state.input_image_pil = None
        st.session_state.input_caption = None

# --- Run inference if there is an image and the button was clicked ---
if analyze_clicked:
    if img_to_analyze is None:
        st.warning("Please upload or select a sample image first.")
    else:
        try:
            with st.spinner("Analyzing..."):
                label, conf, probs, extra = predict_pil(img_to_analyze, threshold=thr)
        except Exception as e:
            st.error("Prediction failed. Try a clearer JPG/PNG or reload the app.")
            st.stop()
            
        left, right = st.columns([2, 1])

        with left:
            if label == "uncertain/healthy":
                st.warning(
                    f"Model is uncertain (top confidence {conf:.2f} < threshold {extra['threshold']:.2f}). "
                    "Showing 'uncertain/healthy'. Consider a clearer, closer photo or consult a clinician."
                )
                top1, p1 = extra["top1"]
                top2, p2 = extra["top2"]
                st.write(f"Most likely classes (not confident): **{top1}** ({p1:.2f}), then **{top2}** ({p2:.2f}).")
            else:
                msg = f"**Prediction:** {label}  \n**Confidence:** {conf:.2f}"
                if extra["borderline"]:
                    msg += "  \n_The top-2 classes are very close; treat as tentative._"
                st.success(msg)

            # Probabilities bar chart
            st.subheader("Class probabilities")
            df = pd.DataFrame({"class": list(probs.keys()), "prob": list(probs.values())})
            st.bar_chart(df.set_index("class"))
            st.caption(f"Decision threshold: {thr:.2f} (configurable in slider)")
        with right:
            st.markdown("### 🧾 Conditions glossary")
            GLOSSARY = {
                "acne": "Clogged follicles; comedones/papules/pustules.",
                "contact_dermatitis": "Itchy red rash after irritant/allergen exposure.",
                "eczema": "Chronic itchy inflammation; often flexural areas.",
                "psoriasis": "Well-demarcated plaques with silvery scale.",
                "rash": "Non-specific widespread redness/exanthem.",
                "scabies": "Intensely itchy burrows; web spaces, wrists.",
                "tinea_ringworm": "Ring-shaped scaly border (fungal).",
                "urticaria": "Transient raised wheals/hives; migratory.",
                "warts": "Small rough papules due to HPV."
            }
            # Show the predicted class first (if confident)
            if label in GLOSSARY:
                st.markdown(f"**{label}** — {GLOSSARY[label]}")
                st.markdown("---")
            # Then show others (short list)
            for cls, desc in GLOSSARY.items():
                if cls != label:
                    st.markdown(f"**{cls}** — {desc}")
            st.caption("_Descriptions are simplified and not diagnostic._")
        
            
