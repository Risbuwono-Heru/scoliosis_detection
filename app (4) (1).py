# ==========================================================
# SPINE X-RAY SCOLIOSIS CLASSIFICATION SYSTEM
# Streamlit Application
# ==========================================================

# ==========================================================
# IMPORTS
# ==========================================================

from pathlib import Path

import streamlit as st
from PIL import Image
import numpy as np

import gradcam_engine as engine

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(

    page_title="Spine X-ray Scoliosis Classification",

    page_icon="🩻",

    layout="wide",

    initial_sidebar_state="expanded"

)

# ==========================================================
# APPLICATION HEADER
# ==========================================================

st.title(
    "🩻 Spine X-ray Scoliosis Classification System"
)

st.caption(
    "Deep Learning Classification with Explainable AI (Grad-CAM)"
)

st.divider()

# ==========================================================
# SESSION STATE
# ==========================================================

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

if "current_model" not in st.session_state:
    st.session_state.current_model = None

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("Configuration")

# ----------------------------------------------------------
# Model Selection
# ----------------------------------------------------------

model_names = list(engine.MODEL_REGISTRY.keys())

selected_model = st.sidebar.selectbox(

    "Select Model",

    model_names

)

# ----------------------------------------------------------
# Load Model
# ----------------------------------------------------------

if (

    not st.session_state.model_loaded

    or

    st.session_state.current_model != selected_model

):

    with st.spinner("Loading model..."):

        engine.load_selected_model(
            selected_model
        )

    st.session_state.model_loaded = True

    st.session_state.current_model = selected_model

st.sidebar.success("Model Loaded")

# ----------------------------------------------------------
# Upload Image
# ----------------------------------------------------------

uploaded_file = st.sidebar.file_uploader(

    "Upload Spine X-ray",

    type=["jpg", "jpeg", "png"]

)

# ----------------------------------------------------------
# Predict Button
# ----------------------------------------------------------

predict_button = st.sidebar.button(

    "Predict",

    use_container_width=True

)

st.sidebar.divider()

st.sidebar.info(

    f"Current Model\n\n{selected_model}"

)

# ==========================================================
# PREDICTION ENGINE
# ==========================================================

if predict_button:

    if uploaded_file is None:

        st.warning(
            "Please upload a spine X-ray image first."
        )

    else:

        # --------------------------------------------------
        # Save uploaded image temporarily
        # --------------------------------------------------

        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)

        image_path = temp_dir / uploaded_file.name

        with open(image_path, "wb") as f:

            f.write(uploaded_file.getbuffer())

        # --------------------------------------------------
        # Run Deployment Pipeline
        # --------------------------------------------------

        with st.spinner(
            "Running prediction..."
        ):

            pipeline_result = engine.run_pipeline(
                image_path
            )

        st.session_state.pipeline_result = (
            pipeline_result
        )

        st.success(
            "Prediction completed successfully."
        )

# ==========================================================
# DISPLAY RESULT
# ==========================================================

if st.session_state.pipeline_result is not None:

    result = st.session_state.pipeline_result

    image_result = result["image"]

    prediction = result["prediction"]

    gradcam = result["gradcam"]

    analysis = result["analysis"]

    report = result["report"]


# ==========================================================
# RESULT VISUALIZATION
# ==========================================================

if st.session_state.pipeline_result is not None:

    st.divider()

    # ------------------------------------------------------
    # IMAGE VISUALIZATION
    # ------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")

        st.image(
            image_result["original_image"],
            use_container_width=True
        )

    with col2:

        st.subheader("Grad-CAM Overlay")

        st.image(
            gradcam["overlay"],
            use_container_width=True
        )

    # ------------------------------------------------------
    # PREDICTION SUMMARY
    # ------------------------------------------------------

    st.divider()

    st.subheader("Prediction Summary")

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(
            "Predicted Class",
            prediction["predicted_class"].capitalize()
        )

    with m2:

        st.metric(
            "Confidence",
            f"{prediction['confidence']:.2%}"
        )

    with m3:

        st.metric(
            "Confidence Level",
            prediction["confidence_level"]
        )

    # ------------------------------------------------------
    # CLASS PROBABILITY
    # ------------------------------------------------------

    p1, p2 = st.columns(2)

    with p1:

        st.metric(
            "Normal",
            f"{prediction['confidence_normal']:.2%}"
        )

    with p2:

        st.metric(
            "Scoliosis",
            f"{prediction['confidence_scoliosis']:.2%}"
        )

    # ------------------------------------------------------
    # HEATMAP STATISTICS
    # ------------------------------------------------------

    st.divider()

    st.subheader("Heatmap Statistics")

    s1, s2, s3, s4 = st.columns(4)

    with s1:

        st.metric(
            "Coverage",
            f"{analysis['coverage']:.2f}%"
        )

    with s2:

        st.metric(
            "Maximum",
            f"{analysis['maximum_activation']:.4f}"
        )

    with s3:

        st.metric(
            "Mean",
            f"{analysis['mean_activation']:.4f}"
        )

    with s4:

        st.metric(
            "Dominant Region",
            analysis["dominant_region"]
        )

    # ------------------------------------------------------
    # TECHNICAL INTERPRETATION
    # ------------------------------------------------------

    st.divider()

    st.subheader("Technical Interpretation")

    for item in analysis["technical_interpretation"]:

        st.write(f"• {item}")

    # ------------------------------------------------------
    # CLINICAL INTERPRETATION
    # ------------------------------------------------------

    st.divider()

    st.subheader("Clinical Interpretation")

    for item in analysis["clinical_interpretation"]:

        st.write(f"• {item}")

# ==========================================================
# SAVE RESULT
# ==========================================================

if st.session_state.pipeline_result is not None:

    st.divider()

    st.subheader("Export Result")

    col1, col2 = st.columns(2)

    # ------------------------------------------------------
    # SAVE OVERLAY
    # ------------------------------------------------------

    with col1:

        overlay = gradcam["overlay"]

        overlay_filename = (

            Path(image_result["image_path"]).stem

            + "_overlay.png"

        )

        overlay_path = (

            engine.OVERLAY_DIR /

            overlay_filename

        )

        import cv2

        cv2.imwrite(

            str(overlay_path),

            cv2.cvtColor(

                overlay,

                cv2.COLOR_RGB2BGR

            )

        )

        with open(

            overlay_path,

            "rb"

        ) as f:

            st.download_button(

                label="📥 Download Overlay",

                data=f,

                file_name=overlay_filename,

                mime="image/png",

                use_container_width=True

            )

    # ------------------------------------------------------
    # SAVE REPORT
    # ------------------------------------------------------

    with col2:

        report_filename = (

            Path(image_result["image_path"]).stem

            + "_report.txt"

        )

        report_path = (

            engine.REPORT_DIR /

            report_filename

        )

        with open(

            report_path,

            "w",

            encoding="utf-8"

        ) as f:

            f.write(

                report["formatted"]

            )

        with open(

            report_path,

            "rb"

        ) as f:

            st.download_button(

                label="📄 Download Report",

                data=f,

                file_name=report_filename,

                mime="text/plain",

                use_container_width=True

            )

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(

    "Spine X-ray Scoliosis Classification System | Deep Learning + Explainable AI (Grad-CAM)"

)

