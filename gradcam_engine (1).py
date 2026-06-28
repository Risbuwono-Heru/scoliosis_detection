# -*- coding: utf-8 -*-
"""
gradcam_engine.py

Grad-CAM deployment engine for Scoliosis GUI.
Digunakan sebagai library oleh app.py (Streamlit).
"""

# ==========================================================
# IMPORTS
# ==========================================================

import os
import json
import warnings
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIGURATION
# ==========================================================

CONFIG = {

    "image_size": (224, 224),

    "color_mode": "rgb",

    "normalization": "divide_255",

    "prediction_threshold": 0.5,

    "overlay_alpha": 0.45,

    "heatmap_threshold": 0.5,

    "figure_size": (16, 6),

    "positive_class": "scoliosis",

    "negative_class": "normal"

}

# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent

# ==========================================================
# PROJECT PATHS
# ==========================================================

MODEL_DIR = PROJECT_DIR / "Model"

METADATA_DIR = PROJECT_DIR / "Metadata"

OUTPUT_DIR = PROJECT_DIR / "Output"

OVERLAY_DIR = OUTPUT_DIR / "Overlay"

REPORT_DIR = OUTPUT_DIR / "Report"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# RESOURCE DISCOVERY
# ==========================================================

MODEL_FILES = sorted(

    MODEL_DIR.glob("*.keras")

)

SUMMARY_PATH = METADATA_DIR / "evaluation_summary.json"

PREDICTION_PATH = METADATA_DIR / "prediction_results.csv"

with open(SUMMARY_PATH, "r") as f:

    EVALUATION_SUMMARY = json.load(f)

PREDICTION_RESULT = pd.read_csv(
    PREDICTION_PATH
)

# ==========================================================
# MODEL REGISTRY
# ==========================================================

MODEL_REGISTRY = {}

for model_path in MODEL_FILES:

    filename = model_path.stem.lower()

    architecture = (
        "DenseNet121"
        if "densenet" in filename
        else "ResNet50"
    )

    experiment = filename.split("_")[-1].upper()

    MODEL_REGISTRY[model_path.name] = {

        "architecture": architecture,

        "experiment": experiment,

        "path": model_path

    }

# ==========================================================
# APPLICATION CONTEXT
# ==========================================================

APP_CONTEXT = {

    "config": CONFIG,

    "model_registry": MODEL_REGISTRY,

    "evaluation_summary": EVALUATION_SUMMARY,

    "prediction_result": PREDICTION_RESULT,

    "current_model": None,

    "current_model_name": None

}

APP_CONTEXT["current_image"] = None

APP_CONTEXT["current_prediction"] = None

APP_CONTEXT["current_heatmap"] = None

APP_CONTEXT["current_analysis"] = None

APP_CONTEXT["current_report"] = None

APP_CONTEXT["current_last_conv_layer"] = None

APP_CONTEXT["current_result"] = None

# ==========================================================
# MODEL LOADER
# ==========================================================

def load_selected_model(model_name):
    """
    Load selected model from MODEL_REGISTRY.

    Parameters
    ----------
    model_name : str
        Filename stored inside MODEL_REGISTRY.

    Returns
    -------
    tf.keras.Model
    """

    if model_name not in MODEL_REGISTRY:

        raise ValueError(
            f"Unknown model : {model_name}"
        )

    model_path = MODEL_REGISTRY[model_name]["path"]

    model = load_model(model_path)

    APP_CONTEXT["current_model"] = model

    APP_CONTEXT["current_model_name"] = model_name

    return model

# ==========================================================
# IMAGE LOADER & PREPROCESSING
# ==========================================================

def load_and_preprocess_image(image_path):
    """
    Load and preprocess an image for prediction.

    Parameters
    ----------
    image_path : str or Path

    Returns
    -------
    result : dict
        {
            "original_image" : PIL.Image,
            "image_array"    : np.ndarray,
            "image_path"     : str
        }
    """

    # ------------------------------------------------------
    # Load Image
    # ------------------------------------------------------

    original_image = image.load_img(
        image_path,
        target_size=CONFIG["image_size"]
    )

    # ------------------------------------------------------
    # Convert to Array
    # ------------------------------------------------------

    image_array = image.img_to_array(
        original_image
    )

    image_array = image_array.astype(
        np.float32
    )

    # ------------------------------------------------------
    # Normalize
    # ------------------------------------------------------

    if CONFIG["normalization"] == "divide_255":

        image_array /= 255.0

    # ------------------------------------------------------
    # Batch Dimension
    # ------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # ------------------------------------------------------
    # Save Current Image
    # ------------------------------------------------------

    APP_CONTEXT["current_image"] = str(image_path)

    return {

        "original_image": original_image,

        "image_array": image_array,

        "image_path": str(image_path)

    }

# ==========================================================
# CONFIDENCE LEVEL
# ==========================================================

def get_confidence_level(confidence):
    """
    Convert prediction confidence into a qualitative level.
    """

    if confidence >= 0.90:
        return "Very High"

    elif confidence >= 0.75:
        return "High"

    elif confidence >= 0.60:
        return "Moderate"

    else:
        return "Low"

# ==========================================================
# PREDICTION ENGINE
# ==========================================================

def predict_image(image_result):
    """
    Perform image prediction using the currently loaded model.

    Parameters
    ----------
    image_result : dict

    Returns
    -------
    prediction_result : dict
    """

    if APP_CONTEXT["current_model"] is None:

        raise RuntimeError(
            "No model has been loaded."
        )

    model = APP_CONTEXT["current_model"]

    image_array = image_result["image_array"]

    probabilities = model.predict(
        image_array,
        verbose=0
    )

    # ------------------------------------------------------
    # Automatic Output Detection
    # ------------------------------------------------------

    if probabilities.shape[1] == 1:

        confidence_scoliosis = float(
            probabilities[0][0]
        )

        confidence_normal = (
            1.0 - confidence_scoliosis
        )

        predicted_label = int(
            confidence_scoliosis >=
            CONFIG["prediction_threshold"]
        )

    else:

        confidence_normal = float(
            probabilities[0][0]
        )

        confidence_scoliosis = float(
            probabilities[0][1]
        )

        predicted_label = int(
            np.argmax(probabilities)
        )

    # ------------------------------------------------------

    predicted_class = (
        CONFIG["positive_class"]
        if predicted_label == 1
        else CONFIG["negative_class"]
    )

    confidence = max(

        confidence_normal,

        confidence_scoliosis

    )

    prediction_result = {

        "predicted_label": predicted_label,

        "predicted_class": predicted_class,

        "confidence": confidence,

        "confidence_level":
            get_confidence_level(confidence),

        "confidence_normal":
            confidence_normal,

        "confidence_scoliosis":
            confidence_scoliosis,

        "probabilities":
            probabilities

    }

    APP_CONTEXT["current_prediction"] = prediction_result

    return prediction_result

# ==========================================================
# AUTOMATIC GRAD-CAM LAYER DISCOVERY
# ==========================================================

def discover_gradcam_layer(model):
    """
    Automatically discover the best layer for Grad-CAM.

    Priority
    --------
    1. Final ReLU / Activation layer with 4D output.
    2. Last Conv2D layer.
    3. Last DepthwiseConv2D layer.
    """

    # ------------------------------------------------------
    # Priority 1 : Activation Layer
    # ------------------------------------------------------

    for layer in reversed(model.layers):

        if isinstance(
            layer,
            tf.keras.layers.Activation
        ):

            try:

                output_shape = layer.output.shape

                if len(output_shape) == 4:

                    APP_CONTEXT["current_last_conv_layer"] = layer.name

                    return layer

            except Exception:

                continue

    # ------------------------------------------------------
    # Priority 2 : Conv2D / DepthwiseConv2D
    # ------------------------------------------------------

    supported_layers = (

        tf.keras.layers.Conv2D,

        tf.keras.layers.DepthwiseConv2D

    )

    for layer in reversed(model.layers):

        if isinstance(layer, supported_layers):

            APP_CONTEXT["current_last_conv_layer"] = layer.name

            return layer

    raise RuntimeError(
        "No suitable Grad-CAM layer was found."
    )

# ==========================================================
# GENERATE GRAD-CAM HEATMAP
# ==========================================================

def generate_heatmap(image_result):
    """
    Generate Grad-CAM heatmap using the currently loaded model.
    """

    model = APP_CONTEXT["current_model"]

    layer_name = APP_CONTEXT["current_last_conv_layer"]

    image_array = image_result["image_array"]

    grad_model = tf.keras.models.Model(

        inputs=model.inputs,

        outputs=[

            model.get_layer(layer_name).output,

            model.output

        ]

    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image_array
        )

        # Binary Classification
        if predictions.shape[-1] == 1:

            class_score = predictions[:, 0]

        else:

            class_index = tf.argmax(
                predictions[0]
            )

            class_score = predictions[:, class_index]

    gradients = tape.gradient(

        class_score,

        conv_outputs

    )

    pooled_gradients = tf.reduce_mean(

        gradients,

        axis=(0, 1, 2)

    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(

        conv_outputs * pooled_gradients,

        axis=-1

    )

    heatmap = tf.maximum(

        heatmap,

        0

    )

    max_value = tf.reduce_max(heatmap)

    if max_value > 0:

        heatmap /= max_value

    heatmap = heatmap.numpy()

    APP_CONTEXT["current_heatmap"] = heatmap

    return heatmap

# ==========================================================
# GENERATE OVERLAY
# ==========================================================

def generate_overlay(
    image_result,
    heatmap
):
    """
    Create Grad-CAM overlay image.
    """

    original = np.array(
        image_result["original_image"]
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    heatmap_uint8 = cv2.resize(

        heatmap_uint8,

        (

            original.shape[1],

            original.shape[0]

        )

    )

    colored_heatmap = cv2.applyColorMap(

        heatmap_uint8,

        cv2.COLORMAP_JET

    )

    colored_heatmap = cv2.cvtColor(

        colored_heatmap,

        cv2.COLOR_BGR2RGB

    )

    overlay = cv2.addWeighted(

        original,

        1 - CONFIG["overlay_alpha"],

        colored_heatmap,

        CONFIG["overlay_alpha"],

        0

    )

    return overlay

# ==========================================================
# RUN GRAD-CAM
# ==========================================================

def run_gradcam(image_result):
    """
    Execute complete Grad-CAM pipeline.
    """

    heatmap = generate_heatmap(
        image_result
    )

    overlay = generate_overlay(

        image_result,

        heatmap

    )

    result = {

        "heatmap": heatmap,

        "overlay": overlay

    }

    APP_CONTEXT["current_heatmap"] = result

    return result

# ==========================================================
# HEATMAP STATISTICS
# ==========================================================

def compute_heatmap_statistics(heatmap):
    """
    Compute basic Grad-CAM heatmap statistics.
    """

    activation_mask = (
        heatmap >= CONFIG["heatmap_threshold"]
    )

    coverage = (
        np.sum(activation_mask)
        / activation_mask.size
    ) * 100

    max_activation = float(
        np.max(heatmap)
    )

    mean_activation = float(
        np.mean(heatmap)
    )

    y_center, x_center = np.argwhere(
        activation_mask
    ).mean(axis=0) if np.any(activation_mask) else (
        heatmap.shape[0] / 2,
        heatmap.shape[1] / 2
    )

    return {

        "coverage": coverage,

        "maximum_activation": max_activation,

        "mean_activation": mean_activation,

        "center_x": float(x_center),

        "center_y": float(y_center)

    }

# ==========================================================
# DOMINANT REGION
# ==========================================================

def identify_region(stats):
    """
    Identify dominant activation region
    using a 3x3 grid.
    """

    x = stats["center_x"]

    y = stats["center_y"]

    grid = 7 / 3

    col = min(int(x // grid), 2)

    row = min(int(y // grid), 2)

    region_names = {

        (0, 0): "Upper Left",

        (0, 1): "Upper Center",

        (0, 2): "Upper Right",

        (1, 0): "Middle Left",

        (1, 1): "Central Thoracic Region",

        (1, 2): "Middle Right",

        (2, 0): "Lower Left",

        (2, 1): "Lower Center",

        (2, 2): "Lower Right"

    }

    return region_names[(row, col)]

# ==========================================================
# TECHNICAL INTERPRETATION
# ==========================================================

def technical_interpretation(stats):

    interpretation = []

    if stats["coverage"] >= 40:

        interpretation.append(
            "Activation is broadly distributed."
        )

    else:

        interpretation.append(
            "Activation is relatively localized."
        )

    if stats["mean_activation"] >= 0.30:

        interpretation.append(
            "Moderate average activation intensity."
        )

    else:

        interpretation.append(
            "Low average activation intensity."
        )

    if stats["maximum_activation"] >= 0.90:

        interpretation.append(
            "Strong activation peak detected."
        )

    else:

        interpretation.append(
            "Moderate activation peak detected."
        )

    return interpretation

# ==========================================================
# CLINICAL INTERPRETATION
# ==========================================================

def clinical_interpretation(
    prediction_result,
    region
):

    interpretation = []

    interpretation.append(

        f"Primary attention is concentrated around the {region.lower()}."

    )

    interpretation.append(

        "The model appears to utilize multiple anatomical structures during prediction."

    )

    if prediction_result["predicted_class"] == CONFIG["positive_class"]:

        interpretation.append(

            "The highlighted regions likely contain image features associated with scoliosis."

        )

    else:

        interpretation.append(

            "The highlighted regions likely contain image features associated with normal spinal alignment."

        )

    return interpretation

# ==========================================================
# ANALYSIS ENGINE
# ==========================================================

def analyze_gradcam(
    gradcam_result,
    prediction_result
):

    stats = compute_heatmap_statistics(

        gradcam_result["heatmap"]

    )

    region = identify_region(
        stats
    )

    analysis = {

        **stats,

        "dominant_region": region,

        "technical_interpretation":

            technical_interpretation(stats),

        "clinical_interpretation":

            clinical_interpretation(

                prediction_result,

                region

            )

    }

    APP_CONTEXT["current_analysis"] = analysis

    return analysis

# ==========================================================
# REPORT ENGINE
# ==========================================================

def generate_report(
    prediction_result,
    analysis_result
):
    """
    Generate structured report and formatted report.
    """

    structured_report = {

        "generated_time":

            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "model":

            APP_CONTEXT["current_model_name"],

        "prediction":

            prediction_result,

        "analysis":

            analysis_result

    }

    formatted_report = f"""
======================================================================
SPINE X-RAY CLASSIFICATION REPORT
======================================================================

Generated Time
-------------------------
{structured_report['generated_time']}

Model
-------------------------
{structured_report['model']}

Prediction
-------------------------
Predicted Class : {prediction_result['predicted_class']}
Confidence      : {prediction_result['confidence']:.2%}
Confidence Level: {prediction_result['confidence_level']}

Heatmap Statistics
-------------------------
Coverage            : {analysis_result['coverage']:.2f}%
Maximum Activation  : {analysis_result['maximum_activation']:.4f}
Mean Activation     : {analysis_result['mean_activation']:.4f}
Dominant Region     : {analysis_result['dominant_region']}

Technical Interpretation
-------------------------
"""

    for item in analysis_result[
        "technical_interpretation"
    ]:

        formatted_report += f"\n• {item}"

    formatted_report += """

Clinical Interpretation
-------------------------
"""

    for item in analysis_result[
        "clinical_interpretation"
    ]:

        formatted_report += f"\n• {item}"

    report = {

        "structured": structured_report,

        "formatted": formatted_report

    }

    APP_CONTEXT["current_report"] = report

    return report

# ==========================================================
# UNIFIED DEPLOYMENT PIPELINE
# ==========================================================

def run_pipeline(image_path):
    """
    Complete deployment pipeline.

    Parameters
    ----------
    image_path : str or Path

    Returns
    -------
    dict
    """

    # ------------------------------------------------------
    # Load Image
    # ------------------------------------------------------

    image_result = load_and_preprocess_image(
        image_path
    )

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    prediction_result = predict_image(
        image_result
    )
    
    # ------------------------------------------------------
    # Discover Grad-CAM Layer
    # ------------------------------------------------------

    discover_gradcam_layer(
        APP_CONTEXT["current_model"]
    )

    # ------------------------------------------------------
    # Grad-CAM
    # ------------------------------------------------------

    gradcam_result = run_gradcam(
        image_result
    )

    # ------------------------------------------------------
    # Analysis
    # ------------------------------------------------------

    analysis_result = analyze_gradcam(

        gradcam_result,

        prediction_result

    )

    # ------------------------------------------------------
    # Report
    # ------------------------------------------------------

    report_result = generate_report(

        prediction_result,

        analysis_result

    )

    # ------------------------------------------------------
    # Pipeline Result
    # ------------------------------------------------------

    pipeline_result = {

        "image": image_result,

        "prediction": prediction_result,

        "gradcam": gradcam_result,

        "analysis": analysis_result,

        "report": report_result

    }

    APP_CONTEXT["current_result"] = pipeline_result

    return pipeline_result
