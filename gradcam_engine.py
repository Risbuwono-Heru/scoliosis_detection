# -*- coding: utf-8 -*-
"""
gradcam_engine.py
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

MODEL_DIR    = PROJECT_DIR / "Model"
METADATA_DIR = PROJECT_DIR / "Metadata"
OUTPUT_DIR   = PROJECT_DIR / "Output"
OVERLAY_DIR  = OUTPUT_DIR / "Overlay"
REPORT_DIR   = OUTPUT_DIR / "Report"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# GOOGLE DRIVE MODEL MAP
# ==========================================================

DRIVE_MODEL_MAP = {
    "best_densenet121_e5.keras": "1H7RVy6gyX7oacoS7_d63_4OPSe5lLWgH",
    "best_densenet121_e4.keras": "1n3JdcdVfqYFNlYGVeywipspexElt8QPG",
    "best_resnet50_e3.keras":    "16LP1O5DgsdQFA3AdaIKLAVD3CpBuBpCs",
    "best_resnet50_e2.keras":    "133J3b37cFkZyrQMpzmz49cupaSzLJlCE",
    "best_resnet50_e1.keras":    "1ng1pesRs6J-CCDrFhG3QkQyyO51q_OuL",
}

# ==========================================================
# MODEL DOWNLOADER
# ==========================================================

def ensure_models_downloaded():
    import gdown
    for filename, file_id in DRIVE_MODEL_MAP.items():
        dest = MODEL_DIR / filename
        if not dest.exists():
            url = f"https://drive.google.com/uc?id={file_id}"
            print(f"Downloading {filename} ...")
            gdown.download(url, str(dest), quiet=False)

ensure_models_downloaded()

# ==========================================================
# RESOURCE DISCOVERY
# ==========================================================

MODEL_FILES = sorted(MODEL_DIR.glob("*.keras"))

SUMMARY_PATH    = PROJECT_DIR / "evaluation_summary.json"
PREDICTION_PATH = PROJECT_DIR / "prediction_results.csv"

with open(SUMMARY_PATH, "r") as f:
    EVALUATION_SUMMARY = json.load(f)

PREDICTION_RESULT = pd.read_csv(PREDICTION_PATH)

# ==========================================================
# MODEL REGISTRY
# ==========================================================

MODEL_REGISTRY = {}

for model_path in MODEL_FILES:
    filename     = model_path.stem.lower()
    architecture = "DenseNet121" if "densenet" in filename else "ResNet50"
    experiment   = filename.split("_")[-1].upper()
    MODEL_REGISTRY[model_path.name] = {
        "architecture": architecture,
        "experiment":   experiment,
        "path":         model_path
    }

# ==========================================================
# APPLICATION CONTEXT
# ==========================================================

APP_CONTEXT = {
    "config":                  CONFIG,
    "model_registry":          MODEL_REGISTRY,
    "evaluation_summary":      EVALUATION_SUMMARY,
    "prediction_result":       PREDICTION_RESULT,
    "current_model":           None,
    "current_model_name":      None,
    "current_image":           None,
    "current_prediction":      None,
    "current_heatmap":         None,
    "current_analysis":        None,
    "current_report":          None,
    "current_last_conv_layer": None,
    "current_result":          None,
}

# ==========================================================
# MODEL LOADER
# ==========================================================

def load_selected_model(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model : {model_name}")
    model_path = MODEL_REGISTRY[model_name]["path"]
    model = load_model(model_path)
    APP_CONTEXT["current_model"]      = model
    APP_CONTEXT["current_model_name"] = model_name
    return model

# ==========================================================
# IMAGE LOADER & PREPROCESSING
# ==========================================================

def load_and_preprocess_image(image_path):
    original_image = image.load_img(image_path, target_size=CONFIG["image_size"])
    image_array    = image.img_to_array(original_image).astype(np.float32)
    if CONFIG["normalization"] == "divide_255":
        image_array /= 255.0
    image_array = np.expand_dims(image_array, axis=0)
    APP_CONTEXT["current_image"] = str(image_path)
    return {
        "original_image": original_image,
        "image_array":    image_array,
        "image_path":     str(image_path)
    }

# ==========================================================
# CONFIDENCE LEVEL
# ==========================================================

def get_confidence_level(confidence):
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
    if APP_CONTEXT["current_model"] is None:
        raise RuntimeError("No model has been loaded.")
    model         = APP_CONTEXT["current_model"]
    image_array   = image_result["image_array"]
    probabilities = model.predict(image_array, verbose=0)
    if probabilities.shape[1] == 1:
        confidence_scoliosis = float(probabilities[0][0])
        confidence_normal    = 1.0 - confidence_scoliosis
        predicted_label      = int(confidence_scoliosis >= CONFIG["prediction_threshold"])
    else:
        confidence_normal    = float(probabilities[0][0])
        confidence_scoliosis = float(probabilities[0][1])
        predicted_label      = int(np.argmax(probabilities))
    predicted_class = CONFIG["positive_class"] if predicted_label == 1 else CONFIG["negative_class"]
    confidence      = max(confidence_normal, confidence_scoliosis)
    prediction_result = {
        "predicted_label":      predicted_label,
        "predicted_class":      predicted_class,
        "confidence":           confidence,
        "confidence_level":     get_confidence_level(confidence),
        "confidence_normal":    confidence_normal,
        "confidence_scoliosis": confidence_scoliosis,
        "probabilities":        probabilities
    }
    APP_CONTEXT["current_prediction"] = prediction_result
    return prediction_result

# ==========================================================
# AUTOMATIC GRAD-CAM LAYER DISCOVERY
# ==========================================================

def discover_gradcam_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Activation):
            try:
                if len(layer.output.shape) == 4:
                    APP_CONTEXT["current_last_conv_layer"] = layer.name
                    return layer
            except Exception:
                continue
    supported_layers = (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)
    for layer in reversed(model.layers):
        if isinstance(layer, supported_layers):
            APP_CONTEXT["current_last_conv_layer"] = layer.name
            return layer
    raise RuntimeError("No suitable Grad-CAM layer was found.")

# ==========================================================
# GENERATE GRAD-CAM HEATMAP
# ==========================================================

def generate_heatmap(image_result):
    model       = APP_CONTEXT["current_model"]
    layer_name  = APP_CONTEXT["current_last_conv_layer"]
    image_array = image_result["image_array"]
    grad_model  = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        if predictions.shape[-1] == 1:
            class_score = predictions[:, 0]
        else:
            class_index = tf.argmax(predictions[0])
            class_score = predictions[:, class_index]
    gradients        = tape.gradient(class_score, conv_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs     = conv_outputs[0]
    heatmap          = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap          = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if max_value > 0:
        heatmap /= max_value
    heatmap = heatmap.numpy()
    APP_CONTEXT["current_heatmap"] = heatmap
    return heatmap

# ==========================================================
# GENERATE OVERLAY
# ==========================================================

def generate_overlay(image_result, heatmap):
    original   = np.array(image_result["original_image"])
    heatmap_u8 = np.uint8(255 * heatmap)
    heatmap_u8 = cv2.resize(heatmap_u8, (original.shape[1], original.shape[0]))
    colored    = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    colored    = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay    = cv2.addWeighted(
        original, 1 - CONFIG["overlay_alpha"],
        colored,  CONFIG["overlay_alpha"],
        0
    )
    return overlay

# ==========================================================
# RUN GRAD-CAM
# ==========================================================

def run_gradcam(image_result):
    heatmap = generate_heatmap(image_result)
    overlay = generate_overlay(image_result, heatmap)
    result  = {"heatmap": heatmap, "overlay": overlay}
    APP_CONTEXT["current_heatmap"] = result
    return result

# ==========================================================
# HEATMAP STATISTICS
# ==========================================================

def compute_heatmap_statistics(heatmap):
    activation_mask = heatmap >= CONFIG["heatmap_threshold"]
    coverage        = (np.sum(activation_mask) / activation_mask.size) * 100
    if np.any(activation_mask):
        y_center, x_center = np.argwhere(activation_mask).mean(axis=0)
    else:
        y_center = heatmap.shape[0] / 2
        x_center = heatmap.shape[1] / 2
    return {
        "coverage":           coverage,
        "maximum_activation": float(np.max(heatmap)),
        "mean_activation":    float(np.mean(heatmap)),
        "center_x":           float(x_center),
        "center_y":           float(y_center)
    }

# ==========================================================
# DOMINANT REGION
# ==========================================================

def identify_region(stats):
    grid = 7 / 3
    col  = min(int(stats["center_x"] // grid), 2)
    row  = min(int(stats["center_y"] // grid), 2)
    region_names = {
        (0, 0): "Upper Left",    (0, 1): "Upper Center",           (0, 2): "Upper Right",
        (1, 0): "Middle Left",   (1, 1): "Central Thoracic Region", (1, 2): "Middle Right",
        (2, 0): "Lower Left",    (2, 1): "Lower Center",            (2, 2): "Lower Right"
    }
    return region_names[(row, col)]

# ==========================================================
# TECHNICAL INTERPRETATION
# ==========================================================

def technical_interpretation(stats):
    interpretation = []
    interpretation.append("Activation is broadly distributed." if stats["coverage"] >= 40 else "Activation is relatively localized.")
    interpretation.append("Moderate average activation intensity." if stats["mean_activation"] >= 0.30 else "Low average activation intensity.")
    interpretation.append("Strong activation peak detected." if stats["maximum_activation"] >= 0.90 else "Moderate activation peak detected.")
    return interpretation

# ==========================================================
# CLINICAL INTERPRETATION
# ==========================================================

def clinical_interpretation(prediction_result, region):
    interpretation = [
        f"Primary attention is concentrated around the {region.lower()}.",
        "The model appears to utilize multiple anatomical structures during prediction.",
    ]
    if prediction_result["predicted_class"] == CONFIG["positive_class"]:
        interpretation.append("The highlighted regions likely contain image features associated with scoliosis.")
    else:
        interpretation.append("The highlighted regions likely contain image features associated with normal spinal alignment.")
    return interpretation

# ==========================================================
# ANALYSIS ENGINE
# ==========================================================

def analyze_gradcam(gradcam_result, prediction_result):
    stats    = compute_heatmap_statistics(gradcam_result["heatmap"])
    region   = identify_region(stats)
    analysis = {
        **stats,
        "dominant_region":          region,
        "technical_interpretation": technical_interpretation(stats),
        "clinical_interpretation":  clinical_interpretation(prediction_result, region),
    }
    APP_CONTEXT["current_analysis"] = analysis
    return analysis

# ==========================================================
# REPORT ENGINE
# ==========================================================

def generate_report(prediction_result, analysis_result):
    structured_report = {
        "generated_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model":          APP_CONTEXT["current_model_name"],
        "prediction":     prediction_result,
        "analysis":       analysis_result
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
    for item in analysis_result["technical_interpretation"]:
        formatted_report += f"\n• {item}"
    formatted_report += "\n\nClinical Interpretation\n-------------------------\n"
    for item in analysis_result["clinical_interpretation"]:
        formatted_report += f"\n• {item}"
    report = {"structured": structured_report, "formatted": formatted_report}
    APP_CONTEXT["current_report"] = report
    return report

# ==========================================================
# UNIFIED DEPLOYMENT PIPELINE
# ==========================================================

def run_pipeline(image_path):
    image_result      = load_and_preprocess_image(image_path)
    prediction_result = predict_image(image_result)
    discover_gradcam_layer(APP_CONTEXT["current_model"])
    gradcam_result    = run_gradcam(image_result)
    analysis_result   = analyze_gradcam(gradcam_result, prediction_result)
    report_result     = generate_report(prediction_result, analysis_result)
    pipeline_result   = {
        "image":      image_result,
        "prediction": prediction_result,
        "gradcam":    gradcam_result,
        "analysis":   analysis_result,
        "report":     report_result
    }
    APP_CONTEXT["current_result"] = pipeline_result
    return pipeline_result
