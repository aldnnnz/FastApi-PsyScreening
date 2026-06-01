import re
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from deep_translator import GoogleTranslator

from app.custom_layers import AttentionLayer


# =========================================================
# LOAD ARTIFACTS
# =========================================================

tabular_model = tf.keras.models.load_model(
    "./models/tabular_model.keras"
)

text_model = tf.keras.models.load_model(
    "./models/text_model.keras",
    custom_objects={
        "AttentionLayer": AttentionLayer
    }
)

encoders = joblib.load("./models/encoders.pkl")
scaler = joblib.load("./models/scaler.pkl")

label_tab = joblib.load("./models/label_tab.pkl")
label_txt = joblib.load("./models/label_txt.pkl")


# =========================================================
# UTILITIES
# =========================================================

def clean_text(text: str) -> str:

    text = str(text)

    text = re.sub(
        r"[^\x00-\x7F]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def translate_to_english(text: str) -> str:

    try:
        return GoogleTranslator(
            source="auto",
            target="en"
        ).translate(text)

    except Exception:
        return text


# =========================================================
# MAIN PREDICTION
# =========================================================

def run_late_fusion_prediction(
    user_text: str,
    tabular_dict: dict,
    nlp_weight: float = 0.75,
    tabular_weight: float = 0.25,
):

    # ==========================
    # TEXT MODEL
    # ==========================

    translated_text = translate_to_english(user_text)

    text_clean = clean_text(
        translated_text
    )

    prob_txt = text_model.predict(
        np.array([text_clean]),
        verbose=0
    )[0]

    nlp_labels = label_txt.classes_.tolist()

    nlp_probs = {
        str(label): float(prob)
        for label, prob in zip(
            nlp_labels,
            prob_txt
        )
    }

    primary_nlp_label = nlp_labels[
        np.argmax(prob_txt)
    ]

    # ==========================
    # NLP RISK
    # ==========================

    risk_keywords = [
        "suicidal",
        "depression",
        "anxiety"
    ]

    nlp_risk = 0

    for label, prob in nlp_probs.items():

        if str(label).lower() in risk_keywords:
            nlp_risk += prob

    nlp_risk = float(
        np.clip(
            nlp_risk,
            0,
            1
        )
    )

    # ==========================
    # TABULAR MODEL
    # ==========================

    row = []

    for col in encoders.keys():

        value = str(
            tabular_dict.get(
                col,
                ""
            )
        )

        try:

            encoded = encoders[col].transform(
                [value]
            )[0]

        except Exception:

            encoded = 0

        row.append(encoded)

    row = pd.DataFrame(
        [row],
        columns=list(encoders.keys())
    )

    row = scaler.transform(row)

    prob_tab = tabular_model.predict(
        row,
        verbose=0
    )[0]

    treatment_confidence = float(
        prob_tab[1]
    )

    # ==========================
    # FUSION
    # ==========================

    fused_risk = (
        nlp_weight * nlp_risk
        + tabular_weight * treatment_confidence
    )

    fused_risk = float(
        np.clip(
            fused_risk,
            0,
            1
        )
    )

    # ==========================
    # SEVERITY
    # ==========================

    if fused_risk >= 0.85:
        severity = "CRITICAL"

    elif fused_risk >= 0.65:
        severity = "HIGH"

    elif fused_risk >= 0.40:
        severity = "MODERATE"

    else:
        severity = "LOW"

    # ==========================
    # RECOMMENDATIONS
    # ==========================

    recommendations = []

    if fused_risk >= 0.80:

        recommendations.append(
            "Segera konsultasi dengan profesional kesehatan mental."
        )

    elif fused_risk >= 0.60:

        recommendations.append(
            "Disarankan mencari dukungan profesional."
        )

    elif fused_risk >= 0.40:

        recommendations.append(
            "Lakukan monitoring kondisi psikologis secara berkala."
        )

    else:

        recommendations.append(
            "Kondisi relatif stabil."
        )

    if treatment_confidence >= 0.70:

        recommendations.append(
            "Data survei menunjukkan kemungkinan membutuhkan treatment."
        )

    elif treatment_confidence <= 0.30:

        recommendations.append(
            "Data survei menunjukkan risiko treatment rendah."
        )

    return {
        "original_text": user_text,
        "translated_text": translated_text,
        "primary_label": str(primary_nlp_label),
        "nlp_probabilities": nlp_probs,
        "nlp_risk": round(nlp_risk, 4),
        "treatment_confidence": round(
            treatment_confidence,
            4
        ),
        "fused_risk": round(
            fused_risk,
            4
        ),
        "severity": severity,
        "recommendations": recommendations,
    }