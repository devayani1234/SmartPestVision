"""
advisor_agent.py
Usage:
  python advisor_agent.py --img "path/to/image.jpg" [--weather "temp,hum"]
Examples:
  python advisor_agent.py --img "dataset/val/Rice weevil/1.jpg"
  python advisor_agent.py --img "dataset/val/LARVA/larva1.jpg" --weather "30,75"
"""

import argparse
import json
from pathlib import Path
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import datetime
import csv

# CONFIG
BASE = Path(r"C:\Users\kdeva\Desktop\SmartPestVision\dataset")
MODEL_PATH = BASE / "pest_classifier_model.keras"   # or best_pest_model.keras
KB_PATH = Path(r"C:\Users\kdeva\Desktop\SmartPestVision\pest_knowledge.json")
LOG_CSV = BASE / "advisor_log.csv"

IMG_SIZE = (224, 224)

# Load model & knowledge base
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")
with open(KB_PATH, "r") as f:
    KB = json.load(f)

# Helper functions
def load_and_preprocess(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr

def predict_label(img_path):
    arr = load_and_preprocess(img_path)
    preds = model.predict(arr)
    idx = int(np.argmax(preds[0]))
    # classes are in alphabetical order of folder names because of flow_from_directory
    # We'll rebuild class order by listing train/ folder
    classes = sorted([d.name for d in (BASE/"train").iterdir() if d.is_dir()])
    return classes[idx], float(np.max(preds[0])), preds[0].tolist()

def risk_score(kb_entry, confidence, temperature=None, humidity=None):
    # base severity
    severity = kb_entry.get("severity_index", 0.5)
    score = severity * confidence
    # temperature and humidity modifiers (optional)
    if humidity is not None:
        # higher humidity -> higher risk (example)
        score *= (1 + (humidity - 50) / 200.0)  # small scale factor
    if temperature is not None:
        if temperature > 30:
            score *= 1.05
        elif temperature < 15:
            score *= 0.95
    return min(score, 1.0)

def recommend_actions(pest_key, confidence, temperature=None, humidity=None):
    kb = KB.get(pest_key, None)
    if not kb:
        return {"message":"No knowledge base entry found.", "actions":[], "urgency":0.5}
    score = risk_score(kb, confidence, temperature, humidity)
    urgency = "Low"
    if score > 0.8:
        urgency = "High"
    elif score > 0.5:
        urgency = "Medium"
    recs = {
        "pest": kb.get("common_name", pest_key),
        "damage": kb.get("damage", ""),
        "confidence": round(confidence*100, 2),
        "urgency": urgency,
        "recommended_actions": kb.get("treatments", [])[:3],
        "prevention": kb.get("prevention", [])[:3],
        "risk_score": round(score, 3)
    }
    # extra note based on environment
    notes = []
    if humidity is not None and humidity > 65:
        notes.append("High humidity detected -> drying and desiccation recommended.")
    if temperature is not None and temperature > 30:
        notes.append("Warm conditions may accelerate development.")
    if notes:
        recs["notes"] = notes
    return recs

def log_decision(img_path, pest, confidence, recs, temperature, humidity):
    header = ["timestamp","image","predicted_pest","confidence","urgency","risk_score","temperature","humidity","notes"]
    exists = LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="", encoding="utf8") as csvf:
        writer = csv.writer(csvf)
        if not exists:
            writer.writerow(header)
        writer.writerow([
            datetime.datetime.now().isoformat(),
            str(img_path),
            pest,
            confidence,
            recs.get("urgency"),
            recs.get("risk_score"),
            temperature if temperature is not None else "",
            humidity if humidity is not None else "",
            "; ".join(recs.get("notes", []))
        ])

# CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="Path to image to analyze")
    parser.add_argument("--weather", required=False, help="Optional: 'temp,hum' e.g. 30,70")
    args = parser.parse_args()

    img_path = Path(args.img)
    if not img_path.exists():
        print("Image not found:", img_path)
        return

    try:
        pest_key, conf, full_probs = predict_label(img_path)
    except Exception as e:
        print("Prediction failed:", e)
        return

    temperature = humidity = None
    if args.weather:
        try:
            t_s, h_s = args.weather.split(",")
            temperature = float(t_s); humidity = float(h_s)
        except Exception:
            print("Weather parse failed. expected format temp,hum e.g. 30,70")

    recs = recommend_actions(pest_key, conf, temperature, humidity)
    print("\n=== AGENT ADVISORY ===")
    print("Predicted:", recs["pest"], f"({recs['confidence']}% confidence)")
    print("Urgency:", recs["urgency"], "| risk score:", recs["risk_score"])
    print("\nDamage summary:")
    print(KB.get(pest_key, {}).get("damage", ""))
    print("\nRecommended actions:")
    for a in recs["recommended_actions"]:
        print(" -", a)
    print("\nPrevention tips:")
    for p in recs["prevention"]:
        print(" -", p)
    if recs.get("notes"):
        print("\nNotes:")
        for n in recs["notes"]:
            print(" -", n)

    # log
    log_decision(img_path, pest_key, recs["confidence"], recs, temperature, humidity)
    print("\nDecision logged to", LOG_CSV)

if __name__ == "__main__":
    main()
