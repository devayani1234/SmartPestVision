# streamlit_advisor_ultra.py
from pathlib import Path
import json, csv, datetime
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ---------- CONFIG ----------
BASE = Path(r"C:\Users\kdeva\Desktop\SmartPestVision")
DATASET = BASE / "dataset"
MODEL_PATH = DATASET / "pest_classifier_model.keras"   # or best_pest_model.keras
KB_PATH = BASE / "pest_knowledge.json"
LOG_CSV = DATASET / "advisor_log.csv"
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="SmartPestVision — Agentic Advisor",
    page_icon="🪲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- THEME / CSS ----------
st.markdown(
    """
    <style>
    :root{
      --bg:#0f172a;           /* header bg */
      --fg:#f8fafc;           /* header fg */
      --card:#ffffff;         /* card bg */
      --muted:#6b7280;        /* muted text */
      --shadow:0 8px 24px rgba(2,6,23,0.08);
    }
    .hero{
      background: linear-gradient(135deg,#0ea5e9 0%,#6366f1 50%,#8b5cf6 100%);
      color: var(--fg);
      padding: 22px 26px;
      border-radius: 16px;
      box-shadow: var(--shadow);
      margin-bottom: 18px;
    }
    .card{
      background: var(--card);
      border-radius: 14px;
      padding: 18px 18px;
      box-shadow: var(--shadow);
      margin-bottom: 16px;
      border: 1px solid rgba(2,6,23,0.045);
    }
    .muted{color:var(--muted);font-size:13px}
    .badge{display:inline-block;padding:6px 10px;border-radius:999px;font-weight:600;font-size:13px}
    .high{background:#fee2e2;color:#b91c1c;border:1px solid #fecaca}
    .med{ background:#fef3c7;color:#b45309;border:1px solid #fde68a}
    .low{ background:#dcfce7;color:#166534;border:1px solid #bbf7d0}
    .riskbar{height:12px;background:#f1f5f9;border-radius:999px;position:relative;overflow:hidden}
    .riskfill{height:100%;border-radius:999px;background:linear-gradient(90deg,#22c55e,#f59e0b,#ef4444)}
    .mono{font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace;}
    .small{font-size:13px}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
      <h2 style="margin:0">🧠 SmartPestVision — Agentic Advisor</h2>
      <div class="muted">Detect • Assess Risk • Recommend Actions • Log</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- LOAD MODEL & KB (cached) ----------
@st.cache_resource
def load_model_kb():
    model = tf.keras.models.load_model(str(MODEL_PATH))
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)
    classes = sorted([d.name for d in (DATASET / "train").iterdir() if d.is_dir()])
    return model, kb, classes

with st.spinner("Loading model & knowledge base…"):
    model, KB, CLASS_LIST = load_model_kb()
st.success("Model & knowledge base ready ✅")

# ---------- HELPERS ----------
def preprocess_img(p: Path):
    img = image.load_img(p, target_size=IMG_SIZE)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr

def predict(p: Path):
    arr = preprocess_img(p)
    preds = model.predict(arr, verbose=0)
    idx = int(np.argmax(preds[0]))
    conf = float(np.max(preds[0]))
    return CLASS_LIST[idx], conf, preds[0]

def risk_score(kb_entry, confidence, temperature=None, humidity=None):
    severity = kb_entry.get("severity_index", 0.5)
    score = severity * confidence
    if humidity is not None: score *= (1 + (humidity - 50) / 200.0)
    if temperature is not None:
        if temperature > 30: score *= 1.05
        elif temperature < 15: score *= 0.95
    return float(np.clip(score, 0, 1))

def urgency_badge(score: float) -> str:
    if score > 0.8:  return '<span class="badge high">🔴 High</span>'
    if score > 0.5:  return '<span class="badge med">🟡 Medium</span>'
    return '<span class="badge low">🟢 Low</span>'

def recommend(pest_key, conf, temperature=None, humidity=None):
    kb = KB.get(pest_key, {})
    score = risk_score(kb, conf, temperature, humidity)
    return {
        "pest": kb.get("common_name", pest_key),
        "damage": kb.get("damage",""),
        "confidence": round(conf*100, 2),
        "score": round(score, 3),
        "urgency_html": urgency_badge(score),
        "actions": kb.get("treatments", [])[:5],
        "prevention": kb.get("prevention", [])[:5],
        "notes": (
            (["High humidity → Drying recommended."] if (humidity and humidity>65) else []) +
            (["Warm conditions → Faster development likely."] if (temperature and temperature>30) else [])
        )
    }

def log_decision(img_path, pest_key, conf, recs, temperature, humidity):
    header = ["timestamp","image","predicted_pest","confidence(%)","risk_score","temperature","humidity","notes"]
    exists = LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists: w.writerow(header)
        w.writerow([
            datetime.datetime.now().isoformat(timespec="seconds"),
            str(img_path),
            pest_key,
            recs["confidence"],
            recs["score"],
            temperature if temperature is not None else "",
            humidity if humidity is not None else "",
            "; ".join(recs["notes"])
        ])

def risk_bar_html(score: float) -> str:
    pct = int(score * 100)
    return f"""
    <div class="riskbar"><div class="riskfill" style="width:{pct}%"></div></div>
    <div class="muted small">Risk score: <span class="mono">{pct}%</span></div>
    """

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f9fafb 0%, #eef2ff 100%);
            padding-top: 1rem;
        }
        .stButton > button {
            background: linear-gradient(90deg,#4f46e5,#06b6d4);
            color: white;
            font-weight: 600;
            border-radius: 8px;
            border: none;
            transition: all .2s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(79,70,229,0.3);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.header("⚙️ Input")
    uploaded = st.file_uploader("Upload image (JPG/PNG)", type=["jpg","jpeg","png"])
    temp = st.number_input("🌡️ Temperature (°C)", value=25.0, step=0.5)
    hum  = st.number_input("💧 Humidity (%)", value=50.0, step=1.0)
    run  = st.button("🔍 Analyze & Recommend", use_container_width=True)



# ---------- MAIN LAYOUT ----------
left, right = st.columns([1.05, 1.2], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Image")
    if uploaded is not None:
        tmp_path = DATASET / "temp_upload.jpg"
        with open(tmp_path, "wb") as f: f.write(uploaded.getbuffer())
        st.image(str(tmp_path), use_container_width=True, caption="Uploaded image")

    else:
        st.info("Upload a grain/pest image from the sidebar to begin.")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    result_container = st.container()
    with st.markdown('<div class="card">', unsafe_allow_html=True):
        st.subheader("Activity Log")
        if LOG_CSV.exists():
            df = pd.read_csv(LOG_CSV)
            st.dataframe(df.tail(8), use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download full log (CSV)",
                df.to_csv(index=False).encode("utf-8"),
                file_name="advisor_log.csv",
                use_container_width=True
            )
        else:
            st.info("No log yet. Run an analysis to create entries.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- ACTION ----------
if run:
    if uploaded is None:
        st.warning("Please upload an image first.")
    else:
        with st.spinner("Thinking like an agent…"):
            pest_key, conf, probs = predict(tmp_path)
            recs = recommend(pest_key, conf, temperature=temp, humidity=hum)
            log_decision(tmp_path, pest_key, conf, recs, temp, hum)

        # --- RESULT CARD ---
        with result_container:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Result")
            cols = st.columns([1, 1, 1.2])
            with cols[0]:
                st.markdown("**Predicted pest**")
                st.markdown(f"{recs['pest']}  \n<span class='muted small'>Class:</span> **{pest_key}**", unsafe_allow_html=True)
            with cols[1]:
                st.markdown("**Confidence**")
                st.metric(label="", value=f"{recs['confidence']}%")
            with cols[2]:
                st.markdown("**Urgency**")
                st.markdown(recs["urgency_html"], unsafe_allow_html=True)
                st.markdown(risk_bar_html(recs["score"]), unsafe_allow_html=True)

            st.markdown("---")
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown("### Recommended actions")
                for a in recs["actions"]:
                    st.markdown(f"- {a}")
            with c2:
                st.markdown("### Prevention tips")
                for p in recs["prevention"]:
                    st.markdown(f"- {p}")

            if recs["notes"]:
                st.markdown("---")
                st.markdown("**Notes**")
                for n in recs["notes"]:
                    st.info(n)

            st.markdown("</div>", unsafe_allow_html=True)

        # --- TOP-K PROBS MINI CHART ---
        probs_df = pd.DataFrame({
            "Class": CLASS_LIST,
            "Probability": [float(x) for x in probs]
        }).sort_values("Probability", ascending=False).head(5).reset_index(drop=True)
        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Top classes")
            st.bar_chart(probs_df.set_index("Class"))
            st.markdown("</div>", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("<div class='muted'>© SmartPestVision • Agentic Advisor</div>", unsafe_allow_html=True)



