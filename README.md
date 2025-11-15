🧠 SmartPestVision — Agentic Advisor

SmartPestVision is an AI-powered pest detection and advisory system that uses deep learning and agentic reasoning to identify storage pests from grain images and suggest optimal treatment and prevention strategies.

The system combines:
A trained CNN model (TensorFlow/Keras) for pest image classification.
An Agentic AI layer that interprets model output using a structured knowledge base (pest_knowledge.json) to generate context-aware recommendations.
A Streamlit web interface for easy interaction, image upload, and live advisory results.


🌾 Key Features
Upload pest or grain images to get instant pest identification.
Adjust temperature and humidity to get dynamic, real-time recommendations.
Integrated knowledge base to provide risk level, urgency badges, and preventive actions.
Activity Log automatically records each detection and recommendation in CSV format.
Simple, modern Streamlit UI with gradient theme and agentic decision logic.


⚙️ Folder Structure
SmartPestVision/
│
├── streamlit_advisor_ultra.py      # Main Streamlit UI (Agentic AI version)
├── pest_classifier_model.keras     # Trained CNN model
├── pest_knowledge.json             # Expert knowledge base (rules & treatments)
├── classes.json                    # List of pest classes
├── requirements.txt                # Python dependencies
├── evaluate_model.py               # Evaluation/testing script
├── sample_image.jpg                # Example pest image (optional)
├── dataset/                        # (Optional) Contains train/val data and backups
└── logs/                           # (Auto-created) Stores detection logs (CSV)



🧩 Setup & Run (Local)

Create and activate a virtual environment:
python -m venv .venv
.venv\Scripts\Activate.ps1       # PowerShell (Windows)


Install all required libraries:
pip install -r requirements.txt


Launch the web app:
streamlit run streamlit_advisor_ultra.py


Open the link shown in terminal (usually http://localhost:8501/) in your browser.
Upload an image → enter temperature & humidity → click “Analyze & Recommend.”

🧪 Test the Model (Google Colab)
If your faculty prefers Colab verification:
Upload the following files to your Drive folder:
pest_classifier_model.keras
classes.json
sample_image.jpg


Open a new Colab notebook and run:

from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/SmartPestVision
!pip install -q tensorflow pillow

import tensorflow as tf, numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("pest_classifier_model.keras")
img = image.load_img("sample_image.jpg", target_size=(224,224))
arr = np.expand_dims(image.img_to_array(img)/255.0, axis=0)
preds = model.predict(arr)
print("Predicted class index:", preds.argmax(), "Confidence:", preds.max())


This verifies that the trained model loads correctly and performs inference.

📊 Optional Scripts
Script	Purpose
train_pest_model_with_weights.py	Retrain or fine-tune model using new dataset.
evaluate_model.py	Evaluate accuracy and generate performance metrics.
advisor_agent.py	Command-line version of the advisor (no UI).

🧠 Agentic AI Integration
The Agentic layer in SmartPestVision mimics expert reasoning:
Interprets model confidence + environmental context (temp/humidity).
Cross-references the pest_knowledge.json file for pest severity and control measures.
Calculates a risk score and classifies it as High / Medium / Low urgency.
Provides actionable insights with prevention and treatment recommendations.