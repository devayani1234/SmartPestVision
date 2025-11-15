import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import json

# ===============================================================
# ✅ CONFIGURATION
# ===============================================================
BASE_DIR = Path(r"C:\Users\kdeva\Desktop\SmartPestVision\dataset")
MODEL_PATH = BASE_DIR / "pest_classifier_model.keras"
DATASET_PATH = BASE_DIR / "train"

# ===============================================================
# ✅ LOAD MODEL
# ===============================================================
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ===============================================================
# ✅ LOAD CLASS LABELS
# ===============================================================
class_labels = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(DATASET_PATH / d)])
print("\n📂 Classes found:")
for i, cls in enumerate(class_labels):
    print(f"{i}: {cls}")

# ===============================================================
# ✅ LABEL MAP FOR READABLE OUTPUT
# ===============================================================
label_map = {
    "Pulse beetle": "Pulse Beetle (Callosobruchus chinensis)",
    "Rice weevil": "Rice Weevil (Sitophilus oryzae)",
    "Storage pests in grains": "Common Grain Storage Pests",
    "EGG": "Pest Egg Stage",
    "LARVA": "Pest Larva Stage",
    "Bore holes": "Grain Bore Holes",
    "Grain - Rice weevil": "Rice Weevil Damage on Grains"
}

# ===============================================================
# ✅ FUNCTION: PREDICT A SINGLE IMAGE
# ===============================================================
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = round(100 * np.max(predictions[0]), 2)

    folder_label = class_labels[predicted_index]
    insect_name = label_map.get(folder_label, folder_label)

    print(f"\n✅ Predicted: {insect_name} ({confidence}% confidence)")

    plt.imshow(image.load_img(img_path))
    plt.axis("off")
    plt.title(f"Pest: {insect_name}\nConfidence: {confidence}%", color="green", fontsize=12)
    plt.show()

# ===============================================================
# ✅ TEST
# ===============================================================
test_image = input("\nEnter path of an image to test: ").strip('"')
predict_image(test_image)
