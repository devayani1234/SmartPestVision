import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt

# ===============================================================
# ✅ Load the trained model
# ===============================================================
MODEL_PATH = r"C:\Users\kdeva\Desktop\SmartPestVision\dataset\pest_classifier_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# ===============================================================
# ✅ Path to test image
# ===============================================================
IMG_PATH = r"C:\Users\kdeva\Desktop\Green gram20251004160226857.jpg"
# ===============================================================
# ✅ Path to dataset (where your classes are)
# ===============================================================
DATASET_PATH = r"C:\Users\kdeva\Desktop\SmartPestVision\dataset\train"

# Collect valid class folders (ignore hidden files)
class_labels = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
print("Class Labels found:", class_labels)

# ===============================================================
# ✅ Map dataset folder names → readable insect names
# ===============================================================
label_map = {
    "Pulse beetle-20251002T130229Z-1-001": "Pulse Beetle (Callosobruchus chinensis)",
    "Rice weevil-20251002T130145Z-1-001": "Rice Weevil (Sitophilus oryzae)",
    "Storage pests in grains-20251002T130302Z-1-001": "Common Grain Storage Pests"
}

# ===============================================================
# ✅ Preprocess image
# ===============================================================
img = image.load_img(IMG_PATH, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # normalize

# ===============================================================
# ✅ Predict
# ===============================================================
predictions = model.predict(img_array)
predicted_index = int(np.argmax(predictions[0]))
confidence = round(100 * np.max(predictions[0]), 2)

# ✅ Get clean readable label
if predicted_index < len(class_labels):
    folder_label = class_labels[predicted_index]
    insect_name = label_map.get(folder_label, folder_label)
else:
    insect_name = f"Unknown Pest (index {predicted_index})"

print(f"\n✅ Predicted Pest: {insect_name} ({confidence}% confidence)")

# ===============================================================
# ✅ Display image with prediction
# ===============================================================
plt.figure(figsize=(6, 6))
plt.imshow(image.load_img(IMG_PATH))
plt.axis("off")
plt.title(f"Pest: {insect_name}\nConfidence: {confidence}%", fontsize=13, color="green")
plt.show()
