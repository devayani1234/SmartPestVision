import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.utils.class_weight import compute_class_weight
import datetime

# =========================================================
# ✅ CONFIGURATION
# =========================================================
BASE_DIR = Path(r"C:\Users\kdeva\Desktop\SmartPestVision\dataset")
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 10
LEARNING_RATE_STAGE1 = 1e-4
LEARNING_RATE_STAGE2 = 1e-5

BEST_MODEL_PATH = BASE_DIR / "best_pest_model.keras"
FINAL_MODEL_PATH = BASE_DIR / "pest_classifier_model.keras"

# =========================================================
# ✅ DATA PREPROCESSING & AUGMENTATION
# =========================================================
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_gen = ImageDataGenerator(rescale=1.0 / 255)

train_ds = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_ds = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# =========================================================
# ✅ COMPUTE CLASS WEIGHTS (for imbalance handling)
# =========================================================
classes = list(train_ds.class_indices.keys())
labels = train_ds.classes
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

print("\nClass Weights:", class_weight_dict)

# =========================================================
# ✅ MODEL CREATION (Transfer Learning)
# =========================================================
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # freeze base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
predictions = Dense(len(classes), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_STAGE1),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# =========================================================
# ✅ CALLBACKS
# =========================================================
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True, monitor="val_accuracy"),
    TensorBoard(log_dir=log_dir, histogram_freq=1)
]

# =========================================================
# ✅ TRAINING — STAGE 1
# =========================================================
print("\n🔹 Stage 1: Training top layers (base frozen)...\n")
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# =========================================================
# ✅ FINE-TUNING — STAGE 2
# =========================================================
print("\n🔹 Stage 2: Fine-tuning deeper layers...\n")
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_STAGE2),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# =========================================================
# ✅ SAVE FINAL MODEL
# =========================================================
model.save(FINAL_MODEL_PATH)
print("\n✅ Model saved at:", FINAL_MODEL_PATH)

# =========================================================
# ✅ PLOT TRAINING HISTORY
# =========================================================
def plot_training(hist1, hist2):
    acc = hist1.history["accuracy"] + hist2.history["accuracy"]
    val_acc = hist1.history["val_accuracy"] + hist2.history["val_accuracy"]
    loss = hist1.history["loss"] + hist2.history["loss"]
    val_loss = hist1.history["val_loss"] + hist2.history["val_loss"]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Model Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(BASE_DIR / "training_plot.png")
    plt.show()

plot_training(history1, history2)
print("\n✅ Training complete. Plot saved as 'training_plot.png'")
