import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# ===============================================================
# ✅ CONFIGURATION
# ===============================================================
BASE_DIR = r"C:\Users\kdeva\Desktop\SmartPestVision\dataset"  # parent folder containing 'train' and 'val'
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

# ===============================================================
# ✅ DATA AUGMENTATION
# ===============================================================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

print("\n✅ Classes found:", train_generator.class_indices)

# ===============================================================
# ✅ MODEL CREATION — Transfer Learning with MobileNetV2
# ===============================================================
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # freeze base layers initially

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
predictions = Dense(train_generator.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ===============================================================
# ✅ COMPILE MODEL
# ===============================================================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ===============================================================
# ✅ CALLBACKS
# ===============================================================
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ModelCheckpoint("best_pest_model.keras", save_best_only=True)
]

# ===============================================================
# ✅ TRAIN MODEL (Stage 1 - Freeze base layers)
# ===============================================================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ===============================================================
# ✅ FINE-TUNE (Stage 2 - Unfreeze some base layers)
# ===============================================================
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False  # keep early layers frozen

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n🔧 Fine-tuning model...")
fine_tune_history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=callbacks
)

# ===============================================================
# ✅ SAVE FINAL MODEL
# ===============================================================
model.save(os.path.join(BASE_DIR, "pest_classifier_model.keras"))
print("\n✅ Model saved as pest_classifier_model.keras")

# ===============================================================
# ✅ EVALUATION
# ===============================================================
print("\n🔍 Evaluating model on validation data...")
val_generator.reset()
Y_pred = model.predict(val_generator)
y_pred = np.argmax(Y_pred, axis=1)

print("\n📊 Classification Report:\n")
print(classification_report(val_generator.classes, y_pred, target_names=list(val_generator.class_indices.keys())))

print("\n🧩 Confusion Matrix:\n", confusion_matrix(val_generator.classes, y_pred))

# ===============================================================
# ✅ PLOT TRAINING HISTORY
# ===============================================================
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc (Stage 1)')
plt.plot(history.history['val_accuracy'], label='Val Acc (Stage 1)')
if 'accuracy' in fine_tune_history.history:
    plt.plot(fine_tune_history.history['accuracy'], label='Train Acc (Fine-tune)')
    plt.plot(fine_tune_history.history['val_accuracy'], label='Val Acc (Fine-tune)')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss (Stage 1)')
plt.plot(history.history['val_loss'], label='Val Loss (Stage 1)')
if 'loss' in fine_tune_history.history:
    plt.plot(fine_tune_history.history['loss'], label='Train Loss (Fine-tune)')
    plt.plot(fine_tune_history.history['val_loss'], label='Val Loss (Fine-tune)')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
