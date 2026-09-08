import tensorflow as tf
from tensorflow.keras import layers, models
import os

IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32
EPOCHS = 15

DATASET_DIR = "dataset"

print("Loading dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names

print("Classes found:", class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

model = models.Sequential([
    layers.Input(shape=(180, 180, 3)),

    data_augmentation,

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nStarting training...\n")

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

os.makedirs("model", exist_ok=True)

model.save("model/classifier.keras")

with open("model/classes.txt", "w") as file:
    for class_name in class_names:
        file.write(class_name + "\n")

print("\n================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("================================")
print("Model saved: model/classifier.keras")
print("Classes:", class_names)