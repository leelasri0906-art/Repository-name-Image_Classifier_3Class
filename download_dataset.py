import tensorflow as tf
from PIL import Image
import os

# Load CIFAR-10 dataset
(x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()

# CIFAR-10 classes
class_names = {
    1: "car",
    3: "cat",
    5: "dog"
}

# Create folders
for folder in class_names.values():
    os.makedirs(os.path.join("dataset", folder), exist_ok=True)

# Number of images per class
images_per_class = 1000

# Counters
counters = {
    "car": 0,
    "cat": 0,
    "dog": 0
}

# Save images
for image, label in zip(x_train, y_train):

    class_id = int(label[0])

    if class_id not in class_names:
        continue

    class_name = class_names[class_id]

    if counters[class_name] >= images_per_class:
        continue

    filename = os.path.join(
        "dataset",
        class_name,
        f"{class_name}_{counters[class_name] + 1}.jpg"
    )

    Image.fromarray(image).save(filename)

    counters[class_name] += 1

    print(
        f"{class_name}: "
        f"{counters[class_name]}/{images_per_class}"
    )

    if all(count >= images_per_class for count in counters.values()):
        break

print("\n==============================")
print("DATASET CREATED SUCCESSFULLY!")
print("==============================")

print(f"Car images : {counters['car']}")
print(f"Cat images : {counters['cat']}")
print(f"Dog images : {counters['dog']}")