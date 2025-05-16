import os
import cv2
import numpy as np

DATASET_PATH = 'data/101_ObjectCategories'

# Clases a usar
CLASSES = ['airplanes', 'motorbikes']

# Tamaño objetivo
IMG_SIZE = 128

# Función para redimensionar manteniendo la proporción y agregando padding
def resize_and_pad(img, size):
    h, w = img.shape
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h))

    # Crear imagen cuadrada con padding
    top = (size - new_h) // 2
    bottom = size - new_h - top
    left = (size - new_w) // 2
    right = size - new_w - left

    padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded

# Listas para imágenes y etiquetas
images = []
labels = []

# Cargar imágenes
for label_idx, class_name in enumerate(CLASSES):
    class_path = os.path.join(DATASET_PATH, class_name)
    for filename in os.listdir(class_path):
        file_path = os.path.join(class_path, filename)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img_processed = resize_and_pad(img, IMG_SIZE)
        images.append(img_processed)
        labels.append(label_idx)

print(f"Imágenes cargadas: {len(images)}")
print(f"Clases: {CLASSES}")

# Convertir a arrays
images_np = np.array(images)
labels_np = np.array(labels)

# Guardar
np.save("imagenes.npy", images_np)
np.save("etiquetas.npy", labels_np)

print(f"imagenes.npy shape: {images_np.shape}")
print(f"etiquetas.npy shape: {labels_np.shape}")
