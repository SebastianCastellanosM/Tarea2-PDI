import numpy as np
import pandas as pd

DATASET_PATH = 'fer2013.csv'  # Ruta al archivo CSV
IMG_SIZE = 48  # Tamaño original FER2013

# Función opcional para redimensionar y agregar padding (descomentarla si la usas)
# import cv2
# def resize_and_pad(img, size):
#     h, w = img.shape
#     scale = size / max(h, w)
#     new_w, new_h = int(w * scale), int(h * scale)
#     img_resized = cv2.resize(img, (new_w, new_h))
#     top = (size - new_h) // 2
#     bottom = size - new_h - top
#     left = (size - new_w) // 2
#     right = size - new_w - left
#     padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
#     return padded

# Leer CSV
data = pd.read_csv(DATASET_PATH)

# Filtrar filas Training y PublicTest para ejemplo (opcional)
data = data[data['Usage'].isin(['Training', 'PublicTest'])]

images = []
labels = []

for idx, row in data.iterrows():
    pixels = row['pixels']
    emotion = row['emotion']

    # Convertir string de pixeles a numpy array
    img_array = np.array([int(p) for p in pixels.split()]).reshape(IMG_SIZE, IMG_SIZE).astype(np.uint8)

    # Si quieres redimensionar y agregar padding, descomenta y usa:
    # img_array = resize_and_pad(img_array, 128)

    images.append(img_array)
    labels.append(emotion)

images_np = np.array(images)
labels_np = np.array(labels)

print(f"Imágenes cargadas: {images_np.shape}")
print(f"Etiquetas cargadas: {labels_np.shape}")

# Guardar para usar luego
np.save("imagenes.npy", images_np)
np.save("etiquetas.npy", labels_np)
print("Datos guardados en imagenes.npy y etiquetas.npy")