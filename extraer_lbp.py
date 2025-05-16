import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

# Cargar imágenes preprocesadas
imagenes = np.load("imagenes.npy")
print(f"Imágenes cargadas: {imagenes.shape}")

# Parámetros LBP
radius = 1            # Radio de vecindad
n_points = 8 * radius # Número de puntos vecinos
METHOD = 'uniform'    # Método de cálculo

lbp_features = []

for idx, img in enumerate(imagenes):
    lbp = local_binary_pattern(img, n_points, radius, METHOD)
    # Histograma LBP como vector de características
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, n_points + 3),
                             range=(0, n_points + 2))
    # Normalizar histograma
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    lbp_features.append(hist)

    if idx % 1000 == 0:
        print(f"Procesadas {idx} imágenes...")

lbp_features = np.array(lbp_features)
print(f"Extracción LBP completada. Shape final: {lbp_features.shape}")

np.save("lbp_features.npy", lbp_features)
print("Guardado exitosamente como lbp_features.npy")

# Visualización LBP para una imagen de ejemplo
ejemplo = 0
img = imagenes[ejemplo]

lbp_img = local_binary_pattern(img, n_points, radius, METHOD)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Imagen original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(lbp_img, cmap='gray')
plt.title("Visualización LBP")
plt.axis("off")

plt.tight_layout()
plt.show()