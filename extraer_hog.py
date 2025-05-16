import numpy as np
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

# Cargar imágenes preprocesadas
imagenes = np.load("imagenes.npy")
print(f"📂 Imágenes cargadas: {imagenes.shape}")

# Lista para guardar los vectores de características HOG
hog_features = []

# Parámetros de HOG
orientations = 9         # Número de orientaciones de gradiente
pixels_per_cell = (8, 8) # Tamaño de celda en píxeles
cells_per_block = (2, 2) # Número de celdas por bloque

# Extraer características HOG para cada imagen
for idx, img in enumerate(imagenes):
    feature_vector = hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        visualize=False
    )
    hog_features.append(feature_vector)

    # Progreso
    if idx % 100 == 0:
        print(f"Procesadas {idx} imágenes...")

# Convertir lista a array de NumPy
hog_features = np.array(hog_features)
print(f"✅ Extracción HOG completada. Shape final: {hog_features.shape}")

# Guardar el array en un archivo .npy
np.save("hog_features.npy", hog_features)
print("💾 Guardado exitosamente como hog_features.npy")

# === VISUALIZACIÓN DE HOG PARA UNA IMAGEN DE EJEMPLO ===
ejemplo = 0  # Puedes cambiar el índice si quieres ver otra imagen
img = imagenes[ejemplo]

# Obtener el HOG y la visualización
fd, hog_image = hog(
    img,
    orientations=orientations,
    pixels_per_cell=pixels_per_cell,
    cells_per_block=cells_per_block,
    block_norm='L2-Hys',
    visualize=True
)

# Ajustar contraste para mejor visualización
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Mostrar imagen original y su HOG
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Imagen original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(hog_image_rescaled, cmap='gray')
plt.title("Visualización HOG")
plt.axis("off")

plt.tight_layout()
plt.show()