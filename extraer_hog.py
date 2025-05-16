import numpy as np
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

# Cargar imágenes preprocesadas
imagenes = np.load("imagenes.npy")
print(f"Imágenes cargadas: {imagenes.shape}")

hog_features = []

# Parámetros HOG
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

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

    if idx % 1000 == 0:
        print(f"Procesadas {idx} imágenes...")

hog_features = np.array(hog_features)
print(f"Extracción HOG completada. Shape final: {hog_features.shape}")

np.save("hog_features.npy", hog_features)
print("Guardado exitosamente como hog_features.npy")

# Visualización HOG para ejemplo
ejemplo = 0
img = imagenes[ejemplo]

fd, hog_image = hog(
    img,
    orientations=orientations,
    pixels_per_cell=pixels_per_cell,
    cells_per_block=cells_per_block,
    block_norm='L2-Hys',
    visualize=True
)

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

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