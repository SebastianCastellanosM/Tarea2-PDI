import numpy as np
from skimage.feature import local_binary_pattern
from skimage import exposure

# Parámetros de LBP
radius = 3                # Radio del vecindario circular
n_points = 8 * radius     # Número de puntos considerados
method = 'uniform'        # Método de patrón

# Cargar imágenes preprocesadas
imagenes = np.load("imagenes.npy")
print(f"Imágenes cargadas: {imagenes.shape}")

# Lista para guardar características
lbp_features = []

# Extraer LBP para cada imagen
for idx, img in enumerate(imagenes):
    # Calcular patrón LBP
    lbp = local_binary_pattern(img, n_points, radius, method)

    # Normalizar histograma
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    lbp_features.append(hist)

    if idx % 100 == 0:
        print(f"Procesadas {idx} imágenes...")

# Convertir a array de NumPy
lbp_features = np.array(lbp_features)
print(f"LBP extraído. Shape final: {lbp_features.shape}")

# Guardar
np.save("lbp_features.npy", lbp_features)
print("Guardado como lbp_features.npy")