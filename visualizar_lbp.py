import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage import exposure

# Parámetros de LBP (deben coincidir con los usados para extraer)
radius = 3
n_points = 8 * radius
method = 'uniform'


imagenes = np.load("imagenes.npy")
idx = 0  # Cambia el índice para ver otra imagen

img = imagenes[idx]

# Calcular LBP
lbp = local_binary_pattern(img, n_points, radius, method)

# Calcular histograma
n_bins = int(lbp.max() + 1)
hist, bins = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

# Visualizar
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Imagen original
ax1.imshow(img, cmap='gray')
ax1.set_title('Imagen original')
ax1.axis('off')

# Imagen LBP
ax2.imshow(lbp, cmap='gray')
ax2.set_title('Patrón LBP')
ax2.axis('off')


ax3.bar(bins[:-1], hist, width=0.5, edgecolor='black')
ax3.set_title('Histograma de LBP')
ax3.set_xlabel('Valor LBP')
ax3.set_ylabel('Frecuencia')

plt.tight_layout()
plt.show()