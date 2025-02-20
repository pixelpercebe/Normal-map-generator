import numpy as np
import cv2
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt


# Función para aplicar la transformada de Fourier a un canal de color
def apply_fourier(channel):
	# Aplicar la Transformada de Fourier en 2D
	F_transform = fft2(channel)
	# Desplazar la transformada para centrar las frecuencias bajas
	F_transform_shifted = fftshift(F_transform)
	return F_transform_shifted


# Función para aplicar un filtro con suavizado
def apply_frequency_filter(F_transform_shifted, cutoff_radius):
	# Filtro circular: paso bajo con suavizado
	rows, cols = F_transform_shifted.shape
	center_row, center_col = rows // 2, cols // 2

	# Crear una máscara gaussiana suave
	x, y = np.ogrid[:rows, :cols]
	distance_from_center = np.sqrt((x - center_row) ** 2 + (y - center_col) ** 2)

	# Máscara que va suavizando la transición
	mask = np.exp(-(distance_from_center ** 2 / (2 * (cutoff_radius ** 2))))

	# Aplicar la máscara
	F_transform_filtered = F_transform_shifted * mask
	return F_transform_filtered


# Función para reconstruir el canal desde la transformada filtrada
def reconstruct_from_fourier(F_transform_filtered):
	# Deshacer el desplazamiento y aplicar la Transformada Inversa de Fourier
	F_transform_shifted_back = ifftshift(F_transform_filtered)
	reconstructed_channel = ifft2(F_transform_shifted_back)
	# Tomar solo la parte real de la imagen reconstruida
	return np.abs(reconstructed_channel)


# Cargar la imagen a color
image = cv2.imread('toy.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Separar los canales de color (R, G, B)
r_channel, g_channel, b_channel = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]

# Aplicar la Transformada de Fourier a cada canal
F_r = apply_fourier(r_channel)
F_g = apply_fourier(g_channel)
F_b = apply_fourier(b_channel)

# Aplicar un filtro de frecuencias en cada canal con un radio ajustable
cutoff_radius = 60  # Ajusta este valor para controlar la cantidad de frecuencias
F_r_filtered = apply_frequency_filter(F_r, cutoff_radius)
F_g_filtered = apply_frequency_filter(F_g, cutoff_radius)
F_b_filtered = apply_frequency_filter(F_b, cutoff_radius)

# Reconstruir la imagen desde las frecuencias filtradas
r_reconstructed = reconstruct_from_fourier(F_r_filtered)
g_reconstructed = reconstruct_from_fourier(F_g_filtered)
b_reconstructed = reconstruct_from_fourier(F_b_filtered)

# Combinar los canales reconstruidos
image_reconstructed = np.stack((r_reconstructed, g_reconstructed, b_reconstructed), axis=2)

# Mostrar la imagen original y la reconstruida
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Imagen Original')
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Imagen Reconstruida con Menos Pérdida')
plt.imshow(np.uint8(np.clip(image_reconstructed, 0, 255)))
plt.axis('off')

plt.show()

# Guardar los coeficientes de Fourier para cada canal (puedes almacenarlos en un archivo)
np.save('F_r_coefficients.npy', F_r)
np.save('F_g_coefficients.npy', F_g)
np.save('F_b_coefficients.npy', F_b)

# Para cargar y reconstruir a partir de los coeficientes guardados
F_r_loaded = np.load('F_r_coefficients.npy')
F_g_loaded = np.load('F_g_coefficients.npy')
F_b_loaded = np.load('F_b_coefficients.npy')

# Reconstruir la imagen a partir de los coeficientes cargados
r_reconstructed_loaded = reconstruct_from_fourier(F_r_loaded)
g_reconstructed_loaded = reconstruct_from_fourier(F_g_loaded)
b_reconstructed_loaded = reconstruct_from_fourier(F_b_loaded)

# Combinar y mostrar la imagen reconstruida desde los coeficientes guardados
image_reconstructed_loaded = np.stack((r_reconstructed_loaded, g_reconstructed_loaded, b_reconstructed_loaded), axis=2)

plt.figure(figsize=(6, 6))
plt.title('Imagen Reconstruida desde los Coeficientes Guardados')
plt.imshow(np.uint8(np.clip(image_reconstructed_loaded, 0, 255)))
plt.axis('off')
plt.show()
