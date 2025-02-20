import numpy as np
import cv2
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt


# Función para aplicar la Transformada de Fourier a un canal de color
def apply_fourier(channel):
	F_transform = fft2(channel)
	F_transform_shifted = fftshift(F_transform)
	return F_transform_shifted


# Función para reconstruir la imagen desde la transformada
def reconstruct_from_fourier(F_transform_shifted):
	F_transform_shifted_back = ifftshift(F_transform_shifted)
	reconstructed_channel = ifft2(F_transform_shifted_back)
	return np.abs(reconstructed_channel)


# Función para combinar las frecuencias y generar una onda sumada ponderada
def combine_frequencies(F_transform_shifted, weight_factor=0.01):
	combined_wave = np.zeros(F_transform_shifted.shape)
	rows, cols = F_transform_shifted.shape

	for i in range(rows):
		for j in range(cols):
			real_part = np.real(F_transform_shifted[i, j])
			imag_part = np.imag(F_transform_shifted[i, j])
			frequency_magnitude = np.sqrt(real_part ** 2 + imag_part ** 2)

			# Ponderar más las frecuencias importantes
			weight = frequency_magnitude ** weight_factor

			# Crear una onda sinusoidal ponderada para cada frecuencia
			wave = (real_part * np.cos(2 * np.pi * (i + j)) + imag_part * np.sin(2 * np.pi * (i + j))) * weight
			combined_wave[i, j] += wave

	return combined_wave


# Función para aplicar un filtro de paso alto (High-Pass)
def apply_high_pass_filter(F_transform_shifted, cutoff=0.1):
	rows, cols = F_transform_shifted.shape
	center_row, center_col = rows // 2, cols // 2
	mask = np.zeros((rows, cols), dtype=np.float32)

	# Crear una máscara circular de paso alto
	for i in range(rows):
		for j in range(cols):
			distance_to_center = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
			if distance_to_center > cutoff * np.sqrt(center_row ** 2 + center_col ** 2):
				mask[i, j] = 1

	# Aplicar la máscara
	F_transform_shifted_filtered = F_transform_shifted * mask
	return F_transform_shifted_filtered


# Función para normalizar los valores a un rango [0, 255] para visualización
def normalize_image(img):
	normalized = np.uint8(255 * (img - np.min(img)) / (np.max(img) - np.min(img)))
	return normalized


# Cargar la imagen en color
image = cv2.imread('toy.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Separar los canales de color (Rojo, Verde, Azul)
r_channel, g_channel, b_channel = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]

# Aplicar la Transformada de Fourier a cada canal
F_r = apply_fourier(r_channel)
F_g = apply_fourier(g_channel)
F_b = apply_fourier(b_channel)

# Aplicar el filtro de paso alto a cada canal
F_r_filtered = apply_high_pass_filter(F_r, cutoff=0.1)  # El valor de cutoff controla cuántas frecuencias bajas filtrar
F_g_filtered = apply_high_pass_filter(F_g, cutoff=0.1)
F_b_filtered = apply_high_pass_filter(F_b, cutoff=0.1)

# Combinar las frecuencias para cada canal usando una ponderación
combined_r = combine_frequencies(F_r_filtered, weight_factor=0.01)
combined_g = combine_frequencies(F_g_filtered, weight_factor=0.01)
combined_b = combine_frequencies(F_b_filtered, weight_factor=0.01)

# Normalizar las imágenes combinadas
combined_r_norm = normalize_image(combined_r)
combined_g_norm = normalize_image(combined_g)
combined_b_norm = normalize_image(combined_b)

# Mostrar la combinación de las ondas resultantes de cada canal
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Onda Combinada (Rojo)')
plt.imshow(combined_r_norm, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Onda Combinada (Verde)')
plt.imshow(combined_g_norm, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Onda Combinada (Azul)')
plt.imshow(combined_b_norm, cmap='gray')
plt.axis('off')

plt.show()