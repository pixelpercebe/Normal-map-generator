import numpy as np
import cv2
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

# Función para aplicar la Transformada de Fourier a un canal de color
def apply_fourier(channel):
    F_transform = fft2(channel)
    F_transform_shifted = fftshift(F_transform)
    return F_transform_shifted

# Función para aplicar un filtro de frecuencias basado en magnitud
def filter_frequencies(F_transform_shifted, threshold_ratio):
    magnitude = np.abs(F_transform_shifted)
    threshold = threshold_ratio * np.max(magnitude)
    mask = magnitude >= threshold
    F_filtered = F_transform_shifted * mask
    return F_filtered

# Función para reconstruir la imagen desde la transformada filtrada
def reconstruct_from_fourier(F_transform_shifted):
    F_transform_shifted_back = ifftshift(F_transform_shifted)
    reconstructed_channel = ifft2(F_transform_shifted_back)
    return np.abs(reconstructed_channel)

# Función para normalizar la imagen para visualización
def normalize_image(img):
    img = img - np.min(img)
    if np.max(img) != 0:
        normalized = np.uint8(255 * img / np.max(img))
    else:
        normalized = np.uint8(img)
    return normalized

# Cargar la imagen en color
image = cv2.imread('toy.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Separar los canales de color (Rojo, Verde, Azul)
r_channel = image_rgb[:, :, 0]
g_channel = image_rgb[:, :, 1]
b_channel = image_rgb[:, :, 2]

# Aplicar la Transformada de Fourier a cada canal
F_r = apply_fourier(r_channel)
F_g = apply_fourier(g_channel)
F_b = apply_fourier(b_channel)

# Aplicar el filtro de frecuencias (ajusta threshold_ratio según sea necesario)
threshold_ratio = 0.0005  # Mantener el 5% de las frecuencias más altas
F_r_filtered = filter_frequencies(F_r, threshold_ratio)
F_g_filtered = filter_frequencies(F_g, threshold_ratio)
F_b_filtered = filter_frequencies(F_b, threshold_ratio)

# Función para combinar las frecuencias filtradas y generar una onda sumada
def combine_filtered_frequencies(F_transform_shifted_filtered):
    # Obtener las dimensiones de la imagen
    rows, cols = F_transform_shifted_filtered.shape
    combined_wave = np.zeros((rows, cols), dtype=complex)

    # Coordenadas de frecuencia
    u = np.arange(-rows//2, rows//2)
    v = np.arange(-cols//2, cols//2)
    U, V = np.meshgrid(u, v, indexing='ij')

    # Sumatoria de las ondas de las frecuencias filtradas
    for i in range(rows):
        for j in range(cols):
            if F_transform_shifted_filtered[i, j] != 0:
                amplitude = np.abs(F_transform_shifted_filtered[i, j])
                phase = np.angle(F_transform_shifted_filtered[i, j])
                freq_u = U[i, j]
                freq_v = V[i, j]
                # Crear la onda y sumarla
                wave = amplitude * np.cos(2 * np.pi * (freq_u * i / rows + freq_v * j / cols) + phase)
                combined_wave[i, j] += wave

    return np.real(combined_wave)

# Ajuste del factor de escala para aumentar las amplitudes
amplitude_factor = 10000  # Aumenta este valor para visualizar mejor la onda combinada

# Multiplicamos la onda combinada por el factor de amplitud
combined_wave_r_scaled = amplitude_factor * combine_filtered_frequencies(F_r_filtered)
combined_wave_g_scaled = amplitude_factor * combine_filtered_frequencies(F_g_filtered)
combined_wave_b_scaled = amplitude_factor * combine_filtered_frequencies(F_b_filtered)

# Visualizar la ecuación sumada para el canal Rojo, Verde y Azul
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Onda Combinada (Rojo)')
plt.imshow(combined_wave_r_scaled, cmap='gray', vmin=np.min(combined_wave_r_scaled), vmax=np.max(combined_wave_r_scaled))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Onda Combinada (Verde)')
plt.imshow(combined_wave_g_scaled, cmap='gray', vmin=np.min(combined_wave_g_scaled), vmax=np.max(combined_wave_g_scaled))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Onda Combinada (Azul)')
plt.imshow(combined_wave_b_scaled, cmap='gray', vmin=np.min(combined_wave_b_scaled), vmax=np.max(combined_wave_b_scaled))
plt.axis('off')

plt.show()
