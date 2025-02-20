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

# Reconstruir los canales desde las frecuencias filtradas
r_reconstructed = reconstruct_from_fourier(F_r_filtered)
g_reconstructed = reconstruct_from_fourier(F_g_filtered)
b_reconstructed = reconstruct_from_fourier(F_b_filtered)

# Normalizar los canales reconstruidos para visualización
r_norm = normalize_image(r_reconstructed)
g_norm = normalize_image(g_reconstructed)
b_norm = normalize_image(b_reconstructed)

# Combinar los canales reconstruidos
image_reconstructed = np.stack((r_norm, g_norm, b_norm), axis=2)

# Mostrar la imagen original y la reconstruida
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Imagen Original')
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f'Imagen Reconstruida con Filtro ({int(threshold_ratio*100)}% Frecuencias)')
plt.imshow(image_reconstructed)
plt.axis('off')

plt.show()


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
