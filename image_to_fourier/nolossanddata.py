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

# Función para reconstruir el canal desde la transformada sin filtrar
def reconstruct_from_fourier(F_transform_shifted):
    # Deshacer el desplazamiento y aplicar la Transformada Inversa de Fourier
    F_transform_shifted_back = ifftshift(F_transform_shifted)
    reconstructed_channel = ifft2(F_transform_shifted_back)
    # Tomar solo la parte real de la imagen reconstruida
    return np.abs(reconstructed_channel)

# Cargar la imagen a color
image = cv2.imread('toy.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Separar los canales de color (R, G, B)
r_channel, g_channel, b_channel = image_rgb[:,:,0], image_rgb[:,:,1], image_rgb[:,:,2]

# Aplicar la Transformada de Fourier a cada canal sin filtrar
F_r = apply_fourier(r_channel)
F_g = apply_fourier(g_channel)
F_b = apply_fourier(b_channel)

# Función para imprimir los coeficientes de Fourier
def print_fourier_coefficients(F_transform, channel_name):
    print(f"\nCoeficientes de Fourier para el canal {channel_name}:")
    # Imprimir la parte real e imaginaria de los primeros 5x5 coeficientes
    for i in range(5):
        for j in range(5):
            real_part = np.real(F_transform[i, j])
            imag_part = np.imag(F_transform[i, j])
            print(f"({i},{j}): Real: {real_part:.2f}, Imaginario: {imag_part:.2f}")

# Imprimir coeficientes de Fourier para los canales R, G, B
print_fourier_coefficients(F_r, "Rojo")
print_fourier_coefficients(F_g, "Verde")
print_fourier_coefficients(F_b, "Azul")

# Reconstruir la imagen sin filtrar
r_reconstructed = reconstruct_from_fourier(F_r)
g_reconstructed = reconstruct_from_fourier(F_g)
b_reconstructed = reconstruct_from_fourier(F_b)

# Combinar los canales reconstruidos
image_reconstructed = np.stack((r_reconstructed, g_reconstructed, b_reconstructed), axis=2)

# Mostrar la imagen original y la reconstruida
plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
plt.title('Imagen Original')
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Imagen Reconstruida sin Filtrado (Máxima Calidad)')
plt.imshow(np.uint8(np.clip(image_reconstructed, 0, 255)))
plt.axis('off')

plt.show()
