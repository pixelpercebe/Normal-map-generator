import cv2
import numpy as np


# Función para convertir una imagen en código binario
def image_to_binary(image_path, output_txt_path):
	# Cargar la imagen en formato de escala de grises
	image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

	# Verificar que la imagen se haya cargado correctamente
	if image is None:
		print("Error: No se pudo cargar la imagen.")
		return

	# Convertir la imagen en una matriz de enteros
	rows, cols = image.shape

	# Abrir el archivo de salida en modo escritura
	with open(output_txt_path, 'w') as file:
		for row in range(rows):
			for col in range(cols):
				# Obtener el valor del píxel (entre 0 y 255)
				pixel_value = image[row, col]

				# Convertir el valor a binario (rellenar con ceros para tener 8 bits)
				binary_value = format(pixel_value, '08b')

				# Escribir el valor binario en el archivo
				file.write(binary_value)

				# Agregar un espacio entre cada píxel para legibilidad
				file.write(" ")

			# Saltar a la siguiente línea para cada fila
			file.write("\n")

	print(f"Código binario guardado en: {output_txt_path}")


# Ejemplo de uso
image_path = 'toy.jpg'  # Reemplaza esto con la ruta de tu imagen
output_txt_path = 'imagen_binaria.txt'

# Convertir la imagen a binario y guardar en un archivo .txt
image_to_binary(image_path, output_txt_path)
