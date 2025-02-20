from PIL import Image
import io
import qrcode
import base64


def convertir_imagen_a_binario(nombre_archivo):
	with Image.open(nombre_archivo) as img:
		byte_arr = io.BytesIO()
		img.save(byte_arr, format='PNG')
		datos_binarios = byte_arr.getvalue()
	return datos_binarios


def generar_qr_desde_binario(data_binaria, nombre_archivo):
	qr = qrcode.QRCode(
		version=1,
		error_correction=qrcode.constants.ERROR_CORRECT_L,
		box_size=10,
		border=4,
	)
	qr.add_data(data_binaria)
	qr.make(fit=True)
	img = qr.make_image(fill='black', back_color='white')
	img.save(nombre_archivo)


# Ejemplo de uso
nombre_imagen = "C:/Users/icalb/Desktop/Ilustración_sin_título (3).png"  # Reemplaza con la ruta a tu imagen
datos_binarios = convertir_imagen_a_binario(nombre_imagen)
len_bin = len(datos_binarios)
print(len_bin)
if len_bin<7000:
	generar_qr_desde_binario(datos_binarios, "qr_code_imagen.png")
else:
	print("tamaño mayor de 7000")
