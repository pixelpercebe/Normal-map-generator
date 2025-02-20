def obtener_binario_imagen(imagen_ruta, archivo_salida):
    # Abrimos la imagen en modo binario
    with open(imagen_ruta, 'rb') as imagen:
        # Leemos to do el contenido de la imagen en formato binario
        datos_binarios = imagen.read()

    # Abrimos el archivo de salida para escribir los bytes como enteros
    with open(archivo_salida, 'w') as archivo:
        # Iteramos sobre cada byte
        for byte in datos_binarios:
            # Convertimos el byte a entero y lo escribimos en el archivo
            archivo.write(f"{byte}\n")

    print(f"El código binario de la imagen ha sido guardado en {archivo_salida}")

# Uso del programa
imagen_ruta = 'cat.jpeg'  # Ruta de la imagen de entrada
archivo_salida = 'binario_imagen.txt'  # Archivo de salida

obtener_binario_imagen(imagen_ruta, archivo_salida)


