from PIL import Image, ImageFilter,ImageEnhance
import numpy as np
from scipy import signal

# Convert RGB image to a greyscale image.
image = Image.open("stylized_stone.jpg")
image_name = image.filename[0:-4]
print(image_name + " loaded | mode: " + image.mode + " | format: " + image.format)
greyscale_image: Image = image.convert("L").filter(ImageFilter.SMOOTH_MORE)
gaussian_image = image.filter(ImageFilter.GaussianBlur(radius=0)).convert("L")
gaussian_image.save("gaussian_version.png")
print(image_name + " converted to " + greyscale_image.mode)
print(image_name + " updated | mode: " + greyscale_image.mode + " | format: png")
greyscale_image.save(image_name + "_greyscale.png", format="png")

greyscale_image = gaussian_image

# introduce the values of
img_array = np.array(greyscale_image)

gx = np.asarray([[-1, 0, 1],
                 [-2, 0, 2],
                 [-1, 0, 1]])

gxP = np.asarray([[-1, 0, 1],
                 [-0.5, 0, 0.5],
                 [-1, 0, 1]])

gx5 = np.asarray([[2, 2, 4, 2, 2],
                  [1, 1, 1, 1, 1],
                  [0, 0, 0, 0, 0],
                  [-1, -1, -2, -1, - 1],
                  [-2, -2, -4, -2, -2]])

gy = np.asarray([[-1, -2, -1],
                 [0, 0, 0],
                 [1, 2, 1]])

gyP = np.asarray([[-1, -2, -1],
                 [0, 0, 0],
                 [1, 2, 1]])

gy5 = np.asarray([[2, 1, 0, -1, -2],
                  [2, 1, 0, -1, -2],
                  [4, 2, 0, -2, -4],
                  [2, 1, 0, -1, -2],
                  [2, 1, 0, -1, -2]])


strength = 20  # Factor de "altura" (mayor = más pronunciado)

# Get x-gradient and y-gradient in "sx" and "sy"
sx = signal.correlate2d(img_array, gx, boundary='fill', mode='full')
sy = -signal.correlate2d(img_array, gy, boundary='fill', mode='full')

# Normalizar los valores al rango [0, 255]
norm = np.sqrt(sx ** 2 + sy ** 2 + 1.0)

normal_x = sx / norm
sx = (sx - np.min(sx)) / (np.max(sx) - np.min(sx)) * 255
normal_y = sy / norm
sy = (sy - np.min(sy)) / (np.max(sy) - np.min(sy)) * 255
normal_z = (1.0 / norm) + 1

normal_x = sx.astype(np.uint8)
normal_y = sy.astype(np.uint8)

# Compute the normal map
# normal_map = np.stack([normal_x, normal_y, normal_z], axis=-1)
normal_map = np.stack([normal_x, normal_y, normal_z], axis=-1)
normal_map = -normal_map - np.min(normal_map) / (np.max(normal_map) - np.min(normal_map)) * 255  # bien normalizado ?
normal_map = normal_map.astype(np.uint8)

# Convertir los array en dos imagenes separadas
img_x = Image.fromarray(normal_x, 'L')  # Usar 'L' para escala de grises
img_y = Image.fromarray(normal_y, 'L')  # Usar 'L' para escala de grises
normal_map_image = Image.fromarray(normal_map, mode="RGB")

#Aqui sucede la magia



mejorador_sharpness = ImageEnhance.Sharpness(normal_map_image)
normal_map_image = mejorador_sharpness.enhance(1)

mejorador_contraste = ImageEnhance.Contrast(normal_map_image)
normal_map_image = mejorador_contraste.enhance(1).filter(ImageFilter.BoxBlur(3))

mejorador_saturacion = ImageEnhance.Color(normal_map_image)
normal_map_image = mejorador_saturacion.enhance(4)


 #mirar como cambiar la altura del nnormal map. el contraste lo ahce azul >:(





img_x.save('sx.png')
img_y.save('sy.png')
normal_map_image.save("normal_map.png")
normal_map_image.show()

# Get square root of sum of squares
