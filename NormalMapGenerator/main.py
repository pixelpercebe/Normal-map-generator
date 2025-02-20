from PIL import Image, ImageFilter
import numpy as np
from scipy import signal

# Convert RGB image to a greyscale image.
image = Image.open("th-3633531839.jpg")
image_name = image.filename[0:-4]
print(image_name + " loaded | mode: " + image.mode + " | format: " + image.format)
greyscale_image: Image = image.convert("L").filter(ImageFilter.SMOOTH_MORE)
print(image_name + " converted to " + greyscale_image.mode)
print(image_name + " updated | mode: " + greyscale_image.mode + " | format: png")
greyscale_image.save(image_name + "_greyscale.png", format="png")

# introduce the values of
img_array = np.array(greyscale_image)

gx = np.asarray([[-1, 0, 1],
                 [-2, 0, 2],
                 [-1, 0, 1]])

gy = np.asarray([[-1, -2, -1],
                 [0, 0, 0],
                 [1, 2, 1]])

# Get x-gradient and y-gradient in "sx" and "sy"
sx = signal.correlate2d(img_array, gx, boundary='symm', mode='same')
sy = signal.correlate2d(img_array, gy, boundary='symm', mode='same')

norm = np.sqrt(sx**2+sy**2+1.0)
print(norm)
normal_x = (sx / norm) * 0.5 + 0.5
normal_y = (sy / norm) * 0.5 + 0.5
normal_z = 1.0 / norm

# Normalizar los valores al rango [0, 255]
sx = (sx - np.min(sx)) / (np.max(sx) - np.min(sx)) * 255
sy = (sy - np.min(sy)) / (np.max(sy) - np.min(sy)) * 255
sx = sx.astype(np.uint8)
sy = sy.astype(np.uint8)

sx2 = sx ** 2
sy2 = sy ** 2



normal_map = np.stack([sx, sy, normal_z], axis=-1)
normal_map = (normal_map * 255).astype(np.uint8)

# Convertir los array en dos imagenes separadas
img_x = Image.fromarray(sx, 'L')  # Usar 'L' para escala de grises
img_y = Image.fromarray(sy, 'L')  # Usar 'L' para escala de grises
img_xy = Image.fromarray(normal_map, mode="RGB")
img_xy.save("normal_map.png")
img_x.save('sx.png')
img_y.save('sy.png')

# Get square root of sum of squares

