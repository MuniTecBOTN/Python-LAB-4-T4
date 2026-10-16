import cv2
import numpy as np

# Cargar imagen original
imagen_original = cv2.imread("figuras.jpg")

if imagen_original is None:
    raise FileNotFoundError("No se encontró figuras.jpg")

# Obtener dimensiones originales
alto, ancho, _ = imagen_original.shape

# Calcular la nueva altura manteniendo la proporción
relacion_aspecto = alto / ancho
ancho_nuevo = 800
alto_nuevo = int(relacion_aspecto * ancho_nuevo)

# Redimensionar la imagen original
imagen_redimensionada = cv2.resize(
    imagen_original,
    (ancho_nuevo, alto_nuevo),
)

imagen_hsv = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_BGR2HSV)

limite_inferior = np.array([115, 100, 100])
limite_superior = np.array([120, 255, 255])

mascara = cv2.inRange(imagen_hsv, limite_inferior, limite_superior)
cv2.imshow("Mascara", mascara)

cv2.imshow("Imagen HSV", imagen_hsv)
cv2.imshow("Imagen Redimensionada", imagen_redimensionada)

cv2.waitKey(0)
cv2.destroyAllWindows()
