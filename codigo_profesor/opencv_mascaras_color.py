import cv2
import numpy as np

# Cargar imagen original
imagen_original = cv2.imread("manzanas.webp")

if imagen_original is None:
    raise FileNotFoundError("No se encontró manzanas.webp")

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

limite_inferior = np.array([171, 73, 51])
limite_superior = np.array([180, 255, 255])

limite_inferior_2 = np.array([0, 0, 0])
limite_superior_2 = np.array([13, 255, 255])

mascara_1 = cv2.inRange(imagen_hsv, limite_inferior, limite_superior)
mascara_2 = cv2.inRange(imagen_hsv, limite_inferior_2, limite_superior_2)

mascara_combinada = cv2.bitwise_or(mascara_1, mascara_2)

imagen_filtrada = cv2.bitwise_and(
    imagen_redimensionada,
    imagen_redimensionada,
    mask=mascara_combinada
)

# Crear un fondo verde
resultado_verde = np.full_like(imagen_redimensionada, (80, 170, 120))
print(f"codigo fondo: {resultado_verde[0,0]}")

resultado_verde[mascara_combinada > 0] = imagen_redimensionada[mascara_combinada > 0]

cv2.imshow("Resultado Verde", resultado_verde)
cv2.imshow("Imagen Filtrada", imagen_filtrada)
cv2.imshow("Mascara", mascara_1)
cv2.imshow("Imagen HSV", imagen_hsv)
cv2.imshow("Imagen Redimensionada", imagen_redimensionada)

cv2.waitKey(0)
cv2.destroyAllWindows()
