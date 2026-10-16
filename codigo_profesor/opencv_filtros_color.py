import cv2
import numpy as np

# Cargar imagen_redimensionada
imagen_original = cv2.imread("berries.jpg")

if imagen_original is None:
    raise FileNotFoundError("No se encontró berries.jpg")

# Obtener dimensiones originales
alto, ancho, _ = imagen_original.shape

# Definir el ancho deseado
nuevo_ancho = 600

# Calcular la altura manteniendo la proporción
nuevo_alto = int((alto / ancho) * nuevo_ancho)

# Redimensionar la imagen_redimensionada
imagen_redimensionada = cv2.resize(
    imagen_original,
    (nuevo_ancho, nuevo_alto),
    interpolation=cv2.INTER_AREA
)

# Convertir la imagen_redimensionada a HSV
hsv = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_BGR2HSV)
cv2.imshow("Imagen HSV", hsv)
# Definir el rango del color rojo
inferior1 = (0, 100, 80)
superior1 = (255, 255, 255)
# Con NumPy se podrían definir como arrays, pero aquí se usan tuplas directamente.
# inferior1 = np.array([0, 100, 80])

inferior2 = (170, 100, 80)
superior2 = (179, 255, 255)

# Crear dos máscaras para cubrir el rojo
mascara1 = cv2.inRange(hsv, inferior1, superior1)
mascara2 = cv2.inRange(hsv, inferior2, superior2)
cv2.imshow("Mascara1", mascara1)
cv2.imshow("Mascara2", mascara2)
# Unir ambas máscaras
mascara = cv2.bitwise_or(mascara1, mascara2)

# Extraer el color seleccionado
resultado = cv2.bitwise_and(
    imagen_redimensionada, imagen_redimensionada, mask=mascara
)

# Crear un fondo verde
resultado_verde = np.full_like(imagen_redimensionada, (0, 255, 0))

# Conservar las moras y cambiar el fondo
resultado_verde[mascara > 0] = imagen_redimensionada[mascara > 0]
cv2.imshow("Resultado Verde", resultado_verde)
cv2.imshow("Resultado", resultado)
# Mostrar resultados
cv2.imshow("Imagen redimensionada", imagen_redimensionada)
cv2.imshow("Mascara roja", mascara)
cv2.imshow("Resultado", resultado)

# Guardar la imagen_redimensionada procesada
cv2.imwrite("resultado_rojo.jpg", resultado)

cv2.waitKey(0)
cv2.destroyAllWindows()