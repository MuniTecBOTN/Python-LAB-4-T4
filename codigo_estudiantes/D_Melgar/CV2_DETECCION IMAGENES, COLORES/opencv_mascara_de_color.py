import cv2
import numpy as np

imagen_original = cv2.imread("manzana.jpg")

if imagen_original is None:
    raise FileExistsError("No se encontro manzana.jpg")

alto, ancho, _ = imagen_original.shape


#calcular la nueva altura manteniendo la proporción

relacion_aspecto = alto/ancho

ancho_nuevo = 600
alto_nuevo = int(relacion_aspecto * ancho_nuevo)

imagen_redimensionada = cv2.resize(
    imagen_original,
    (ancho_nuevo, alto_nuevo),
)

imagen_hsv = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_BGR2HSV)

limite_inferior = np.array([170, 73, 51])
limite_superior = np.array([179, 255, 255])

limite_inferior2 = np.array([0, 120, 127])
limite_superior2 = np.array([7, 255, 255])

mascara = cv2.inRange(imagen_hsv, limite_inferior, limite_superior)
mascara_2 = cv2.inRange(imagen_hsv, limite_inferior2, limite_superior2)

mascara_combinada = cv2.bitwise_or(mascara, mascara_2)


imagen_filtrada = cv2.bitwise_and(
    imagen_redimensionada,
    imagen_redimensionada,
    mask=mascara 
)


resultado_verde = np.full_like(imagen_redimensionada, (150,233,255))



resultado_verde[mascara > 0] =  imagen_redimensionada[mascara > 0]

cv2.imshow ("Resultado verde", resultado_verde)

cv2.imshow("Imagen Filtrada", imagen_filtrada)

cv2.imshow("Mascara", mascara)


cv2.imshow("imagen_hsv", imagen_hsv)

cv2.imshow ("Imagen_redimensionada", imagen_redimensionada)

cv2.waitKey(0)
cv2.destroyAllWindows()