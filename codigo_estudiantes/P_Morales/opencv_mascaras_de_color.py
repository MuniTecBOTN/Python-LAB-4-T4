import cv2
import numpy as np


imagen_original = cv2.imread("bert_cajero.jpg")

if imagen_original is None:
    raise FileNotFoundError("No se encontró bert_cajero.jpg")


alto, ancho, _ = imagen_original.shape


relacion_aspecto = alto / ancho
ancho_nuevo = 800
alto_nuevo = int(relacion_aspecto * ancho_nuevo)


imagen_redimensionada = cv2.resize(
    imagen_original,
    (ancho_nuevo, alto_nuevo),
)

imagen_hsv = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_BGR2HSV)

limite_inferior = np.array([20, 100, 100])
limite_superior = np.array([35, 255, 255])


mascara = cv2.inRange(imagen_hsv, limite_inferior, limite_superior)


imagen_filtrada = cv2.bitwise_and(
    imagen_redimensionada,
    imagen_redimensionada,
    mask = mascara,
)


resultado_verde = np.full_like(imagen_redimensionada, (0, 255, 0))
resultado_verde[mascara > 0] = imagen_redimensionada[mascara > 0]


cv2.imshow("Resultado Verde", resultado_verde)
cv2.imshow("Imagen Filtrada", imagen_filtrada)
cv2.imshow("Mascara", mascara)
cv2.imshow("Imagen HSV", imagen_hsv)
cv2.imshow("Imagen Redimensionada", imagen_redimensionada)



cv2.waitKey(0)
cv2.destroyAllWindows()