import cv2
import numpy as np

imagen = cv2.imread("Figuras.jpg")

if imagen is None:
    raise FileNotFoundError(
        "No se encontró figuras.jpg"
    )

hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

# Rango del color azul
inferior = np.array([90, 80, 50])
superior = np.array([130, 255, 255])

mascara = cv2.inRange(
    hsv, inferior, superior
)

# Buscar regiones conectadas en la máscara
contornos, jerarquia = cv2.findContours(
    mascara,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Trabajar sobre una copia
resultado = imagen.copy()

contador = 0

for contorno in contornos:

    # Calcular el área de la región
    area = cv2.contourArea(contorno)

    # Ignorar regiones demasiado pequeñas
    if area < 300:
        continue

    # Obtener las coordenadas del rectángulo
    x, y, ancho, alto = cv2.boundingRect(
        contorno
    )

    contador += 1

    # Dibujar el rectángulo
    cv2.rectangle(
        resultado,
        (x, y),
        (x + ancho, y + alto),
        (0, 255, 0),
        2
    )

    # Añadir una etiqueta
    cv2.putText(
        resultado,
        "Azul " + str(contador),
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

print("Regiones detectadas:", contador)

cv2.imshow("Mascara", mascara)
cv2.imshow("Objetos detectados", resultado)

cv2.imwrite("objetos_detectados.jpg", resultado)

cv2.waitKey(0)
cv2.destroyAllWindows()