import cv2
import numpy as np

imagen = cv2.imread("Figuras.jpg")

if imagen is None:
    raise FileNotFoundError("No se encontró objetos_colores.jpg")

hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

# Definir rangos aproximados para cada color
colores = {
    "verde": (np.array([35, 70, 50]), np.array([85, 255, 255])),
    "azul": (np.array([90, 70, 50]), np.array([130, 255, 255])),
    "amarillo": (np.array([20, 100, 100]), np.array([35, 255, 255])),
}

mascaras = []

for nombre, (inferior, superior) in colores.items():

    # Crear una máscara para el color actual
    mascara = cv2.inRange(hsv, inferior, superior)

    # Extraer los píxeles del color
    resultado = cv2.bitwise_and(imagen, imagen, mask=mascara)

    # Mostrar cada resultado
    cv2.imshow("Mascara " + nombre, mascara)

    cv2.imshow("Objetos " + nombre, resultado)

    # Guardar los resultados
    cv2.imwrite("mascara_" + nombre + ".jpg", mascara)

    cv2.imwrite("objetos_" + nombre + ".jpg", resultado)
    mascaras.append(mascara)

mascara_combinada = mascaras[0] + mascaras[1] + mascaras[2]


cv2.waitKey(0)
cv2.destroyAllWindows()
