import cv2
import numpy as np

imagen = cv2.imread("Figuras.jpg")

if imagen is None:
    raise FileNotFoundError("No se encontró figuras.jpg")

hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

# Rango del color azul
rangos = {
    "azul": (np.array([90, 80, 50]), np.array([130, 255, 255])),
    "granate": (np.array([165, 80, 40]), np.array([179, 255, 220])),
    "amarillo": (np.array([18,80,10]), np.array([39, 255, 255])),
    "verde": (np.array([40,60,40]), np.array([90, 255, 255])),
    "rojo": (np.array([0, 80, 100]), np.array([12, 255, 255])),
}

colores = {
    "azul": (255, 0, 0),
    "granate": (0, 0, 150),
    "amarillo": (0, 255, 255),
    "verde": (0, 180, 0),
    "rojo": (0, 0, 255),
}

# Trabajar sobre una copia
resultado = imagen.copy()


mascara_total = np.zeros(imagen.shape[:2], dtype=np.uint8)

kernel = np.ones((3, 3), np.uint8)

contador = 0

for color, (inferior, superior) in rangos.items():

    mascara = cv2.inRange(hsv, inferior, superior)

    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara_total = cv2.bitwise_or(mascara_total, mascara)

    # Buscar regiones conectadas en la máscara

    contornos, jerarquia = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contorno in contornos:

        # Calcular el área de la región
        area = cv2.contourArea(contorno)

        print("Color: ", color)
        print("Area: ", area)

        # Ignorar regiones demasiado pequeñas
        if area < 300:
         continue

         # Obtener las coordenadas del rectángulo
        x, y, ancho, alto = cv2.boundingRect(contorno)

        if color == "azul":
            if alto > ancho:
                nombre = "flecha arriba"
            else:
                nombre = "flecha izquierda"
                
        elif color == "granate":
            if area > 8000:
                nombre = "Ovalo grande"
            else:
                nombre = "Circulo granate"

        elif color == "amarillo":
            nombre = "pentagono amarillo"
        elif color == "verde":
            nombre = "triangulo verde"
        elif color == "rojo":
            nombre = "Cuadrado rojo"

        cv2.drawContours(resultado, [contorno], -1, colores[color], 2)
        # Dibujar el rectángulo
        cv2.rectangle(resultado, (x, y), (x + ancho, y + alto), colores[color], 2)

        # Añadir una etiqueta
        cv2.putText(
            resultado,
            nombre,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colores[color],
            2,
            cv2.LINE_AA,
        )

        print(
            "Figura", nombre,
            "Area: ", round (area, 2),
            "|X", x, 
            "|Y", y,
            "|Ancho", ancho,
            "|Alto", alto
            )

        contador += 1

print("Regiones detectadas:", contador)

cv2.imshow("Mascara", mascara_total)
cv2.imshow("Objetos detectados", resultado)

cv2.imwrite("objetos_detectados.jpg", resultado)

cv2.waitKey(0)
cv2.destroyAllWindows()
