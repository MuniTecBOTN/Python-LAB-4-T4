
import cv2
import numpy as np

# 1. Cargar la imagen
imagen = cv2.imread("Figuras.jpg")

if imagen is None:
    raise FileNotFoundError(
        "No se encontro Figuras.jpg"
    )

# 2. Convertir la imagen a HSV
hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

# 3. Definir los rangos de cada color
rangos = {
    "azul": (
        np.array([90, 80, 50]),
        np.array([135, 255, 255])
    ),
    "granate": (
        np.array([165, 80, 40]),
        np.array([179, 255, 220])
    ),
    "amarillo": (
        np.array([18, 80, 100]),
        np.array([40, 255, 255])
    ),
    "verde": (
        np.array([40, 60, 40]),
        np.array([90, 255, 255])
    ),
    "rojo": (
        np.array([0, 80, 100]),
        np.array([10, 255, 255])
    )
}

# 4. Colores para dibujar, en formato BGR
colores = {
    "azul": (255, 0, 0),
    "granate": (0, 0, 150),
    "amarillo": (0, 255, 255),
    "verde": (0, 180, 0),
    "rojo": (0, 0, 255)
}

# 5. Preparar la imagen de salida
resultado = imagen.copy()

# Mascara general para mostrar los colores detectados
mascara_total = np.zeros(
    imagen.shape[:2],
    dtype=np.uint8
)

kernel = np.ones((3, 3), np.uint8)

contador = 0

# 6. Procesar cada color
for color, (inferior, superior) in rangos.items():

    mascara = cv2.inRange(
        hsv, inferior, superior
    )

    # Limpiar pequenos puntos de ruido
    mascara = cv2.morphologyEx(
        mascara, cv2.MORPH_OPEN, kernel
    )

    mascara = cv2.morphologyEx(
        mascara, cv2.MORPH_CLOSE, kernel
    )

    mascara_total = cv2.bitwise_or(
        mascara_total, mascara
    )

    # 7. Buscar todos los contornos del color actual
    contornos, jerarquia = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Analizar cada contorno encontrado
    for contorno in contornos:

        # Calcular el area
        area = cv2.contourArea(contorno)

        # Ignorar regiones pequenas
        if area < 300:
            continue

        # Obtener ubicacion y dimensiones
        x, y, ancho, alto = cv2.boundingRect(
            contorno
        )

        # 8. Asignar el nombre de la figura
        if color == "azul":
            if alto > ancho:
                nombre = "Flecha arriba"
            else:
                nombre = "Flecha izquierda"

        elif color == "granate":
            if area > 8000:
                nombre = "Ovalo granate"
            else:
                nombre = "Circulo granate"

        elif color == "amarillo":
            nombre = "Pentagono amarillo"

        elif color == "verde":
            nombre = "Triangulo verde"

        elif color == "rojo":
            nombre = "Cuadrado rojo"

        # 9. Dibujar el contorno de la figura
        cv2.drawContours(
            resultado,
            [contorno],
            -1,
            colores[color],
            2
        )

        # 10. Dibujar el rectangulo delimitador
        cv2.rectangle(
            resultado,
            (x, y),
            (x + ancho, y + alto),
            colores[color],
            2
        )

        # 11. Escribir el nombre de la figura
        cv2.putText(
            resultado,
            nombre,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colores[color],
            2,
            cv2.LINE_AA
        )

        # 12. Mostrar informacion en la consola
        print(
            "Figura:", nombre,
            "| Area:", round(area, 2),
            "| X:", x,
            "| Y:", y,
            "| Ancho:", ancho,
            "| Alto:", alto
        )

        contador += 1

# 13. Mostrar y guardar los resultados
print("Regiones detectadas:", contador)

cv2.imshow("Mascara de colores", mascara_total)
cv2.imshow("Objetos detectados", resultado)

cv2.imwrite("objetos_detectados.jpg", resultado)

cv2.waitKey(0)
cv2.destroyAllWindows()
