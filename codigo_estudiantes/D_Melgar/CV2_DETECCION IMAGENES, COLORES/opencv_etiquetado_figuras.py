import cv2
import numpy as np

imagen = cv2.imread("Figuras.jpg")

if imagen is None:
    raise FileNotFoundError(
        "No se encontró figuras.jpg"
    )

hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

# Rango del color azul
rangos = {
    "azul": (
        np.array([90, 80, 50]),
        np.array([130, 255, 255])   
    ),
    "granate": (
        np.array([165, 80, 50]),
        np.array([179, 255, 142])
    ),
    
    "amarrillo": (
        np.array([20, 89, 255]),
        np.array([39, 255, 255])
    ),
    "verde": (
        np.array([42, 98, 216]),
        np.array([84, 255, 255])
    ),
    
    "rojo": (
        np.array([0, 80, 120]),
        np.array([12, 255, 255])
    ),
        
}

colores = {
    "azul":(255,0,0),
    "granate": (0,0, 150),
    "amarillo": (0,255, 255),
    "verde":  (0,180,0),
    "rojo" : (0,0,255)
}

# Trabajar sobre una copia
resultado = imagen.copy()


mascara_total = np.zeros(
    imagen.shape[:2], dtype=np.uint8
)

kernel = np.ones((3,3), np.uint8)

contador = 0

for color, (inferior, superior) in range.items():
    
    mascara = cv2.inRange(
        hsv, inferior, superior
    )


# Buscar regiones conectadas en la máscara
contornos, jerarquia = cv2.findContours(
    mascara,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


for contorno in contornos:

    # Calcular el área de la región
    area = cv2.contourArea(contorno)
    
    print("Area: ", area)

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
        (255,0, 0),
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