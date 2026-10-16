import cv2
import numpy as np

imagen = cv2.imread("figuras.jpg")

if imagen is None:
    raise FileNotFoundError(
        "No se encontró figuras.jpg"
    )
# Obtener dimensiones originales
alto, ancho, _ = imagen.shape

# Calcular la nueva altura manteniendo la proporción
relacion_aspecto = alto / ancho
ancho_nuevo = 800
alto_nuevo = int(relacion_aspecto * ancho_nuevo)


imagen = cv2.resize(
    imagen,
    (ancho_nuevo, alto_nuevo)
)   
hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

# Definir rangos aproximados para cada color
colores = {
    "verde": (
        np.array([35, 70, 50]),
        np.array([85, 255, 255])
    ),
    "azul": (
        np.array([90, 70, 50]),
        np.array([130, 255, 255])
    ),
    "amarillo": (
        np.array([20, 100, 100]),
        np.array([35, 255, 255])
    )
}

for nombre, (inferior, superior) in colores.items():

    # Crear una máscara para el color actual
    mascara = cv2.inRange(
        hsv, inferior, superior
    )

    # Extraer los píxeles del color
    resultado = cv2.bitwise_and(
        imagen, imagen, mask=mascara
    )

    # Mostrar cada resultado
    cv2.imshow(
        "Mascara " + nombre,
        mascara
    )

    cv2.imshow(
        "Objetos " + nombre,
        resultado
    )

    # Guardar los resultados
    cv2.imwrite(
        "mascara_" + nombre + ".jpg",
        mascara
    )

    cv2.imwrite(
        "objetos_" + nombre + ".jpg",
        resultado
    )

cv2.waitKey(0)
cv2.destroyAllWindows()