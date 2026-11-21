
import cv2
import numpy as np

nombre_archivo = "1_cuadrados.bmp"

# Cargar imagen original
imagen_original = cv2.imread(nombre_archivo)

if imagen_original is None:
    raise FileNotFoundError(
        f"No se encontro {nombre_archivo}. "
        "Verifica que esté en la carpeta del programa."
    )

# Obtener dimensiones originales
altura_imagen, ancho_imagen, canales_color = imagen_original.shape

print(
    f"INF_IMAGEN: {altura_imagen}x"
    f"{ancho_imagen}-{canales_color}"
)

# Convertir imagen a escala de grises
imagen_grises = cv2.cvtColor(
    imagen_original,
    cv2.COLOR_BGR2GRAY
)

# Convertir a imagen binaria
_, imagen_binaria = cv2.threshold(
    imagen_grises,
    127,
    255,
    cv2.THRESH_BINARY_INV
)

# Encontrar contornos externos
contornos, _ = cv2.findContours(
    imagen_binaria,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Recorrer cada objeto detectado
for index, contorno in enumerate(contornos):
    area = cv2.contourArea(contorno)

    x_pi, y_pi, ancho_caja, alto_caja = (
        cv2.boundingRect(contorno)
    )

    # Dibujar rectángulo alrededor del objeto
    cv2.rectangle(
        imagen_original,
        (x_pi, y_pi),
        (x_pi + ancho_caja, y_pi + alto_caja),
        (123, 255, 77),
        2
    )

    # Dibujar contorno del objeto
    cv2.drawContours(
        imagen_original,
        [contorno],
        -1,
        (255, 255, 0),
        1
    )

    # Calcular posición de las etiquetas
    x_etiqueta_objeto = max(0, x_pi - 10)
    y_etiqueta_objeto = y_pi - 20

    if y_etiqueta_objeto <= 0:
        y_etiqueta_objeto = min(
            altura_imagen - 5,
            y_pi + alto_caja + 20
        )

    # Mostrar número del objeto
    cv2.putText(
        imagen_original,
        f"OBJETO {index}",
        (x_etiqueta_objeto, y_etiqueta_objeto),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 26, 200),
        2
    )

    # Mostrar área del objeto
    cv2.putText(
        imagen_original,
        f"Area: {area:.0f} px2",
        (max(0, x_pi - 10), min(
            altura_imagen - 5,
            y_pi + alto_caja + 30
        )),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 26, 200),
        2
    )

    # Mostrar información en la terminal
    print(f"\nCONTORNO #{index}")
    print(f"AREA: {area:.2f} px2")

    for punto in contorno:
        print(punto)

# Mostrar imagen con las detecciones
cv2.imshow("Imagen Original", imagen_original)
cv2.imshow("Imagen Binaria", imagen_binaria)

# Esperar hasta presionar una tecla
cv2.waitKey(0)

# Cerrar las ventanas
cv2.destroyAllWindows()
