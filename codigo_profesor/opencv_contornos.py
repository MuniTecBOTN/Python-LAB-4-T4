import cv2
import numpy as np

nombre_archivo = "3_cuadrados.bmp"
imagen_original = cv2.imread(nombre_archivo)

if imagen_original is None:
    raise FileNotFoundError(f"No se encontró {nombre_archivo}")

# Obtener dimensiones originales
altura_imagen, ancho_imagen, canales_color = imagen_original.shape

print(f"INF_IMAGEN: {altura_imagen}x{ancho_imagen} - {canales_color}")

imagen_grises = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)

_, imagen_binaria = cv2.threshold(
    imagen_grises,
    127,
    255,
    cv2.THRESH_BINARY,
)

contornos, _ = cv2.findContours(
    imagen_binaria,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

for index, contorno in enumerate(contornos):
    area = cv2.contourArea(contorno)
    if area < 10000:
        continue
    
    x_pi, y_pi, ancho_caja, alto_caja = cv2.boundingRect(contorno)

    cv2.rectangle(
        imagen_original,
        (x_pi - 10, y_pi - 10),
        (x_pi + ancho_caja + 10, y_pi + alto_caja + 10),
        (255, 255, 255),
        2,
    )
    cv2.drawContours(
        imagen_original,
        contornos,
        index,
        (255, 255, 255),
        1,
    )
    y_etiqueta_objeto = y_pi - 20
    x_etiqueta_objeto = x_pi - 10
    if y_etiqueta_objeto <= 0:
        y_etiqueta_objeto = y_pi + alto_caja + 45
        
    cv2.putText(
        imagen_original,
        f"OBJETO {index}",
        (x_etiqueta_objeto, y_etiqueta_objeto),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        imagen_original,
        f"AREA: {area} px2",
        (x_pi - 10, y_pi + alto_caja + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    print(f"\nCONTORNO #{index}")
    print(f"AREA: {area} px2")
    for punto in contorno:
        print(f"{punto}")

cv2.imshow("Imagen Original", imagen_original)
cv2.imshow("Imagen Grises", imagen_grises)
cv2.imshow("Imagen Binaria", imagen_binaria)
cv2.waitKey(0)
cv2.destroyAllWindows()
