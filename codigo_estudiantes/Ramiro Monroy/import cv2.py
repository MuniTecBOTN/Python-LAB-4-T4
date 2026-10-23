import cv2
import numpy as np
nombre_archivo = "1_cuadrados.bmp"
imagen_original = cv2.imread(nombre_archivo)

if imagen_original is None:
    raise FileNotFoundError(
        f"No se encontro {nombre_archivo}"
    )

# obtener dimensiones originales 
altura_imagen, ancho_imagen, canales_color = imagen_original.shape
print(f"INF_IMAGEN: {altura_imagen}x{ancho_imagen}-{canales_color}")

imagen_grises = cv2.cvtcolor(imagen_original, cv2.COLOR_BAYER_BG2GRAY)
_, imagen_binaria = cv2.threshold(
    imagen_grises,
    127,
    255,
    cv2.THRESH_BINARY,
)

contornos, _= cv2.findcontours(
    imagen_binaria,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

for index, contorno in enumerate(contornos):
    area = cv2.contourArea(contorno)
    x_pi, y_pi, ancho_caja, alto_caja = cv2.boundingRect(contorno)

    cv2.rectangle(
        imagen_original,
        (x_pi, y_pi),
        (x_pi + ancho_caja, y_pi + alto_caja)
        (123, 255, 77),
        2
    )

    cv2.drawContours(
        imagen_original,
        contornos,
        index,
        (255, 255, 0),
        1
    )



    print(f"\nCONTORNO #{index}")
    print(f"AREA: {area} px2")
    for punto in contorno:
        print(f"{punto}")

cv2.imshow("Imagen Original", imagen_original)


cv2.waitKey(0)
cv2.destroyAllWindows
