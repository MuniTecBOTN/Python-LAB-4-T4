import cv2
import numpy as np

nombre_archivo = "3_cuadrados.bmp"
fotograma = cv2.imread(nombre_archivo)

if fotograma is None:
    raise FileNotFoundError(
        f"No se encontró {nombre_archivo}"
    )

altura_imagen, ancho_imagen, canales_color = fotograma.shape
print(f"INF_IMAGEN: {altura_imagen}x{ancho_imagen} - {canales_color}")

imagen_grises = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)

_, imagen_binaria = cv2.threshold(
    imagen_grises,
    127,
    255,
    cv2.THRESH_BINARY
)


contornos, _ = cv2.findContours(
    imagen_binaria,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)
escala_fuente = min(ancho_imagen, altura_imagen) * 0.001
grosor_fuente = max(1, round(escala_fuente * 2))

for index, contorno in enumerate(contornos):
    area = cv2.contourArea(contorno)
    if area < 10000:
        continue
    x_pi, y_pi, ancho_caja, alto_caja = cv2.boundingRect(contorno)

    
    cv2.rectangle(
        fotograma,
        (x_pi-10, y_pi-10),
        (x_pi + ancho_caja + 10, y_pi + alto_caja + 10),
        (123, 255, 77),
        2,
    )
    
    cv2.drawContours(
        fotograma,
        contornos,
        index,
        (255, 26, 200),
        1
    )

    y_etiqueta_objeto = y_pi - 20
    x_etiqueta_objeto = x_pi - 10
    if y_etiqueta_objeto <= 0:
        y_etiqueta_objeto = y_pi + alto_caja + 45
    cv2.putText(
        fotograma,
        f"CONTORNO #{index}",
        (x_etiqueta_objeto, y_etiqueta_objeto),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 26, 200),
        1,
    )


    etiqueta_area = f"AREA: {area:.2f} px2"
    posicion_area = (x_pi, y_pi + alto_caja + 25)
    cv2.putText(
        fotograma,
        etiqueta_area,
        posicion_area,
        cv2.FONT_HERSHEY_SIMPLEX,
        escala_fuente,
        (255, 26, 200),
        grosor_fuente,
    )
    
    print(f"\nCONTORNO #{index}")
    print(f"AREA: {area} px2")
    for punto in contorno:
        print(f"{punto}")

cv2.imshow("Imagen Original", fotograma)
cv2.imshow("Imagen en Grises", imagen_grises)
cv2.imshow("Imagen Binaria", imagen_binaria)

cv2.waitKey(0)
cv2.destroyAllWindows()