import cv2
import numpy as np

nombre_archivo = "3_cuadrados.bmp"

imagen_original = cv2.imread(nombre_archivo)

if imagen_original is None:
    raise FileNotFoundError(f"No se encontró {nombre_archivo}")

# Obtener dimensiones originales
altura_imagen, ancho_imagen, canales_color = imagen_original.shape


imagen_grises = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)

_, imagen_binaria = cv2.threshold(imagen_grises, 127, 255, cv2.THRESH_BINARY)

contornos, jerarquia = cv2.findContours(
    imagen_grises, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
print(f"INF_IMAGEN:\n{altura_imagen}x{ancho_imagen}")

for index, contorno in enumerate(contornos):

    area_contorno = cv2.contourArea(contorno)

    if area_contorno < 10:
        continue
    else:
        x_inicial, y_inicial, ancho_caja, alto_caja = cv2.boundingRect(contorno)

        cv2.rectangle(
            imagen_original,
            (x_inicial - 20, y_inicial - 20),
            (x_inicial + ancho_caja + 20, y_inicial + alto_caja + 20),
            (255, 26, 200),
            2,
        )

        cv2.putText(
            imagen_original,
            f"OBJETO #{index}\nAREA:{area_contorno}",
            (x_inicial - 20, y_inicial - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 26, 200),  # BGR color format for green
            2,
        )

    # cv2.drawContours(imagen_original, contornos, index, (0, 255, 0), 2)

    print(f"\nContorno {index}")
    for punto in contorno:
        print(f"{punto}")

cv2.imshow("Imagen binaria", imagen_original)

cv2.waitKey(0)
cv2.destroyAllWindows()
