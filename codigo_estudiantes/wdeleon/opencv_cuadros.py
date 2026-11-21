import cv2 as cv
import numpy as np

nombre_archivo = "3_cuadrados.bmp"

imagen_original = cv.imread(nombre_archivo)

if imagen_original is None:
    raise FileExistsError(
        f"No se encontró {nombre_archivo}"
    )


#obtener dimensiones originales
altura_imagen, ancho_imagen, canales_color = imagen_original.shape
print(f"INF_IMAGEN: \n Altura: {altura_imagen} \n Ancho: {ancho_imagen} \n Canal de color: {canales_color}")

imagen_grises = cv.cvtColor(imagen_original, cv.COLOR_BGR2GRAY)

_, imagen_binaria = cv.threshold(imagen_grises, 127, 255, cv.THRESH_BINARY_INV,)

contornos, _ = cv.findContours(imagen_binaria, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE,)
print(contornos)


for index, contorno in enumerate(contornos):
    area = cv.contourArea(contorno)
    if area < 10000:
        continue

    x_pi, y_pi, ancho_caja, alto_caja = cv.boundingRect(contorno)

    cv.rectangle(imagen_original, (x_pi, y_pi), (x_pi + ancho_caja, y_pi + alto_caja), (123,255,77), 2)

    cv.drawContours(imagen_original, contornos, index, (0,255,0), 2)

    y_etiqueta_objeto = y_pi - 20 
    x_etiqueta_objeto = x_pi -10

    if y_etiqueta_objeto <= 0:
        y_etiqueta_objeto = y_pi + alto_caja + 10

    #objeto = index + 1
    cv.putText(imagen_original, f"Objeto No: {index}", (x_pi - 5, y_pi - 5), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
    cv.putText(imagen_original, f"Área: {area} pX2", ( x_pi - 10, y_pi + alto_caja + 20), cv.FONT_HERSHEY_PLAIN, 1, (255,0,255), 2)

    print(f"\n Contorno #{index}")
    print(f"\n Area: {area} px2")
    for punto in contorno:
        print(f"{punto}")
    print(f"\n Objetos marcaos: {index}")




cv.imshow("Imagen original", imagen_original)
cv.imshow("image binaria", imagen_binaria)

cv.waitKey(0)
cv.destroyAllWindows()