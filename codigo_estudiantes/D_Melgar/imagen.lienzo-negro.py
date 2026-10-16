import numpy as np
import cv2 as cv
"""""
#LIENZOS DE COLORES
alto, ancho = 300, 400
lienzo_negro = np.zeros((alto, ancho, 3), dtype=np.uint8)

lienzo_azul = np.zeros((alto, ancho, 3), dtype=np.uint8)
lienzo_azul[:] = (255,0,0)

lienzo_rojo = np.zeros((alto, ancho, 3), dtype=np.uint8)
lienzo_rojo[:] = (0,0,255)

lienzo_verde = np.zeros((alto, ancho, 3), dtype=np.uint8)
lienzo_verde[:] = (0,255,0)

cv.imshow("Lienzo negro", lienzo_negro)
cv.imshow("Lienzo azul", lienzo_azul)
cv.imshow("Lienzo rojo", lienzo_rojo)
cv.imshow("Lienzo verde", lienzo_verde)
cv.waitKey(0)       # Espera una tecla
cv.destroyAllWindows()
"""
"""""
#FIGURAS GEOMETRICAS
lienzo = np.zeros((300,400, 3), dtype=np.uint8)

cv.rectangle(lienzo, (50, 50), (200,200), (183,220,112), -1)

cv.circle(lienzo, (300,150), 50, (255,255,255), -1)

cv.imshow("lienzo", lienzo)
cv.waitKey(0)
cv.destroyAllWindows()

"""

#MODIFICAR REGIONES ESPECIFICAS (ROI)
alto, ancho = 300, 500
imagen =  np.zeros((alto, ancho, 3), dtype=np.uint8)

superior = imagen[0:300, 0:166]
superior[:] = [0, 255,0]
central = imagen[0:300, 166:333]
central[:] = [255,0,0]
inferior = imagen[0:300, 333:500]
inferior[:] = [0,0,255]
cv.imshow("Imagen con ROI modificado", imagen)
cv.waitKey(0)
cv.destroyAllWindows()

"""""
""""""
imagen = cv.imread("c:\\Users\\GP\\Downloads\\WhatsApp Image 2026-10-02 at 10.19.45 AM.jpeg")
gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
print(gris.shape)
if imagen is None:
    print("Error: No se pudo cargar la imagen. Revisa la ruta.")
else:
    
    alto, ancho, canales = imagen.shape
    print(f"Dimensiones de la imagen: {alto}x{ancho}, Canales: {canales}")
    print(f"Canales de color: {canales}")
    cv.imshow("Imagen cargada", imagen)
    cv.waitKey(0)
    cv.destroyAllWindows()
"""