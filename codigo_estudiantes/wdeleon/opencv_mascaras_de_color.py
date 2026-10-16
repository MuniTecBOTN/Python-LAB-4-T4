import cv2 as cv
import numpy as np

ImagenOriginal = cv.imread("manzanas.jpg")

if ImagenOriginal is None:
    raise FileNotFoundError("No se encontró manzanas.jpg")

Alto, Ancho, _ = ImagenOriginal.shape

print(f"Medidas Originales: \n Alto: {Alto} \n Ancho: {Ancho}")

RelacionAspecto = Alto/Ancho
AnchoNuevo = 800
AltoNuevo = int(RelacionAspecto * AnchoNuevo)

print(f"Medidas Redimencionadas:  \n ALto: {AltoNuevo} \n Ancho: {AnchoNuevo}")

ImagenRedimencionada = cv.resize(ImagenOriginal, (AnchoNuevo, AltoNuevo), interpolation=cv.INTER_CUBIC)

ImagenHSV = cv.cvtColor(ImagenRedimencionada, cv.COLOR_BGR2HSV)

LimiteInferior = np.array([170, 100, 120])
LimiteSuperior = np.array([179, 255, 255])

Mascara = cv.inRange(ImagenHSV, LimiteInferior, LimiteSuperior)

ImagenFiltrada = cv.bitwise_and(ImagenRedimencionada, ImagenRedimencionada, mask=Mascara)

print(f"Rango de HSV: \n Limite Inferior: \n H: {LimiteInferior[0]}        S:{LimiteInferior[1]}       V: {LimiteInferior[2]} \n Limite Superior: \n H: {LimiteSuperior[0]}        S: {LimiteSuperior[1]}      V:{LimiteSuperior[2]}")

ResultadoFondo = np.full_like(ImagenRedimencionada, (80,150,120))
ResultadoFondo[Mascara > 0] = ImagenRedimencionada[Mascara > 0]

print(f"Codigo Fondo (BGR): \n B: {ResultadoFondo[0,0,0]}        G: {ResultadoFondo[0,0,1]}        R: {ResultadoFondo[0,0,2]}")

#cv.imshow("Imagen Redimencionada", ImagenRedimencionada)
cv.imshow("Imagen con HSV", ImagenHSV)
cv.imshow("Imagen Mascara", Mascara)
cv.imshow("Imagen Filtrada", ImagenFiltrada)
cv.imshow("Imagen Fondo", ResultadoFondo)

cv.waitKey(0)
cv.destroyAllWindows()