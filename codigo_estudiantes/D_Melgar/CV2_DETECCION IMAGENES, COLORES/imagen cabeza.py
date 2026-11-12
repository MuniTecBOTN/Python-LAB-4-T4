import cv2

imagen = cv2.imread("images.webp")

if imagen is None:
    raise FileNotFoundError("No se encontró images.webp")

print("Dimensiones:", imagen.shape)

recorte = imagen[100:300, 0:185]

alto, ancho, canales = imagen.shape

reduccion = cv2.resize(imagen, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)

rotacion = cv2.getRotationMatrix2D((ancho // 2, alto // 2), 45, 1.0)
rotacion = cv2.warpAffine(imagen, rotacion, (ancho, alto))

anotacion = reduccion.copy()

cv2.rectangle(anotacion, (90, 10), (300, 324), (0, 255, 0), 2)
cv2.circle(anotacion, (195, 167), 85, (255, 0, 0), 2)
cv2.putText(anotacion, "CABEZA", (90, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

cv2.imwrite("bert_anotada.jpg", anotacion)

cv2.imshow("Imagen original", imagen)
cv2.imshow("Region recortada", recorte)
cv2.imshow("Imagen reducida", reduccion)
cv2.imshow("Imagen rotada", rotacion)
cv2.imshow("Imagen con anotacion", anotacion)
cv2.imshow("Imagen anotada en escala de grises", gris)

cv2.waitKey(0)
cv2.destroyAllWindows()