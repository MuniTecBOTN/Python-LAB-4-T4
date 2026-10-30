import cv2

imagen = cv2.imread("bert.jpg")

if imagen is None:
    raise FileNotFoundError("No se encontró bert.jpg")

print("Dimensiones:", imagen.shape)

y_inicio = int(input("Fila inicial del recorte: "))
y_fin = int(input("Fila final del recorte: "))
x_inicio = int(input("Columna inicial del recorte: "))
x_fin = int(input("Columna final del recorte: "))

recorte = imagen[y_inicio:y_fin, x_inicio:x_fin]

alto, ancho, canales = imagen.shape

escala = float(input("Factor de escala: "))
reduccion = cv2.resize(imagen, None, fx=escala, fy=escala,interpolation=cv2.INTER_AREA)

angulo = float(input("Ángulo de rotación: "))
matriz = cv2.getRotationMatrix2D((ancho // 2, alto // 2), angulo, 1.0)
rotacion = cv2.warpAffine(imagen, matriz, (ancho, alto))

cv2.imshow("Imagen original", imagen)
cv2.imshow("Region recortada", recorte)
cv2.imshow("Imagen reducida", reduccion)
cv2.imshow("Imagen rotada", rotacion)

cv2.waitKey(0)
cv2.destroyAllWindows()