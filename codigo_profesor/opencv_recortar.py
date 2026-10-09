import cv2

imagen = cv2.imread("mrrobot.jpg")

if imagen is None:
    raise FileNotFoundError("No se encontró mrrobot.jpg")

print("Dimensiones:", imagen.shape)

# Recortar una región de la imagen
imagen_recortada = imagen[750:1250, 1660:2170]

# Desempaquetando las dimensiones de la imagen
alto, ancho, canales = imagen.shape


# Imagen reducida
imagen_reducida = cv2.resize(
    imagen,
    None,
    fx=0.2,
    fy=0.2,
    interpolation=cv2.INTER_AREA,
)

imagen_rotada = cv2.rotate(imagen_reducida, cv2.ROTATE_90_CLOCKWISE)

alto_reducida, ancho_reducida, _ = imagen_reducida.shape

matriz_rotacion = cv2.getRotationMatrix2D(
    (ancho_reducida // 2, alto_reducida // 2),
    45,
    1,
)

imagen_rotada = cv2.warpAffine(
    imagen_reducida,
    matriz_rotacion,
    (ancho_reducida, alto_reducida)
)

imagen_anotada = imagen_reducida.copy()

cv2.rectangle(
    imagen_anotada,
    (ancho_reducida//2-100, alto_reducida//2-100),
    (ancho_reducida//2+100, alto_reducida//2+100),
    (255, 26, 200),  # BGR color format for blue
    2
)

cv2.circle(
    imagen_anotada,
    (ancho_reducida//2, alto_reducida//2),
    100,
    (0, 0, 255),  # BGR color format for red
    2
)

cv2.putText(
    imagen_anotada,
    "PERSONA",
    (ancho_reducida//2 - 102, alto_reducida//2 - 101),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255, 26, 200),  # BGR color format for green
    2
)

imagen_filtrada = cv2.cvtColor(imagen_anotada, cv2.COLOR_BGR2LAB)

imagen_canny = cv2.Canny(imagen_anotada, 100, 200)

cv2.imwrite("imagen_anotada.jpg", imagen_anotada)

#cv2.imshow("Imagen original", imagen)
#cv2.imshow("Region recortada", imagen_recortada)
#cv2.imshow("Imagen reducida", imagen_reducida)
#cv2.imshow("Imagen rotada", imagen_rotada)
cv2.imshow("Imagen anotada", imagen_anotada)
cv2.imshow("Imagen HSV", imagen_filtrada)
cv2.imshow("Imagen Canny", imagen_canny)



cv2.waitKey(0)
cv2.destroyAllWindows()
