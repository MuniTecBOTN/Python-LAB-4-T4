import cv2

imagen = cv2.imread("mrrobot.jpg")

if imagen is None:
    raise FileNotFoundError("No se encontró mrrobot.jpg")

print("Dimensiones:", imagen.shape)

# Desempaquetando las dimensiones de la imagen
alto, ancho, canales = imagen.shape

relacion_aspecto = alto / ancho
nuevo_ancho = 600
nuevo_alto = int(relacion_aspecto * nuevo_ancho)

# Imagen reducida
imagen_reducida = cv2.resize(
    imagen,
    (nuevo_ancho, nuevo_alto),
    interpolation=cv2.INTER_AREA,
)

print("Dimensiones de la imagen reducida:", imagen_reducida.shape)

cv2.imshow("IMAGEN REDUCIDA", imagen_reducida)

seccion_recortada = imagen_reducida[110:210, 230:370]

alto_seccion, ancho_seccion,_ = seccion_recortada.shape

matriz_rotacion = cv2.getRotationMatrix2D(
    (ancho_seccion // 2, alto_seccion // 2),
    5,
    1,
)
imagen_rotada = cv2.warpAffine(
    seccion_recortada,
    matriz_rotacion,
    (ancho_seccion, alto_seccion)
)


copia_imagen = imagen_reducida.copy()

cv2.rectangle(
    copia_imagen,
    (230, 110),
    (370, 210),
    (255, 26, 200),  # BGR color format for blue
    4
)

cv2.imshow("IMAGEN ANOTADA", copia_imagen)


cv2.waitKey(0)
cv2.destroyAllWindows()
