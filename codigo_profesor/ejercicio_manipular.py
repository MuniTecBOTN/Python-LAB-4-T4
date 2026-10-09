import cv2

try:
    coordenada_x_inicial = int(input("Ingrese la coordenada x inicial: "))
    coordenada_y_inicial = int(input("Ingrese la coordenada y inicial: "))

    coordenada_x_final = int(input("Ingrese la coordenada x final: "))
    coordenada_y_final = int(input("Ingrese la coordenada y final: "))
except ValueError:
    print("Error: Debe ingresar valores enteros válidos para las coordenadas.")
    exit()
    
    
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
    imagen_reducida, matriz_rotacion, (ancho_reducida, alto_reducida)
)

cv2.imshow("Imagen original", imagen)
cv2.imshow("Region recortada", imagen_recortada)
cv2.imshow("Imagen reducida", imagen_reducida)
cv2.imshow("Imagen rotada", imagen_rotada)


cv2.waitKey(0)
cv2.destroyAllWindows()
