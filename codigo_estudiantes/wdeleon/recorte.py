import cv2

imagen = cv2.imread("perro.jpg")

if imagen is None:
    raise FileNotFoundError("No se encontró perro.jpg")

print("Dimensiones", imagen.shape)

recorte = imagen[150:750, 800:1300]

#Desempaquetando las dimensiones de la imagen
alto, ancho, canales = imagen.shape

#Imagen reducida
reduccion = cv2.resize(imagen,( ancho //2, alto//2), interpolation=cv2.INTER_CUBIC)

#Imagen rotada
rotacion = cv2.rotate(reduccion, cv2.ROTATE_90_CLOCKWISE)

AltoReducido, AnchoReducid, _ = reduccion.shape

centro = (AnchoReducid//2, AltoReducido//2)

MatrizRotacion = cv2.getRotationMatrix2D(centro, 90, 1)

ImagenRotada = cv2.warpAffine(reduccion, MatrizRotacion,(AnchoReducid, AltoReducido))

ImagenAnotada = reduccion.copy()
cv2.rectangle(
    ImagenAnotada, 
    (400,100),
    (650,600),
    (0, 255, 0), 
    2
)

cv2.circle(
    ImagenAnotada, 
    (520, 235),
    100,
    (0, 255, 0),
    2 
)

cv2.putText(
    ImagenAnotada,
    "Objeto marcado",
    (AnchoReducid//2 - 115, AltoReducido//2 - 245),
    cv2.FONT_HERSHEY_COMPLEX,
    1,
    (0, 255, 0),
    2

)

ImgaenEscalaGris = cv2.cvtColor(ImagenAnotada, cv2.COLOR_BGR2YUV_Y422)

cv2.imwrite("imagen_anotada.jpg", ImagenAnotada)

#cv2.imshow("Imagen Original", imagen)
#cv2.imshow("Imagen Recortada", recorte)
#cv2.imshow("Imagen Reducida", reduccion)
#cv2.imshow("Imagen Rotada", rotacion)
#cv2.imshow("Imagen Rotada 2", ImagenRotada)
cv2.imshow("Imagen Anotada", ImagenAnotada)
cv2.imshow("Imagen Filtro", ImgaenEscalaGris)

cv2.waitKey(0)
cv2.destroyAllWindows()