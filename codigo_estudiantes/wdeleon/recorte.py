import cv2

imagen = cv2.imread("perro.jpg")

if imagen is None:
    raise FileNotFoundError("No se encontró perro.jpg")

print("Dimensiones", imagen.shape)

Recorte = imagen[150:750, 800:1300]

#Desempaquetando las dimensiones de la imagen
alto, ancho, canales = imagen.shape

RelacionAspecto = alto / ancho
NuevoAncho = 150
NuevoAlto = int(RelacionAspecto * NuevoAncho)
print(f"Nuevas medidas: {NuevoAlto, NuevoAncho}")

#Imagen reducida
Reduccion = cv2.resize(imagen,(NuevoAncho, NuevoAlto), interpolation=cv2.INTER_CUBIC)

#Imagen rotada
Rotacion = cv2.rotate(Reduccion, cv2.ROTATE_90_CLOCKWISE)
AltoReducido, AnchoReducid, _ = Reduccion.shape
Centro = (AnchoReducid//2, AltoReducido//2)
MatrizRotacion = cv2.getRotationMatrix2D(Centro, 90, 1)
ImagenRotada = cv2.warpAffine(Reduccion, MatrizRotacion,(AnchoReducid, AltoReducido))

#Anotacion de imagen
ImagenAnotada = Reduccion.copy()
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
#Escala de color de la imagen
ImgaenEscalaGris = cv2.cvtColor(ImagenAnotada, cv2.COLOR_BGR2YUV)

cv2.imwrite("imagen_anotada.jpg", ImagenAnotada)

#Mostrador de Imagnes
#cv2.imshow("Imagen Original", imagen)
#cv2.imshow("Imagen Recortada", Recorte)
cv2.imshow("Imagen Reducida", Reduccion)
#cv2.imshow("Imagen Rotada", Rotacion)
#cv2.imshow("Imagen Rotada 2", ImagenRotada)
#cv2.imshow("Imagen Anotada", ImagenAnotada)
#cv2.imshow("Imagen Filtro", ImgaenEscalaGris)

cv2.waitKey(0)
cv2.destroyAllWindows()