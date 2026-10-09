import cv2 as cv

imagen = cv.imread("perro.jpg")

if imagen is None:
    raise FileNotFoundError("No se encontró perro.jpg")

alto, ancho, canales = imagen.shape

print("Dimenciones", imagen.shape)
print("")
print("""0 = Original
1 = Recorte
2 = Redución
3 = Rotación""")
print("")
TipoEdicion = int(input("Tipo de edición a realizar: "))


if TipoEdicion == 1:
    RecorteY1 = int(input("Ingrese el valor de Y1: "))
    RecorteY2 = int(input("Ingrese el valor de Y2: "))
    print("")
    RecorteX1 = int(input("Ingrese el valor de X1: "))
    RecorteX2 = int(input("Ingrese el valor de X2: "))

    RecorteImagen = imagen[RecorteY1:RecorteY2, RecorteX1:RecorteX2]

    cv.imshow("Imagen Perro Recortada", RecorteImagen)
    cv.waitKey(0)
    cv.destroyAllWindows()
elif TipoEdicion == 2:

    ReduccionImagenRango = float(input("Ingrese el valor de reducción: "))
    ReduccionImagen = cv.resize(imagen, (int(ancho*ReduccionImagenRango), int(alto*ReduccionImagenRango)), interpolation=cv.INTER_CUBIC)

    cv.imshow("Imagen Perro Reduccion", ReduccionImagen)
    cv.waitKey(0)
    cv.destroyAllWindows()    
elif TipoEdicion == 3:

    CentroImagen = (ancho//2, alto//2)

    RotacionImagenRango = int(input("Ingrese los grados a rotar: "))
    MatrizRotacion = cv.getRotationMatrix2D(CentroImagen, RotacionImagenRango, 1)
    RotacionImagen = cv.warpAffine(imagen, MatrizRotacion,(ancho, alto))
    
    cv.imshow("Imagen Perro Rotacion", RotacionImagen)
    cv.waitKey(0)
    cv.destroyAllWindows()
elif TipoEdicion == 0:

    cv.imshow("Imagen Perro Original", imagen)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("Dato ingresado invalido, buelva a intentarlo")