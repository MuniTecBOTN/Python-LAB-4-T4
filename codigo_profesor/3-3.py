
import cv2

ruta = input("Ingrese la ruta de la imagen: ")
imagen = cv2.imread(ruta)

if imagen is None:
    print("Error: no se pudo cargar la imagen.")
else:
    alto, ancho = imagen.shape[:2]
    canales = (
        imagen.shape[2]
        if len(imagen.shape) == 3
        else 1
    )

    print("Ancho:", ancho, "pixeles")
    print("Alto:", alto, "pixeles")
    print("Canales:", canales)

    # Procesamiento
    gris = cv2.cvtColor(
        imagen, cv2.COLOR_BGR2GRAY
    )

    desenfocada = cv2.GaussianBlur(
        gris, (5, 5), 0
    )

    bordes = cv2.Canny(
        desenfocada, 50, 150
    )

    # Visualización
    cv2.imshow("Original", imagen)
    cv2.imshow("Escala de grises", gris)
    cv2.imshow("Desenfoque", desenfocada)
    cv2.imshow("Bordes", bordes)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
