import cv2
import numpy as np

# Cargar la imagen
imagen = cv2.imread("3_cuadrados.bmp")
if imagen is not None:

    # Convertir a escala de grises
    gris = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2GRAY
    )

    # Detectar esquinas con Shi-Tomasi
    esquinas = cv2.goodFeaturesToTrack(
        gris,
        maxCorners=100,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=3,
        useHarrisDetector=False
    )

    # Copiar la imagen original
    resultado = imagen.copy()

    # Dibujar las esquinas detectadas
    if esquinas is not None:

        esquinas = esquinas.astype(int)

        for esquina in esquinas:
            x, y = esquina.ravel()

            cv2.circle(
                resultado,
                (x, y),
                5,
                (0, 0, 255),
                -1
            )

        print("Esquinas detectadas:", len(esquinas))

    else:
        print("No se detectaron esquinas.")

    # Mostrar los resultados
    cv2.imshow("Imagen original", imagen)
    cv2.imshow("Esquinas Shi-Tomasi", resultado)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("No se pudo cargar la imagen.")