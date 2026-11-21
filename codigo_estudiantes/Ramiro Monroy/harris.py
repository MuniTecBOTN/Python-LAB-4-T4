
import cv2
import numpy as np

# Cargar imagen
imagen = cv2.imread("figuras.jpg")

if imagen is None:
    raise FileNotFoundError(
        "No se encontro figuras.jpg. "
        "Coloca la imagen en la carpeta del programa."
    )

gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# =====================================
# 1. DETECTOR HARRIS
# =====================================
gris_float = np.float32(gris)

respuesta_harris = cv2.cornerHarris(
    gris_float,
    blockSize=2,
    ksize=3,
    k=0.04
)

# Dilatar para visualizar mejor las respuestas
respuesta_harris = cv2.dilate(
    respuesta_harris,
    None
)

resultado_harris = imagen.copy()

# Marcar las esquinas fuertes en rojo
umbral = 0.01 * respuesta_harris.max()

resultado_harris[respuesta_harris > umbral] = [
    0, 0, 255
]

# =====================================
# 2. DETECTOR SHI-TOMASI
# =====================================
esquinas = cv2.goodFeaturesToTrack(
    gris,
    maxCorners=100,
    qualityLevel=0.01,
    minDistance=10,
    blockSize=3,
    useHarrisDetector=False
)

resultado_shi = imagen.copy()

if esquinas is not None:
    esquinas = np.int32(esquinas)

    for punto in esquinas:
        x, y = punto.ravel()

        cv2.circle(
            resultado_shi,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )

# Mostrar resultados
cv2.imshow("Imagen original", imagen)
cv2.imshow("Detector Harris", resultado_harris)
cv2.imshow("Detector Shi-Tomasi", resultado_shi)

cv2.waitKey(0)
cv2.destroyAllWindows()
