
import cv2 as cv
import numpy as np

# ==============================
# 1. CONFIGURACION DE LA CAMARA
# ==============================

camara = cv.VideoCapture(0)

if not camara.isOpened():
    print("Error: no se pudo abrir la camara.")
    exit()

# ==============================
# 2. PARAMETROS DE DETECCION
# ==============================

AREA_MINIMA = 1500
ANCHO_MINIMO = 35
ALTO_MINIMO = 35
MIN_PUNTOS = 8

# FAST
detector_fast = cv.FastFeatureDetector_create(
    threshold=35,
    nonmaxSuppression=True,
    type=cv.FAST_FEATURE_DETECTOR_TYPE_9_16
)

# Agrupacion de puntos cercanos
kernel = cv.getStructuringElement(
    cv.MORPH_ELLIPSE, (15, 15)
)

while True:

    exito, fotograma = camara.read()

    if not exito:
        print("Error al capturar el fotograma.")
        break

    # ==============================
    # 3. PREPROCESAMIENTO
    # ==============================

    gris = cv.cvtColor(fotograma, cv.COLOR_BGR2GRAY)

    gris = cv.GaussianBlur(gris, (3, 3), 0)

    alto_imagen, ancho_imagen = gris.shape

    # Mascara para combinar los tres detectores
    mascara = np.zeros(
        (alto_imagen, ancho_imagen),
        dtype=np.uint8
    )

    # ==============================
    # 4. DETECTOR FAST
    # ==============================

    puntos_fast = detector_fast.detect(gris, None)

    for punto in puntos_fast:
        x, y = punto.pt
        cv.circle(
            mascara,
            (int(x), int(y)),
            2, 255, -1
        )

    # ==============================
    # 5. DETECTOR HARRIS
    # ==============================

    harris = cv.cornerHarris(
        np.float32(gris),
        blockSize=2,
        ksize=3,
        k=0.04
    )

    # Normalizar respuestas para establecer un umbral
    harris_norm = cv.normalize(
        harris, None, 0, 255, cv.NORM_MINMAX
    ).astype(np.uint8)

    # Seleccionar las esquinas mas fuertes
    _, harris_binario = cv.threshold(
        harris_norm, 150, 255, cv.THRESH_BINARY
    )

    # Combinar puntos Harris con FAST
    mascara = cv.bitwise_or(
        mascara, harris_binario
    )

    # ==============================
    # 6. DETECTOR SHI-TOMASI
    # ==============================

    puntos_shi = cv.goodFeaturesToTrack(
        gris,
        maxCorners=300,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=5,
        useHarrisDetector=False
    )

    if puntos_shi is not None:
        puntos_shi = np.int32(puntos_shi)

        for punto in puntos_shi:
            x, y = punto.ravel()

            cv.circle(
                mascara,
                (int(x), int(y)),
                3, 255, -1
            )

    # ==============================
    # 7. AGRUPAR Y LIMPIAR PUNTOS
    # ==============================

    mascara = cv.dilate(
        mascara, kernel, iterations=2
    )

    mascara = cv.morphologyEx(
        mascara,
        cv.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    # ==============================
    # 8. BUSCAR REGIONES CANDIDATAS
    # ==============================

    contornos, _ = cv.findContours(
        mascara,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    numero_objetos = 0

    for contorno in contornos:

        area = cv.contourArea(contorno)

        if area < AREA_MINIMA:
            continue

        x, y, ancho, alto = cv.boundingRect(contorno)

        if ancho < ANCHO_MINIMO or alto < ALTO_MINIMO:
            continue

        # Contar puntos detectados dentro de la region
        region = mascara[y:y + alto, x:x + ancho]
        cantidad_puntos = cv.countNonZero(region)

        if cantidad_puntos < MIN_PUNTOS:
            continue

        # Dibujar rectangulo
        numero_objetos += 1

        cv.rectangle(
            fotograma,
            (x, y),
            (x + ancho, y + alto),
            (0, 255, 0),
            2
        )

        cv.putText(
            fotograma,
            f"Objeto {numero_objetos}",
            (x, max(y - 10, 20)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # ==============================
    # 9. INFORMACION EN PANTALLA
    # ==============================

    cv.putText(
        fotograma,
        f"FAST: {len(puntos_fast)}",
        (10, 25),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6, (255, 0, 255), 2
    )

    cantidad_shi = (
        0 if puntos_shi is None else len(puntos_shi)
    )

    cv.putText(
        fotograma,
        f"Shi-Tomasi: {cantidad_shi}",
        (10, 50),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6, (255, 0, 255), 2
    )

    cv.imshow("Objetos enmarcados", fotograma)
    cv.imshow("Puntos combinados", mascara)

    # ESC para salir
    if cv.waitKey(1) & 0xFF == 27:
        break

camara.release()
cv.destroyAllWindows()
