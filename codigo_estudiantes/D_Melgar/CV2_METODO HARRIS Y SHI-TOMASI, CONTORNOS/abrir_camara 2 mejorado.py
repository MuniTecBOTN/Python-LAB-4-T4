
import cv2
import numpy as np
import time


# =====================================================
# 1. CONFIGURACION GENERAL
# =====================================================

CAMARA_ID = 0

ANCHO_CAMARA = 640
ALTO_CAMARA = 480

# Configuracion Harris
HARRIS_BLOCK_SIZE = 2
HARRIS_KSIZE = 3
HARRIS_K = 0.04

HARRIS_UMBRAL = 0.01
GAUSSIAN_KERNEL = (5, 5)
GAUSSIAN_SIGMA = 1.0

# Supresion no maxima
NMS_KERNEL = 7
MAX_ESQUINAS = 150

# Shi-Tomasi (opcional)
ACTIVAR_SHI_TOMASI = False
MAX_CORNERS = 100
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 10
BLOCK_SIZE = 3

# Contornos
ACTIVAR_CONTORNOS = True
UMBRAL_BINARIO = 100
AREA_MINIMA = 10000
MARGEN_CAJA = 10

# Visualizacion
RADIO_ESQUINA = 3


# =====================================================
# 2. FUNCION: DETECTOR HARRIS
# =====================================================

def detectar_esquinas_harris(imagen_grises):
    """
    Detecta esquinas utilizando Harris y supresion
    no maxima.

    Retorna:
        respuesta: mapa de respuesta Harris.
        esquinas: coordenadas (x, y) de las esquinas.
    """

    # Suavizado Gaussiano para reducir ruido
    imagen_suavizada = cv2.GaussianBlur(
        imagen_grises,
        GAUSSIAN_KERNEL,
        GAUSSIAN_SIGMA
    )

    # Conversion a float32
    imagen_float = np.float32(imagen_suavizada)

    # Calculo de respuesta Harris
    respuesta = cv2.cornerHarris(
        imagen_float,
        HARRIS_BLOCK_SIZE,
        HARRIS_KSIZE,
        HARRIS_K
    )

    # Considerar solamente respuestas positivas
    respuesta_positiva = np.maximum(
        respuesta,
        0
    )

    maximo = respuesta_positiva.max()

    if maximo <= 0:
        return respuesta, np.empty((0, 2), dtype=np.int32)

    # Umbral relativo
    umbral = HARRIS_UMBRAL * maximo

    # Supresion no maxima
    kernel = np.ones(
        (NMS_KERNEL, NMS_KERNEL),
        dtype=np.uint8
    )

    respuesta_dilatada = cv2.dilate(
        respuesta_positiva,
        kernel
    )

    # Seleccionar maximos locales
    mascara = (
        (respuesta_positiva >= umbral) &
        (respuesta_positiva == respuesta_dilatada)
    )

    # Obtener coordenadas mediante NumPy
    coordenadas = np.argwhere(mascara)

    if len(coordenadas) == 0:
        return respuesta, np.empty((0, 2), dtype=np.int32)

    # Obtener respuestas correspondientes
    valores = respuesta_positiva[
        coordenadas[:, 0],
        coordenadas[:, 1]
    ]

    # Conservar las mejores MAX_ESQUINAS
    if len(coordenadas) > MAX_ESQUINAS:

        indices = np.argpartition(
            valores,
            -MAX_ESQUINAS
        )[-MAX_ESQUINAS:]

        coordenadas = coordenadas[indices]

    # Convertir de (y, x) a (x, y)
    esquinas = coordenadas[:, ::-1].astype(np.int32)

    return respuesta, esquinas


# =====================================================
# 3. FUNCION: DETECTOR SHI-TOMASI
# =====================================================

def detectar_esquinas_shi_tomasi(imagen_grises):
    """
    Detecta esquinas mediante Shi-Tomasi.
    """

    esquinas = cv2.goodFeaturesToTrack(
        imagen_grises,
        maxCorners=MAX_CORNERS,
        qualityLevel=QUALITY_LEVEL,
        minDistance=MIN_DISTANCE,
        blockSize=BLOCK_SIZE
    )

    if esquinas is None:
        return np.empty((0, 2), dtype=np.int32)

    return np.int32(
        np.round(esquinas.reshape(-1, 2))
    )


# =====================================================
# 4. FUNCION: DIBUJAR ESQUINAS
# =====================================================

def dibujar_esquinas(imagen, esquinas, color):
    """
    Dibuja las esquinas detectadas.
    """

    for x, y in esquinas:

        cv2.circle(
            imagen,
            (int(x), int(y)),
            RADIO_ESQUINA,
            color,
            -1
        )


# =====================================================
# 5. FUNCION: DETECCION DE CONTORNOS
# =====================================================

def detectar_contornos(imagen_grises, imagen_color):
    """
    Detecta regiones mediante umbralizacion binaria
    y calcula sus areas.
    """

    _, imagen_binaria = cv2.threshold(
        imagen_grises,
        UMBRAL_BINARIO,
        255,
        cv2.THRESH_BINARY_INV
    )

    contornos, _ = cv2.findContours(
        imagen_binaria,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    numero_objetos = 0

    alto, ancho = imagen_color.shape[:2]

    for contorno in contornos:

        area = cv2.contourArea(contorno)

        if area < AREA_MINIMA:
            continue

        numero_objetos += 1

        x, y, w, h = cv2.boundingRect(contorno)

        # Limitar coordenadas
        x1 = max(0, x - MARGEN_CAJA)
        y1 = max(0, y - MARGEN_CAJA)

        x2 = min(ancho, x + w + MARGEN_CAJA)
        y2 = min(alto, y + h + MARGEN_CAJA)

        # Dibujar rectangulo
        cv2.rectangle(
            imagen_color,
            (x1, y1),
            (x2, y2),
            (127, 255, 22),
            2
        )

        # Dibujar contorno
        cv2.drawContours(
            imagen_color,
            [contorno],
            -1,
            (0, 255, 0),
            1
        )

        # Posicion de etiqueta
        etiqueta_y = y - 35

        if etiqueta_y < 20:
            etiqueta_y = min(
                alto - 35,
                y + h + 20
            )

        cv2.putText(
            imagen_color,
            f"Objeto #{numero_objetos}",
            (x, etiqueta_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.putText(
            imagen_color,
            f"Area: {area:.0f} px^2",
            (x, min(alto - 5, etiqueta_y + 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1
        )

    return imagen_binaria, numero_objetos


# =====================================================
# 6. FUNCION: VISUALIZACION DE RESPUESTA HARRIS
# =====================================================

def visualizar_respuesta_harris(imagen, respuesta):
    """
    Normaliza la respuesta Harris para visualizarla.
    """

    respuesta_normalizada = cv2.normalize(
        respuesta,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    respuesta_normalizada = np.uint8(
        respuesta_normalizada
    )

    mapa_color = cv2.applyColorMap(
        respuesta_normalizada,
        cv2.COLORMAP_JET
    )

    return mapa_color


# =====================================================
# 7. INICIALIZAR CAMARA
# =====================================================

def iniciar_camara():

    camara = cv2.VideoCapture(CAMARA_ID)

    if not camara.isOpened():
        raise RuntimeError(
            "No se pudo abrir la camara web."
        )

    camara.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        ANCHO_CAMARA
    )

    camara.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        ALTO_CAMARA
    )

    ancho_real = int(
        camara.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    alto_real = int(
        camara.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps_real = camara.get(
        cv2.CAP_PROP_FPS
    )

    print("Camara inicializada.")
    print(f"Resolucion: {ancho_real}x{alto_real}")
    print(f"FPS reportados: {fps_real}")

    return camara


# =====================================================
# 8. PROGRAMA PRINCIPAL
# =====================================================

def main():

    camara = iniciar_camara()

    contador_fotogramas = 0
    tiempo_inicio = time.perf_counter()

    fps_procesamiento = 0.0

    try:

        while True:

            inicio_frame = time.perf_counter()

            # Capturar fotograma
            exito, fotograma = camara.read()

            if not exito:
                print("No se pudo capturar el fotograma.")
                break

            # Obtener dimensiones reales
            alto, ancho = fotograma.shape[:2]

            # Conversion a escala de grises
            imagen_grises = cv2.cvtColor(
                fotograma,
                cv2.COLOR_BGR2GRAY
            )

            # -----------------------------------------
            # DETECCION HARRIS
            # -----------------------------------------

            respuesta_harris, esquinas_harris = (
                detectar_esquinas_harris(
                    imagen_grises
                )
            )

            # Marcar esquinas Harris en rojo
            dibujar_esquinas(
                fotograma,
                esquinas_harris,
                (0, 0, 255)
            )

            # -----------------------------------------
            # DETECCION SHI-TOMASI OPCIONAL
            # -----------------------------------------

            if ACTIVAR_SHI_TOMASI:

                esquinas_shi = (
                    detectar_esquinas_shi_tomasi(
                        imagen_grises
                    )
                )

                dibujar_esquinas(
                    fotograma,
                    esquinas_shi,
                    (255, 0, 0)
                )

            # -----------------------------------------
            # DETECCION DE CONTORNOS
            # -----------------------------------------

            numero_objetos = 0

            if ACTIVAR_CONTORNOS:

                imagen_binaria, numero_objetos = (
                    detectar_contornos(
                        imagen_grises,
                        fotograma
                    )
                )

                cv2.imshow(
                    "Imagen binaria",
                    imagen_binaria
                )

            # -----------------------------------------
            # CALCULAR FPS Y TIEMPO
            # -----------------------------------------

            contador_fotogramas += 1

            tiempo_actual = time.perf_counter()

            tiempo_transcurrido = (
                tiempo_actual - tiempo_inicio
            )

            if tiempo_transcurrido >= 1.0:

                fps_procesamiento = (
                    contador_fotogramas /
                    tiempo_transcurrido
                )

                contador_fotogramas = 0
                tiempo_inicio = tiempo_actual

            tiempo_frame = (
                time.perf_counter() - inicio_frame
            ) * 1000

            # -----------------------------------------
            # INFORMACION EN PANTALLA
            # -----------------------------------------

            cv2.putText(
                fotograma,
                f"FPS: {fps_procesamiento:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                fotograma,
                f"Esquinas Harris: {len(esquinas_harris)}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.putText(
                fotograma,
                f"Objetos: {numero_objetos}",
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.putText(
                fotograma,
                f"Tiempo: {tiempo_frame:.1f} ms",
                (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            # -----------------------------------------
            # MOSTRAR RESULTADOS
            # -----------------------------------------

            cv2.imshow(
                "Detector Harris Optimizado",
                fotograma
            )

            # Mapa de respuesta Harris
            mapa_harris = visualizar_respuesta_harris(
                fotograma,
                respuesta_harris
            )

            cv2.imshow(
                "Respuesta Harris",
                mapa_harris
            )

            # Salir con ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:

        camara.release()
        cv2.destroyAllWindows()

        print("Sistema finalizado correctamente.")


# =====================================================
# 9. EJECUCION
# =====================================================

if __name__ == "__main__":

    try:
        main()

    except (RuntimeError, cv2.error) as error:
        print(f"Error del sistema: {error}")
