import numpy as np
import cv2
import time

# Configuración
CAMARA_ID = 0
AREA_MINIMA = 10000
UMBRAL = 100
MARGEN_CAJA = 10

# Configuración Harris
HARRIS_BLOCK_SIZE = 2
HARRIS_KSIZE = 3
HARRIS_K = 0.04
HARRIS_UMBRAL = 0.01

# Configuración Shi-Tomasi
MAX_CORNERS = 100
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 10
BLOCK_SIZE = 3

# Inicializar cámara
camara = cv2.VideoCapture(CAMARA_ID)

if not camara.isOpened():
    raise RuntimeError("No se pudo abrir la cámara web.")

# Solicitar 45 FPS (la cámara puede no admitirlos)
camara.set(cv2.CAP_PROP_FPS, 45)

fps_solicitados = camara.get(cv2.CAP_PROP_FPS)
print(f"FPS reportados por la cámara: {fps_solicitados}")

# Obtener dimensiones una sola vez
ancho = int(camara.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(camara.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Resolución: {ancho}x{alto}")

# Variables para medir FPS de procesamiento
contador_fotogramas = 0
tiempo_inicio = time.perf_counter()
fps_procesamiento = 0.0

try:
    while True:
        exito, fotograma = camara.read()

        if not exito:
            print("No se pudo capturar el fotograma.")
            break

        contador_fotogramas += 1
        tiempo_actual = time.perf_counter()
        tiempo_transcurrido = tiempo_actual - tiempo_inicio

        # Actualizar FPS aproximadamente una vez por segundo
        if tiempo_transcurrido >= 1.0:
            fps_procesamiento = (
                contador_fotogramas / tiempo_transcurrido
            )
            contador_fotogramas = 0
            tiempo_inicio = tiempo_actual

        # Convertir a escala de grises
        imagen_grises = cv2.cvtColor(
            fotograma, cv2.COLOR_BGR2GRAY
        )
        
        # Preparar imagen para Harris
        imagen_float = np.float32(imagen_grises)

        # Aplicar detector Harris
        respuesta_harris = cv2.cornerHarris(
            imagen_float,
            HARRIS_BLOCK_SIZE,
            HARRIS_KSIZE,
            HARRIS_K
        )

        # Dilatar para visualizar mejor las esquinas
        respuesta_harris = cv2.dilate(
            respuesta_harris,
            None
        )

        # Identificar esquinas mediante umbral relativo
        umbral_harris = (
            HARRIS_UMBRAL * respuesta_harris.max()
        )

        # Marcar esquinas Harris en rojo
        fotograma[
            respuesta_harris > umbral_harris
        ] = [0, 0, 255]
        
                # Detectar esquinas mediante Shi-Tomasi
        esquinas = cv2.goodFeaturesToTrack(
            imagen_grises,
            maxCorners=MAX_CORNERS,
            qualityLevel=QUALITY_LEVEL,
            minDistance=MIN_DISTANCE,
            blockSize=BLOCK_SIZE
        )

        # Dibujar esquinas detectadas
        if esquinas is not None:

            esquinas = np.int32(esquinas)

            for esquina in esquinas:

                x, y = esquina.ravel()

                cv2.circle(
                    fotograma,
                    (x, y),
                    5,
                    (255, 0, 0),
                    -1
                )

        # Crear imagen binaria invertida
        _, imagen_binaria = cv2.threshold(
            imagen_grises,
            UMBRAL,
            255,
            cv2.THRESH_BINARY_INV
        )
        
        

        # Encontrar contornos simplificados
        contornos, _ = cv2.findContours(
            imagen_binaria,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        numero_objetos = 0

        for contorno in contornos:
            area = cv2.contourArea(contorno)

            # Ignorar objetos demasiado pequeños
            if area < AREA_MINIMA:
                continue

            numero_objetos += 1

            x, y, w, h = cv2.boundingRect(contorno)

            # Limitar la caja a los bordes de la imagen
            x1 = max(0, x - MARGEN_CAJA)
            y1 = max(0, y - MARGEN_CAJA)
            x2 = min(ancho, x + w + MARGEN_CAJA)
            y2 = min(alto, y + h + MARGEN_CAJA)

            # Dibujar rectángulo
            cv2.rectangle(
                fotograma,
                (x1, y1),
                (x2, y2),
                (127, 255, 22),
                2
            )

            # Dibujar solo el contorno aceptado
            cv2.drawContours(
                fotograma,
                [contorno],
                -1,
                (0, 255, 0),
                1
            )

            # Etiquetas en dos líneas
            etiqueta_y = y - 35
            if etiqueta_y < 20:
                etiqueta_y = min(alto - 35, y + h + 20)

            cv2.putText(
                fotograma,
                f"Objeto #{numero_objetos}",
                (x, etiqueta_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            cv2.putText(
                fotograma,
                f"Area: {area:.0f} px^2",
                (x, min(alto - 5, etiqueta_y + 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1
            )

        # Mostrar información general
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
            f"Objetos: {numero_objetos}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # Mostrar imágenes
        cv2.imshow("Camara original", fotograma)
        cv2.imshow("Camara binaria", imagen_binaria)

        # Salir con ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    camara.release()
    cv2.destroyAllWindows()
