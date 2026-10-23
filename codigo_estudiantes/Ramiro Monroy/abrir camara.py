
import cv2

# ==============================
# CONFIGURACION
# ==============================
CAMARA_ID = 0
UMBRAL = 127
AREA_MINIMA = 10000
ANCHO_MAXIMO = 800

# Activar la camara
camara = cv2.VideoCapture(CAMARA_ID)

if not camara.isOpened():
    raise RuntimeError("No se pudo abrir la camara web")

# Reducir latencia cuando el controlador lo permite
camara.set(cv2.CAP_PROP_BUFFERSIZE, 1)

try:
    while True:
        # Capturar fotograma
        exito, fotograma = camara.read()

        if not exito:
            print("Error al capturar la imagen")
            break

        # Reducir resolucion para mejorar rendimiento
        alto, ancho = fotograma.shape[:2]

        if ancho > ANCHO_MAXIMO:
            alto_nuevo = int(alto * ANCHO_MAXIMO / ancho)
            fotograma = cv2.resize(
                fotograma,
                (ANCHO_MAXIMO, alto_nuevo),
                interpolation=cv2.INTER_AREA
            )

        altura, anchura = fotograma.shape[:2]

        # Crear imagen de trabajo en escala de grises
        gris = cv2.cvtColor(
            fotograma,
            cv2.COLOR_BGR2GRAY
        )

        # Aplicar un desenfoque leve para reducir ruido
        gris = cv2.GaussianBlur(
            gris,
            (5, 5),
            0
        )

        # Umbral binario invertido
        _, binaria = cv2.threshold(
            gris,
            UMBRAL,
            255,
            cv2.THRESH_BINARY_INV
        )

        # Buscar contornos externos
        contornos, _ = cv2.findContours(
            binaria,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        numero_objeto = 0

        for contorno in contornos:
            # Filtrar antes de calcular el rectangulo
            area = cv2.contourArea(contorno)

            if area <= AREA_MINIMA:
                continue

            x, y, w, h = cv2.boundingRect(contorno)
            numero_objeto += 1

            # Dibujar rectangulo y contorno
            cv2.rectangle(
                fotograma,
                (x, y),
                (x + w, y + h),
                (123, 255, 77),
                2
            )

            cv2.drawContours(
                fotograma,
                [contorno],
                -1,
                (255, 255, 0),
                1
            )

            # Etiqueta con numero y area
            texto = f"OBJETO {numero_objeto} | Area: {area:.0f} px2"

            # Evitar que el texto salga por encima de la imagen
            y_texto = y - 10 if y >= 25 else min(altura - 5, y + h + 20)

            cv2.putText(
                fotograma,
                texto,
                (x, y_texto),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 26, 200),
                2,
                cv2.LINE_AA
            )

        # Mostrar ambas ventanas
        cv2.imshow("Camara - Deteccion de objetos", fotograma)
        cv2.imshow("Imagen binaria", binaria)

        # ESC para salir
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # Liberar recursos incluso si ocurre un error
    camara.release()
    cv2.destroyAllWindows()
