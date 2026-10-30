
import cv2
import numpy as np
objeto = "telefono encendido"
# Cargar imagen de referencia
plantilla = cv2.imread(f"{objeto}.jpg")
#plantilla = cv2.resize(plantilla, (640, 480))

if plantilla is None:
    raise FileNotFoundError(f"No se encontro {objeto}.jpg")

# Convertir plantilla a escala de grises
plantilla_gris = cv2.cvtColor(
    plantilla, cv2.COLOR_BGR2GRAY
)

# Crear detector ORB
orb = cv2.ORB_create(
    nfeatures=1000,
    scaleFactor=1.2,
    nlevels=8
)

# Detectar puntos caracteristicos de la plantilla
puntos_plantilla, descriptores_plantilla = orb.detectAndCompute(
    plantilla_gris, None
)

if descriptores_plantilla is None:
    raise ValueError("No se encontraron caracteristicas en la plantilla.")

# Dibujar puntos caracteristicos de la plantilla
plantilla_puntos = cv2.drawKeypoints(
    plantilla,
    puntos_plantilla,
    None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
)

# Crear comparador de descriptores
comparador = cv2.BFMatcher(cv2.NORM_HAMMING)

# Obtener dimensiones de la plantilla
alto, ancho = plantilla_gris.shape

# Definir esquinas de la plantilla
esquinas = np.float32([
    [0, 0],
    [ancho - 1, 0],
    [ancho - 1, alto - 1],
    [0, alto - 1]
]).reshape(-1, 1, 2)

# Activar camara
camara = cv2.VideoCapture(1)

if not camara.isOpened():
    raise RuntimeError("No se pudo abrir la camara.")

# Configurar resolucion
camara.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Parametros de deteccion
umbral_coincidencia = 0.75
minimo_coincidencias = 10
minimo_inliers = 8

try:

    while True:

        # Capturar fotograma
        exito, fotograma = camara.read()
        

        if not exito:
            print("No se pudo capturar el fotograma.")
            break
        
        fotograma = cv2.resize(fotograma, (640, 480))


        # Crear copia para mostrar el fotograma_intermedio
        fotograma_intermedio = fotograma.copy()

        # Convertir fotograma a escala de grises
        fotograma_gris = cv2.cvtColor(
            fotograma, cv2.COLOR_BGR2GRAY
        )

        # Detectar puntos caracteristicos
        puntos_fotograma, descriptores_fotograma = orb.detectAndCompute(
            fotograma_gris, None
        )

        # Mostrar puntos detectados en el fotograma intermedio
        fotograma_intermedio = cv2.drawKeypoints(
            fotograma,
            puntos_fotograma,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        )

        coincidencias_buenas = []

        if descriptores_fotograma is not None:

            # Comparar descriptores
            pares = comparador.knnMatch(
                descriptores_plantilla,
                descriptores_fotograma,
                k=2
            )

            # Filtrar coincidencias
            for par in pares:

                if len(par) == 2:

                    mejor, segundo = par

                    if mejor.distance < umbral_coincidencia * segundo.distance:
                        coincidencias_buenas.append(mejor)

        if len(coincidencias_buenas) >= minimo_coincidencias:

            # Obtener coordenadas de puntos coincidentes
            puntos_origen = np.float32([
                puntos_plantilla[m.queryIdx].pt
                for m in coincidencias_buenas
            ]).reshape(-1, 1, 2)

            puntos_destino = np.float32([
                puntos_fotograma[m.trainIdx].pt
                for m in coincidencias_buenas
            ]).reshape(-1, 1, 2)

            # Calcular homografia
            homografia, mascara = cv2.findHomography(
                puntos_origen,
                puntos_destino,
                cv2.RANSAC,
                5.0
            )

            if homografia is not None and mascara is not None:

                inliers = int(mascara.sum())

                if inliers >= minimo_inliers:

                    # Transformar esquinas de la plantilla
                    esquinas_transformadas = cv2.perspectiveTransform(
                        esquinas,
                        homografia
                    )

                    # Dibujar contorno del objeto
                    cv2.polylines(
                        fotograma_intermedio,
                        [np.int32(esquinas_transformadas)],
                        True,
                        (255, 255, 0),
                        3
                    )

                    cv2.putText(
                        fotograma_intermedio,
                        f"{objeto.upper()}",
                        (int(esquinas_transformadas[0][0][0]), int(esquinas_transformadas[0][0][1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        20
                    )

                    cv2.putText(
                        fotograma_intermedio,
                        f"Coincidencias: {len(coincidencias_buenas)}",
                        (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                else:
                    cv2.putText(
                        fotograma_intermedio,
                        "BUSCANDO OBJETO...",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255,255,0),
                        2
                    )

            else:
                cv2.putText(
                    fotograma_intermedio,
                    "BUSCANDO OBJETO...",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

        else:
            cv2.putText(
                fotograma_intermedio,
                "BUSCANDO OBJETO...",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )
            

        # Mostrar las tres ventanas
        cv2.imshow("1. Plantilla - Puntos ORB", plantilla_puntos)
        cv2.imshow("2. Fotograma intermedio - Puntos ORB", fotograma_intermedio)

        # Salir al presionar ESC
        if cv2.waitKey(1) == 27:
            break

finally:

    camara.release()
    cv2.destroyAllWindows()
