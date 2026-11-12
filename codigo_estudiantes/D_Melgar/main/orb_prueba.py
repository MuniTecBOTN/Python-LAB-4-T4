
import cv2
import numpy as np
import time

# ==========================================
# 1. CONFIGURACION
# ==========================================

PLANTILLAS = [
    # OBJETO 1
    {
        "ruta": "main/plantillas/object1.jpg",
        "objeto": "OBJETO 1",
        "distancia": "LEJOS"
    },
    {
        "ruta": "main/plantillas/objeto_1_cerca.jpg",
        "objeto": "OBJETO 1",
        "distancia": "CERCA"
    },

    # OBJETO 2
    {
        "ruta": "main/plantillas/object2.jpg",
        "objeto": "OBJETO 2",
        "distancia": "LEJOS"
    },
    {
        "ruta": "main/plantillas/objeto_2_cerca.jpg",
        "objeto": "OBJETO 2",
        "distancia": "CERCA"
    },

    # OBJETO 3
    {
        "ruta": "main/plantillas/object3.jpg",
        "objeto": "OBJETO 3",
        "distancia": "LEJOS"
    },
    {
        "ruta": "main/plantillas/objeto_3_cerca.jpg",
        "objeto": "OBJETO 3",
        "distancia": "CERCA"
    },

    # OBJETO 4
    {
        "ruta": "main/plantillas/object4.jpg",
        "objeto": "OBJETO 4",
        "distancia": "LEJOS"
    },
    {
        "ruta": "main/plantillas/objeto_4_cerca.jpg",
        "objeto": "OBJETO 4",
        "distancia": "CERCA"
    }
]

ANCHO = 640
ALTO = 360

MIN_MATCHES = 10
MIN_INLIERS = 9
MIN_INLIER_RATIO = 0.35

SMOOTHING = 0.35
MAX_AREA_RATIO = 0.90


# ==========================================
# 2. PREPROCESAMIENTO
# ==========================================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

orb = cv2.ORB_create(
    nfeatures=2000,
    scaleFactor=1.2,
    nlevels=10
)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


# ==========================================
# 3. CARGAR PLANTILLAS
# ==========================================

templates = []

for plantilla in PLANTILLAS:

    ruta = plantilla["ruta"]
    nombre = plantilla["objeto"]
    distancia = plantilla["distancia"]

    img = cv2.imread(
        ruta,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        print("Error al cargar:", ruta)
        raise SystemExit

    h, w = img.shape

    # Reducir plantillas excesivamente grandes
    if w > 800:

        scale = 800 / w

        img = cv2.resize(
            img,
            None,
            fx=scale,
            fy=scale
        )

    img = clahe.apply(img)

    kp, des = orb.detectAndCompute(
        img,
        None
    )

    if des is None or len(kp) < MIN_MATCHES:

        print("Plantilla con pocos puntos:", nombre, distancia)
        raise SystemExit

    templates.append({
        "name": nombre,
        "distance": distancia,
        "img": img,
        "kp": kp,
        "des": des,
        "shape": img.shape
    })

    print(
        nombre,
        "- Distancia:",
        distancia,
        "- Puntos:",
        len(kp)
    )


# ==========================================
# 4. VALIDAR GEOMETRIA
# ==========================================

def validar_geometria(projected, frame_shape):

    fh, fw = frame_shape[:2]

    if projected is None:
        return False

    if not np.isfinite(projected).all():
        return False

    polygon = projected.reshape(4, 2).astype(np.float32)

    if not cv2.isContourConvex(
        polygon.reshape(-1, 1, 2)
    ):
        return False

    area = abs(cv2.contourArea(polygon))

    if area < 150:
        return False

    if area > fw * fh * MAX_AREA_RATIO:
        return False

    # Comprobar que el poligono no este fuera de pantalla
    if (
        np.max(polygon[:, 0]) < 0 or
        np.max(polygon[:, 1]) < 0 or
        np.min(polygon[:, 0]) > fw or
        np.min(polygon[:, 1]) > fh
    ):
        return False

    return True


# ==========================================
# 5. CAMARA
# ==========================================

cap = cv2.VideoCapture(1)

if not cap.isOpened():

    print("No se pudo abrir la camara")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

smooth_corners = None
previous_name = None

previous_time = time.perf_counter()


# ==========================================
# 6. DETECCION EN TIEMPO REAL
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Error al obtener fotograma")
        break

    original_h, original_w = frame.shape[:2]

    # Reducir imagen para acelerar procesamiento
    small = cv2.resize(
        frame,
        (ANCHO, ALTO)
    )

    gray = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY
    )

    gray = clahe.apply(gray)

    # Evaluar nitidez
    laplacian_var = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    is_evaluable = laplacian_var > 5.0

    # Extraer caracteristicas del fotograma
    kp_frame, des_frame = orb.detectAndCompute(
        gray,
        None
    )

    # Variables de reconocimiento
    mejor_nombre = ""
    mejor_confianza = 0
    mejor_inliers = 0
    mejor_polygon = None
    mejor_distancia = ""

    # ======================================
    # 7. COMPARAR Y AGRUPAR PLANTILLAS
    # ======================================

    # Aqui se guardara la mejor coincidencia
    # de cada objeto, sin importar la distancia.

    mejores_objetos = {}

    if (
        is_evaluable and
        des_frame is not None and
        len(des_frame) >= MIN_MATCHES
    ):

        for t in templates:

            # Comparar descriptores ORB
            matches = matcher.knnMatch(
                t["des"],
                des_frame,
                k=2
            )

            # Filtro de razon de Lowe
            good_matches = []

            for pair in matches:

                if len(pair) == 2:

                    m, n = pair

                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) < MIN_MATCHES:
                continue

            # Obtener puntos correspondientes
            src_pts = np.float32([
                t["kp"][m.queryIdx].pt
                for m in good_matches
            ]).reshape(-1, 1, 2)

            dst_pts = np.float32([
                kp_frame[m.trainIdx].pt
                for m in good_matches
            ]).reshape(-1, 1, 2)

            # Calcular homografia
            matrix, mask = cv2.findHomography(
                src_pts,
                dst_pts,
                cv2.RANSAC,
                4.0
            )

            if matrix is None or mask is None:
                continue

            inliers = mask.ravel().astype(bool)

            cantidad_inliers = np.sum(inliers)

            confianza_actual = (
                cantidad_inliers / len(good_matches)
            ) * 100

            # Validar coincidencias
            if cantidad_inliers < MIN_INLIERS:
                continue

            if confianza_actual < MIN_INLIER_RATIO * 100:
                continue

            # Proyectar esquinas de la plantilla
            h, w = t["shape"]

            corners = np.float32([
                [0, 0],
                [0, h - 1],
                [w - 1, h - 1],
                [w - 1, 0]
            ]).reshape(-1, 1, 2)

            projected = cv2.perspectiveTransform(
                corners,
                matrix
            )

            if not validar_geometria(
                projected,
                gray.shape
            ):
                continue

            # Calcular puntuacion
            score = (
                cantidad_inliers *
                (confianza_actual / 100)
            )

            objeto = t["name"]

            # Agrupar por identidad del objeto
            # y guardar su mejor plantilla.

            if (
                objeto not in mejores_objetos or
                score > mejores_objetos[objeto]["score"]
            ):

                mejores_objetos[objeto] = {

                    "score": score,

                    "inliers": cantidad_inliers,

                    "confianza": confianza_actual,

                    "polygon": projected.reshape(4, 2),

                    "distancia": t["distance"]
                }

        # ==================================
        # 7.1 SELECCIONAR OBJETO PRINCIPAL
        # ==================================

        if mejores_objetos:

            mejor_nombre, mejor = max(
                mejores_objetos.items(),
                key=lambda item: item[1]["score"]
            )

            mejor_inliers = mejor["inliers"]

            mejor_confianza = mejor["confianza"]

            mejor_polygon = mejor["polygon"]

            mejor_distancia = mejor["distancia"]


    # ======================================
    # 8. DIBUJAR RESULTADOS
    # ======================================

    output = frame.copy()

    if not is_evaluable:

        smooth_corners = None
        previous_name = None

        cv2.rectangle(
            output,
            (10, 10),
            (480, 65),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            output,
            "IMAGEN NO EVALUABLE - MEJORAR NITIDEZ",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 165, 255),
            2
        )

    elif mejor_polygon is not None:

        # Reiniciar suavizado si cambia el objeto
        if previous_name != mejor_nombre:

            smooth_corners = mejor_polygon.copy()

        else:

            smooth_corners = (
                SMOOTHING * mejor_polygon +
                (1 - SMOOTHING) * smooth_corners
            )

        previous_name = mejor_nombre

        # Convertir coordenadas al tamaño original
        scale_x = original_w / ANCHO
        scale_y = original_h / ALTO

        polygon = smooth_corners.copy()

        polygon[:, 0] *= scale_x
        polygon[:, 1] *= scale_y

        polygon = np.int32(polygon)

        # Dibujar contorno verde
        cv2.polylines(
            output,
            [polygon],
            True,
            (0, 255, 0),
            3,
            cv2.LINE_AA
        )

        # Centro del objeto
        x, y, bw, bh = cv2.boundingRect(polygon)

        cx = x + bw // 2
        cy = y + bh // 2

        cv2.circle(
            output,
            (cx, cy),
            7,
            (0, 255, 255),
            -1
        )

        # Panel informativo
        cv2.rectangle(
            output,
            (10, 10),
            (490, 180),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            output,
            "OBJETO PRINCIPAL RECONOCIDO",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            "Clasificacion: " + mejor_nombre,
            (25, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            "Distancia: " + mejor_distancia,
            (25, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"Confianza: {mejor_confianza:.1f}%",
            (25, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            f"Inliers: {mejor_inliers}",
            (25, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

    else:

        smooth_corners = None
        previous_name = None

        # ==================================
        # DETECCION DE OBJETOS DESCONOCIDOS
        # ==================================

        blur = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        _, thresh = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for c in contours:

            area = cv2.contourArea(c)

            if area > 1500:

                rx, ry, rw, rh = cv2.boundingRect(c)

                # Convertir coordenadas
                rx = int(rx * original_w / ANCHO)
                ry = int(ry * original_h / ALTO)

                rw = int(rw * original_w / ANCHO)
                rh = int(rh * original_h / ALTO)

                cv2.rectangle(
                    output,
                    (rx, ry),
                    (rx + rw, ry + rh),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    output,
                    "PIEZA NO RECONOCIDA",
                    (rx, max(ry - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2
                )

        cv2.rectangle(
            output,
            (10, 10),
            (460, 85),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            output,
            "OBJETO NO RECONOCIDO",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.putText(
            output,
            "No existe coincidencia suficiente",
            (25, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )


    # ======================================
    # 9. FPS
    # ======================================

    current_time = time.perf_counter()

    elapsed = current_time - previous_time

    fps = 1 / elapsed if elapsed > 0 else 0

    previous_time = current_time

    cv2.putText(
        output,
        f"FPS: {fps:.1f}",
        (original_w - 130, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )


    # ======================================
    # 10. MOSTRAR RESULTADOS
    # ======================================

    cv2.imshow(
        "DETECCION Y EVALUACION ORB",
        output
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ==========================================
# 11. LIBERAR RECURSOS
# ==========================================

cap.release()

cv2.destroyAllWindows()
