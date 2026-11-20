import cv2
import numpy as np
import time

# ============================================================
# 1. CONFIGURACION GENERAL
# ============================================================

PLANTILLAS = [
    {"ruta": "main/plantillas/cap1.jpg", "objeto": "OBJETO 1", "distancia": "LEJOS"},
    {"ruta": "main/plantillas/cap1_cerca.jpg", "objeto": "OBJETO 1", "distancia": "CERCA"},

    {"ruta": "main/plantillas/cap2.jpg", "objeto": "OBJETO 2", "distancia": "LEJOS"},
    {"ruta": "main/plantillas/cap2_cerca.jpg", "objeto": "OBJETO 2", "distancia": "CERCA"},

    {"ruta": "main/plantillas/cap3.jpg", "objeto": "OBJETO 3", "distancia": "LEJOS"},
    {"ruta": "main/plantillas/cap3_cerca.jpg", "objeto": "OBJETO 3", "distancia": "CERCA"},

    {"ruta": "main/plantillas/cap4.jpg", "objeto": "OBJETO 4", "distancia": "LEJOS"},
    {"ruta": "main/plantillas/cap4_cerca.jpg", "objeto": "OBJETO 4", "distancia": "CERCA"},

    {"ruta": "main/plantillas/cap4_mas_lejos.jpg", "objeto": "OBJETO 4", "distancia": "LEJOS"},
    {"ruta": "main/plantillas/cap4_mas_cerca.jpg", "objeto": "OBJETO 4", "distancia": "CERCA"},
]


import cv2
import numpy as np
import time

# ==========================================
# 1. CONFIGURACION
# ==========================================

PLANTILLAS = [
    # OBJETO 1
    {
        "ruta": "main/plantillas/cap1.jpg",
        "objeto": "OBJETO 1",
        "distancia": "CERCA"
    },
    {
        "ruta": "main/plantillas/cap1_mas_cerca.jpg",
        "objeto": "OBJETO 1",
        "distancia": "CERCA"
    },
    
    {
        "ruta": "main/plantillas/cap1_lejos.jpg",
        "objeto": "OBJETO 1",
        "distancia": "LEJOS"
    },
    
    {
        "ruta": "main/plantillas/cap1_mas_lejos.jpg",
        "objeto": "OBJETO 1",   
        "distancia": "LEJOS"
    },

    # OBJETO 2
    {
        "ruta": "main/plantillas/cap2.jpg",
        "objeto": "OBJETO 2",
        "distancia": "CERCA"
    },
    {
        "ruta": "main/plantillas/cap2_mas_cerca.jpg",
        "objeto": "OBJETO 2",
        "distancia": "CERCA"
    },
    
    {
        "ruta": "main/plantillas/cap2_lejos.jpg",
        "objeto": "OBJETO 2",
        "distancia": "LEJOS"
    },
    
    {
        "ruta": "main/plantillas/cap2_mas_lejos.jpg",
        "objeto": "OBJETO 2",
        "distancia": "LEJOS"
    },

    # OBJETO 3
    {
        "ruta": "main/plantillas/cap3.jpg",
        "objeto": "OBJETO 3",
        "distancia": "CERCA"
    },
    {
        "ruta": "main/plantillas/cap3_mas_cerca.jpg",
        "objeto": "OBJETO 3",
        "distancia": "CERCA"
    },
    
    {
        
        "ruta": "main/plantillas/cap3_lejos.jpg",
        "objeto": "OBJETO 3",
        "distancia": "LEJOS"
    },
    
    {
        "ruta": "main/plantillas/cap3_mas_lejos.jpg",
        "objeto": "OBJETO 3",
        "distancia": "LEJOS"
    },

    # OBJETO 4
    {
        "ruta": "main/plantillas/cap4.jpg",
        "objeto": "OBJETO 4",
        "distancia": "LEJOS",
    },
    {
        "ruta": "main/plantillas/cap4_cerca.jpg",
        "objeto": "OBJETO 4",
        "distancia": "CERCA",
    },
    
    {
        "ruta" : "main/plantillas/cap4_mas_lejos.jpg",
        "objeto" : "OBJETO 4",
        "distancia" : "LEJOS",
    },
    
    {
        "ruta" : "main/plantillas/cap4_mas_cerca.jpg",
        "objeto" : "OBJETO 4",
        "distancia" : "CERCA",
    },
]
CAMERA_INDEX = 1

# La captura es 4:3 y el procesamiento tambien, para no deformar
# la imagen. Si la camara no admite 640x480, se adapta igualmente.
CAM_WIDTH = 640
CAM_HEIGHT = 480
PROCESS_WIDTH = 480
PROCESS_HEIGHT = 360

# ORB: punto de equilibrio entre velocidad y cantidad de detalles
ORB_FEATURES = 1400
ORB_LEVELS = 8
ORB_FAST_THRESHOLD = 8

# Criterios de reconocimiento: se conservan umbrales conservadores
MIN_MATCHES = 10
MIN_INLIERS = 9
MIN_INLIER_RATIO = 0.35
RATIO_TEST = 0.75
RANSAC_THRESHOLD = 4.0
MAX_AREA_RATIO = 0.90

# Se procesa reconocimiento cada N fotogramas.
# Como las piezas permanecen quietas durante el reconocimiento,
# se puede reutilizar el ultimo resultado entre detecciones.
DETECTION_EVERY = 2

# Cada 10 ciclos de deteccion se revisan todas las plantillas.
# Entre esos ciclos se prueban solo las plantillas del objeto actual.
FULL_SCAN_EVERY = 10

SMOOTHING = 0.35
BLUR_THRESHOLD = 5.0

# ============================================================
# 2. OPTIMIZACIONES DE OPENCV
# ============================================================

cv2.setUseOptimized(True)

# CLAHE se aplica a las plantillas al inicio y a cada imagen analizada.
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

orb = cv2.ORB_create(
    nfeatures=ORB_FEATURES,
    scaleFactor=1.2,
    nlevels=ORB_LEVELS,
    edgeThreshold=15,
    patchSize=31,
    fastThreshold=ORB_FAST_THRESHOLD
)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# ============================================================
# 3. CARGAR Y PRECALCULAR LAS PLANTILLAS
# ============================================================

templates = []
templates_by_object = {}

for p in PLANTILLAS:
    img = cv2.imread(p["ruta"], cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise SystemExit(f'No se pudo cargar la plantilla: {p["ruta"]}')

    h, w = img.shape

    # Limitar el tamaño de referencia sin cambiar su proporcion
    if max(h, w) > 800:
        scale = 800.0 / max(h, w)
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    img = clahe.apply(img)
    kp, des = orb.detectAndCompute(img, None)

    if des is None or len(kp) < MIN_MATCHES:
        raise SystemExit(
            f'La plantilla {p["ruta"]} tiene muy pocos puntos ORB.'
        )

    t = {
        "name": p["objeto"],
        "distance": p["distancia"],
        "img": img,
        "kp": kp,
        "des": des,
        "shape": img.shape
    }

    templates.append(t)
    templates_by_object.setdefault(p["objeto"], []).append(t)

    print(
        f'{p["objeto"]} | {p["distancia"]} | '
        f'puntos ORB: {len(kp)} | {p["ruta"]}'
    )

# ============================================================
# 4. VALIDACION GEOMETRICA
# ============================================================

def validar_geometria(projected, frame_shape):
    fh, fw = frame_shape[:2]

    if projected is None or not np.isfinite(projected).all():
        return False

    polygon = projected.reshape(4, 2).astype(np.float32)

    if not cv2.isContourConvex(polygon.reshape(-1, 1, 2)):
        return False

    area = abs(cv2.contourArea(polygon))

    if area < 150 or area > fw * fh * MAX_AREA_RATIO:
        return False

    # Rechazar el poligono solo si queda completamente fuera del cuadro
    if (
        np.max(polygon[:, 0]) < 0 or
        np.max(polygon[:, 1]) < 0 or
        np.min(polygon[:, 0]) >= fw or
        np.min(polygon[:, 1]) >= fh
    ):
        return False

    return True

# ============================================================
# 5. RECONOCER UN GRUPO DE PLANTILLAS
# ============================================================

def buscar_mejor_coincidencia(gray, kp_frame, des_frame, lista_plantillas):
    mejor = None

    for t in lista_plantillas:
        # KNN con descriptores binarios ORB
        try:
            matches = matcher.knnMatch(t["des"], des_frame, k=2)
        except cv2.error:
            continue

        buenos = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < RATIO_TEST * n.distance:
                    buenos.append(m)

        if len(buenos) < MIN_MATCHES:
            continue

        src_pts = np.float32([
            t["kp"][m.queryIdx].pt for m in buenos
        ]).reshape(-1, 1, 2)

        dst_pts = np.float32([
            kp_frame[m.trainIdx].pt for m in buenos
        ]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            RANSAC_THRESHOLD
        )

        if matrix is None or mask is None:
            continue

        inliers = mask.ravel().astype(bool)
        inlier_count = int(np.count_nonzero(inliers))
        inlier_ratio = inlier_count / len(buenos)

        if inlier_count < MIN_INLIERS:
            continue

        if inlier_ratio < MIN_INLIER_RATIO:
            continue

        h, w = t["shape"]
        corners = np.float32([
            [0, 0],
            [0, h - 1],
            [w - 1, h - 1],
            [w - 1, 0]
        ]).reshape(-1, 1, 2)

        projected = cv2.perspectiveTransform(corners, matrix)

        if not validar_geometria(projected, gray.shape):
            continue

        # Puntuar por cantidad de inliers y proporcion de inliers
        score = inlier_count * inlier_ratio

        if mejor is None or score > mejor["score"]:
            mejor = {
                "name": t["name"],
                "distance": t["distance"],
                "confidence": inlier_ratio * 100.0,
                "inliers": inlier_count,
                "score": score,
                "polygon": projected.reshape(4, 2)
            }

    return mejor

# ============================================================
# 6. DETECCION COMPLETA O BUSQUEDA RAPIDA
# ============================================================

def detectar(gray, active_object, detection_number):
    # Calcular puntos ORB una sola vez por imagen analizada
    kp_frame, des_frame = orb.detectAndCompute(gray, None)

    if des_frame is None or len(des_frame) < MIN_MATCHES:
        return None, kp_frame

    # Al inicio y periodicamente se revisan todas las plantillas.
    full_scan = (
        active_object is None or
        detection_number % FULL_SCAN_EVERY == 0
    )

    if full_scan:
        candidate_templates = templates
    else:
        # Si el objeto sigue en escena, comparar sus variantes de distancia
        candidate_templates = templates_by_object.get(active_object, templates)

    best = buscar_mejor_coincidencia(
        gray,
        kp_frame,
        des_frame,
        candidate_templates
    )

    # Si la busqueda reducida falla, hacer una busqueda completa
    # inmediatamente para no perder un objeto que haya cambiado.
    if best is None and not full_scan:
        best = buscar_mejor_coincidencia(
            gray,
            kp_frame,
            des_frame,
            templates
        )

    return best, kp_frame

# ============================================================
# 7. DETECCION DE PIEZAS NO RECONOCIDAS
# ============================================================

def dibujar_objetos_desconocidos(output, gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

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
        if cv2.contourArea(c) > 1500:
            x, y, w, h = cv2.boundingRect(c)

            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

            cv2.putText(
                output,
                "PIEZA NO RECONOCIDA",
                (x, max(y - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                2
            )

# ============================================================
# 8. ABRIR CAMARA
# ============================================================

# CAP_DSHOW suele funcionar bien en Windows.
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

# Alternativa por si DirectShow no abre la camara
if not cap.isOpened():
    cap.release()
    cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise SystemExit("No se pudo abrir la camara.")

# Solicitar MJPG: algunas webcams lo admiten y reducen carga USB.
cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print(
    "Camara:",
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    "x",
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    "| FPS reportados:",
    cap.get(cv2.CAP_PROP_FPS)
)

# ============================================================
# 9. VARIABLES DE SEGUIMIENTO Y FPS
# ============================================================

active_object = None
cached_result = None
smooth_corners = None

frame_number = 0
detection_number = 0

fps_frames = 0
fps_start = time.perf_counter()
fps = 0.0

# ============================================================
# 10. BUCLE PRINCIPAL
# ============================================================

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error al obtener fotograma de la camara.")
        break

    frame_number += 1

    # Reducir a 4:3 para mantener la proporcion de la captura 640x480
    small = cv2.resize(
        frame,
        (PROCESS_WIDTH, PROCESS_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    # Reconocer cada dos fotogramas; en los intermedios se reutiliza
    # el resultado porque las piezas permanecen quietas.
    run_detection = (frame_number % DETECTION_EVERY == 0)

    gray = None

    if run_detection:
        detection_number += 1

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)

        # Calculo de nitidez usando float32, algo mas ligero que float64
        sharpness = cv2.Laplacian(gray, cv2.CV_32F).var()
        is_evaluable = sharpness > BLUR_THRESHOLD

        if is_evaluable:
            result, _ = detectar(
                gray,
                active_object,
                detection_number
            )

            cached_result = result

            if result is not None:
                active_object = result["name"]

                if (
                    smooth_corners is None or
                    active_object != getattr(
                        detectar,
                        "_previous_object",
                        None
                    )
                ):
                    smooth_corners = result["polygon"].copy()
                else:
                    smooth_corners = (
                        SMOOTHING * result["polygon"] +
                        (1.0 - SMOOTHING) * smooth_corners
                    )

                detectar._previous_object = active_object

            else:
                active_object = None
                smooth_corners = None
                detectar._previous_object = None

        else:
            is_evaluable = False
            cached_result = None
            active_object = None
            smooth_corners = None
            detectar._previous_object = None

    else:
        # No volver a calcular ORB en este fotograma
        is_evaluable = True

    output = small.copy()

    # ========================================================
    # 11. DIBUJAR EL RESULTADO GUARDADO
    # ========================================================

    if not is_evaluable:
        cv2.rectangle(output, (8, 8), (430, 52), (25, 25, 25), -1)
        cv2.putText(
            output,
            "IMAGEN BORROSA / NO EVALUABLE",
            (18, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2
        )

    elif cached_result is not None and smooth_corners is not None:
        polygon = np.int32(smooth_corners)

        cv2.polylines(
            output,
            [polygon],
            True,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        x, y, w, h = cv2.boundingRect(polygon)
        cx, cy = x + w // 2, y + h // 2

        cv2.circle(output, (cx, cy), 5, (0, 255, 255), -1)

        cv2.rectangle(output, (8, 8), (440, 130), (25, 25, 25), -1)

        cv2.putText(
            output,
            "OBJETO RECONOCIDO",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            f'Clasificacion: {cached_result["name"]}',
            (18, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

        cv2.putText(
            output,
            f'Distancia: {cached_result["distance"]}',
            (18, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1
        )

        cv2.putText(
            output,
            f'Confianza: {cached_result["confidence"]:.1f}%  '
            f'Inliers: {cached_result["inliers"]}',
            (18, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 255, 0),
            1
        )

    else:
        # Solo calcular contornos en los fotogramas de deteccion
        # para no gastar CPU en cada iteracion.
        if run_detection and gray is not None:
            dibujar_objetos_desconocidos(output, gray)

        cv2.rectangle(output, (8, 8), (430, 55), (25, 25, 25), -1)
        cv2.putText(
            output,
            "OBJETO NO RECONOCIDO",
            (18, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 255),
            2
        )

    # ========================================================
    # 12. FPS PROMEDIO DE VISUALIZACION
    # ========================================================

    fps_frames += 1
    now = time.perf_counter()
    elapsed = now - fps_start

    if elapsed >= 1.0:
        fps = fps_frames / elapsed
        fps_frames = 0
        fps_start = now

    cv2.putText(
        output,
        f"FPS: {fps:.1f}",
        (PROCESS_WIDTH - 105, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    cv2.imshow("RECONOCIMIENTO ORB OPTIMIZADO", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ============================================================
# 13. LIBERAR RECURSOS
# ============================================================

cap.release()
cv2.destroyAllWindows()
