import cv2
import numpy as np
import time
from pathlib import Path

# ============================================================
# 1. CONFIGURACION
# ============================================================

# Las rutas se construyen desde la ubicacion de este .py, no desde
# la carpeta actual de la terminal.
BASE_DIR = Path(__file__).resolve().parent
CARPETA_PLANTILLAS = BASE_DIR / "plantillas"

PLANTILLAS = [
    {"ruta": CARPETA_PLANTILLAS / "cap1_cerca.jpg", "objeto": "OBJETO 1", "distancia": "CERCA"},
    {"ruta": CARPETA_PLANTILLAS / "cap1_lejos.jpg", "objeto": "OBJETO 1", "distancia": "LEJOS"},
    {"ruta": CARPETA_PLANTILLAS / "cap2_cerca.jpg", "objeto": "OBJETO 2", "distancia": "CERCA"},
    {"ruta": CARPETA_PLANTILLAS / "cap2_lejos.jpg", "objeto": "OBJETO 2", "distancia": "LEJOS"},
    {"ruta": CARPETA_PLANTILLAS / "cap3_cerca.jpg", "objeto": "OBJETO 3", "distancia": "CERCA"},
    {"ruta": CARPETA_PLANTILLAS / "cap3_lejos.jpg", "objeto": "OBJETO 3", "distancia": "LEJOS"},
    {"ruta": CARPETA_PLANTILLAS / "cap4_cerca.jpg", "objeto": "OBJETO 4", "distancia": "CERCA"},
    {"ruta": CARPETA_PLANTILLAS / "cap4_lejos.jpg", "objeto": "OBJETO 4", "distancia": "LEJOS"},
]

# Resolucion interna de analisis. La salida conserva el tamano original de la camara.
ANCHO = 640
ALTO = 480
INDICE_CAMARA = 0

# Se mantienen la resolucion y la calidad ORB para no perjudicar la deteccion.
# El analisis de objetos reconocidos se ejecuta en CADA fotograma.
# Solo la busqueda de desconocidos (mas costosa) se actualiza cada 2 fotogramas.
FRECUENCIA_DESCONOCIDOS = 2

# OpenCV usa optimizaciones internas. Dos hilos suelen ser un punto de partida
# razonable; si tu CPU tiene pocos nucleos, prueba tambien con 1.
cv2.setUseOptimized(True)
cv2.setNumThreads(2)

# Umbrales ORB: ajustar si hay falsos positivos o no reconoce una plantilla.
MIN_MATCHES = 10
MIN_INLIERS = 7
MIN_INLIER_RATIO = 0.32
RATIO_TEST = 0.75
RANSAC_REPROJ = 4.0
MAX_AREA_RATIO = 0.70

# Filtro de regiones desconocidas (en pixeles de la imagen 640x480).
MIN_UNKNOWN_AREA = 3500
MIN_UNKNOWN_WIDTH = 40
MIN_UNKNOWN_HEIGHT = 40
MAX_UNKNOWN_AREA_RATIO = 0.45
MAX_UNKNOWN_ASPECT = 5.0
UNKNOWN_OVERLAP_WITH_KNOWN = 0.18

# Dos plantillas CERCA/LEJOS del mismo objeto pueden dar el mismo cuadro.
IOU_SAME_OBJECT = 0.30

# ============================================================
# 2. PREPROCESAMIENTO Y ORB
# ============================================================

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
orb = cv2.ORB_create(
    nfeatures=1600,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=19,
    fastThreshold=12,
)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# ============================================================
# 3. CARGAR PLANTILLAS
# ============================================================

templates = []
for item in PLANTILLAS:
    img = cv2.imread(str(item["ruta"]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERROR: no se pudo cargar la plantilla: {item['ruta']}")
        raise SystemExit(1)

    h, w = img.shape
    if w > 550:
        scale = 550.0 / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    img = clahe.apply(img)
    kp, des = orb.detectAndCompute(img, None)
    if des is None or len(kp) < MIN_MATCHES:
        print(f"AVISO: plantilla con pocas caracteristicas, se omite: {item['ruta']}")
        continue

    templates.append({
        "name": item["objeto"],
        "distance": item["distancia"],
        "kp": kp,
        "des": des,
        "shape": img.shape,
        "path": item["ruta"],
    })

if not templates:
    print("ERROR: no se pudo cargar ninguna plantilla util.")
    raise SystemExit(1)

print(f"Plantillas utiles cargadas: {len(templates)} de {len(PLANTILLAS)}")
print(f"Carpeta de plantillas: {CARPETA_PLANTILLAS}")
print("Deteccion ORB: cada fotograma | Desconocidos: cada", FRECUENCIA_DESCONOCIDOS, "fotogramas")

# ============================================================
# 4. FUNCIONES AUXILIARES
# ============================================================

def calcular_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (w1 * h1) + (w2 * h2) - inter
    return inter / union if union > 0 else 0.0


def validar_geometria(projected, frame_shape):
    fh, fw = frame_shape[:2]
    if projected is None or not np.isfinite(projected).all():
        return False

    poly = projected.reshape(4, 2).astype(np.float32)
    if not cv2.isContourConvex(poly.reshape(-1, 1, 2)):
        return False

    area = abs(cv2.contourArea(poly))
    if area < 250 or area > fw * fh * MAX_AREA_RATIO:
        return False

    # Rechazar proyecciones totalmente fuera del fotograma.
    if (poly[:, 0].max() < 0 or poly[:, 1].max() < 0 or
            poly[:, 0].min() >= fw or poly[:, 1].min() >= fh):
        return False

    x, y, w, h = cv2.boundingRect(np.int32(poly))
    if w < 20 or h < 20:
        return False
    aspect = max(w, h) / max(1.0, min(w, h))
    return aspect <= 8.0


def suprimir_duplicados_mismo_objeto(detecciones, iou_thresh=IOU_SAME_OBJECT):
    """Fusiona resultados CERCA/LEJOS repetidos, pero nunca cruza clases."""
    resultado = []
    for det in sorted(detecciones, key=lambda d: d["score"], reverse=True):
        duplicada = any(
            det["name"] == prev["name"] and
            calcular_iou(det["box"], prev["box"]) >= iou_thresh
            for prev in resultado
        )
        if not duplicada:
            resultado.append(det)
    return resultado


def detectar_plantilla_en_fotograma(template, frame_kp, frame_des, frame_shape):
    """Busca una plantilla en todo el fotograma, sin depender de contornos."""
    if frame_des is None or len(frame_des) < MIN_MATCHES:
        return None

    try:
        pairs = matcher.knnMatch(template["des"], frame_des, k=2)
    except cv2.error:
        return None

    buenos = []
    for pair in pairs:
        if len(pair) == 2:
            m, n = pair
            if m.distance < RATIO_TEST * n.distance:
                buenos.append(m)

    if len(buenos) < MIN_MATCHES:
        return None

    # Evita que varios puntos de plantilla apunten al mismo punto del fotograma.
    unicos = {}
    for m in buenos:
        if m.trainIdx not in unicos or m.distance < unicos[m.trainIdx].distance:
            unicos[m.trainIdx] = m
    buenos = list(unicos.values())
    if len(buenos) < MIN_MATCHES:
        return None

    src = np.float32([template["kp"][m.queryIdx].pt for m in buenos]).reshape(-1, 1, 2)
    dst = np.float32([frame_kp[m.trainIdx].pt for m in buenos]).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ)
    if matrix is None or mask is None:
        return None

    inliers = mask.ravel().astype(bool)
    n_inliers = int(inliers.sum())
    confidence = 100.0 * n_inliers / len(buenos)
    if n_inliers < MIN_INLIERS or confidence < MIN_INLIER_RATIO * 100:
        return None

    ht, wt = template["shape"]
    corners = np.float32([
        [0, 0], [0, ht - 1], [wt - 1, ht - 1], [wt - 1, 0]
    ]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, matrix)
    if not validar_geometria(projected, frame_shape):
        return None

    polygon = projected.reshape(4, 2)
    x, y, w, h = cv2.boundingRect(np.int32(polygon))
    score = n_inliers * (confidence / 100.0)

    return {
        "name": template["name"],
        "distance": template["distance"],
        "score": score,
        "inliers": n_inliers,
        "confianza": confidence,
        "polygon": polygon,
        "box": (x, y, w, h),
    }


def detectar_desconocidos(gray, known_detections):
    """Propone regiones grandes no explicadas por una deteccion conocida.

    Es un filtro visual por contornos, no un clasificador semantico: puede
    marcar una region del fondo si tiene suficiente tamano y contraste.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Kernel pequeno para reducir la union accidental de regiones separadas.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    h_frame, w_frame = gray.shape[:2]
    min_area = max(MIN_UNKNOWN_AREA, int(h_frame * w_frame * 0.012))
    max_area = int(h_frame * w_frame * MAX_UNKNOWN_AREA_RATIO)
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < MIN_UNKNOWN_WIDTH or h < MIN_UNKNOWN_HEIGHT:
            continue
        aspect = max(w, h) / max(1.0, min(w, h))
        if aspect > MAX_UNKNOWN_ASPECT:
            continue

        box = (x, y, w, h)
        # No volver a dibujar como desconocida una region ya reconocida.
        if any(calcular_iou(box, det["box"]) >= UNKNOWN_OVERLAP_WITH_KNOWN
               for det in known_detections):
            continue

        # Elimina cajas desconocidas casi duplicadas.
        if any(calcular_iou(box, prev) > 0.45 for prev in candidates):
            continue
        candidates.append(box)

    return candidates

# ============================================================
# 5. CAMARA
# ============================================================

cap = cv2.VideoCapture(INDICE_CAMARA, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"No se pudo abrir la camara con indice {INDICE_CAMARA}.")
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
cap.set(cv2.CAP_PROP_FPS, 30)  # Solicitud; la camara puede no admitir 30 FPS.
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

fps = 0.0
fps_counter = 0
fps_start = time.perf_counter()
unknown_boxes_cache = []
frame_index = 0

# ============================================================
# 6. BUCLE PRINCIPAL
# ============================================================

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer un fotograma de la camara.")
            break

        original_h, original_w = frame.shape[:2]
        small = cv2.resize(frame, (ANCHO, ALTO), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)

        # Calcular caracteristicas una sola vez por fotograma.
        frame_kp, frame_des = orb.detectAndCompute(gray, None)
        detecciones = []

        if frame_des is not None and len(frame_des) >= MIN_MATCHES:
            # Cada plantilla se compara contra el fotograma completo.
            # Asi, un contorno unido no impide buscar otro objeto separado.
            for template in templates:
                det = detectar_plantilla_en_fotograma(
                    template, frame_kp, frame_des, gray.shape
                )
                if det is not None:
                    detecciones.append(det)

        # Las versiones CERCA/LEJOS de un mismo objeto no deben duplicarse.
        detecciones = suprimir_duplicados_mismo_objeto(detecciones)

        # Los objetos conocidos se analizan en TODOS los fotogramas.
        # La segmentacion de desconocidos se actualiza cada N fotogramas para
        # ahorrar CPU; entre actualizaciones se conserva la ultima estimacion.
        frame_index += 1
        if frame_index % FRECUENCIA_DESCONOCIDOS == 0:
            unknown_boxes_cache = detectar_desconocidos(gray, detecciones)
        unknown_boxes = unknown_boxes_cache

        # ====================================================
        # 7. DIBUJAR RESULTADOS EN LA RESOLUCION ORIGINAL
        # ====================================================
        output = frame.copy()
        sx = original_w / float(ANCHO)
        sy = original_h / float(ALTO)

        for det in detecciones:
            # Se dibuja un rectangulo alineado a los ejes, no un poligono.
            # La homografia se conserva internamente para localizar la plantilla.
            bx, by, bw, bh = det["box"]
            x = int(bx * sx)
            y = int(by * sy)
            w = int(bw * sx)
            h = int(bh * sy)
            cv2.rectangle(output, (x, y), (x + w, y + h),
                          (0, 255, 0), 2, cv2.LINE_AA)
            label = f"{det['name']} {det['distance']} {det['confianza']:.0f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.48
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            label_y = max(0, y - th - baseline - 7)
            cv2.rectangle(output, (x, label_y), (x + tw + 8, label_y + th + baseline + 6),
                          (0, 120, 0), -1)
            cv2.putText(output, label, (x + 4, label_y + th + 2), font,
                        font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        for ux, uy, uw, uh in unknown_boxes:
            x = int(ux * sx)
            y = int(uy * sy)
            w = int(uw * sx)
            h = int(uh * sy)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 140, 255), 2, cv2.LINE_AA)
            cv2.putText(output, "DESCONOCIDO", (x, max(18, y - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA)

        # FPS suavizado para que el indicador no fluctue tanto.
        fps_counter += 1
        now = time.perf_counter()
        elapsed = now - fps_start
        if elapsed >= 1.0:
            measured_fps = fps_counter / elapsed
            fps = measured_fps if fps == 0.0 else (0.75 * fps + 0.25 * measured_fps)
            fps_counter = 0
            fps_start = now

        cv2.putText(output, f"FPS: {fps:.1f}", (max(10, original_w - 150), 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("ORB - CUATRO OBJETOS + DESCONOCIDOS", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
