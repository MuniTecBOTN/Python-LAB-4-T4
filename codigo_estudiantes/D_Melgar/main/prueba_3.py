import cv2
import numpy as np
import time
from pathlib import Path
from threading import Thread, Lock

# ============================================================
# 1. CONFIGURACION
# ============================================================
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

ANCHO, ALTO = 640, 480
INDICE_CAMARA = 0
CAMERA_FPS_SOLICITADOS = 30
FRECUENCIA_DESCONOCIDOS = 3  # segmentacion menos frecuente; ORB sigue cada frame

# Mantener la calidad inicial de ORB para no reducir deliberadamente la deteccion.
ORB_FEATURES = 1600
MIN_MATCHES = 10
MIN_INLIERS = 7
MIN_INLIER_RATIO = 0.32
RATIO_TEST = 0.75
RANSAC_REPROJ = 4.0
MAX_AREA_RATIO = 0.70

MIN_UNKNOWN_AREA = 3500
MIN_UNKNOWN_WIDTH = 40
MIN_UNKNOWN_HEIGHT = 40
MAX_UNKNOWN_AREA_RATIO = 0.45
MAX_UNKNOWN_ASPECT = 5.0
UNKNOWN_OVERLAP_WITH_KNOWN = 0.18
IOU_SAME_OBJECT = 0.30

# Margen en píxeles para ajustar el rectángulo alrededor del objeto detectado
MARGEN_RECTANGULO = 12

cv2.setUseOptimized(True)
cv2.setNumThreads(0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
orb = cv2.ORB_create(
    nfeatures=ORB_FEATURES,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=19,
    fastThreshold=12,
)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# ============================================================
# 2. CARGAR PLANTILLAS Y UNIFICAR DESCRIPTORES
# ============================================================
templates = []
for item in PLANTILLAS:
    img = cv2.imread(str(item["ruta"]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"ERROR: no se pudo cargar la plantilla: {item['ruta']}")

    h, w = img.shape
    if w > 550:
        scale = 550.0 / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    img = clahe.apply(img)
    kp, des = orb.detectAndCompute(img, None)
    if des is None or len(kp) < MIN_MATCHES:
        print(f"AVISO: plantilla con pocas caracteristicas, se omite: {item['ruta']}")
        continue

    # Calcular el rectángulo que engloba solo los puntos clave (keypoints) de la plantilla,
    # ignorando márgenes vacíos o fondo sobrante alrededor de la caja.
    pts_kp = np.float32([p.pt for p in kp]).reshape(-1, 1, 2)
    tx, ty, tw, th = cv2.boundingRect(pts_kp)
    obj_corners = np.float32([
        [tx, ty],
        [tx, ty + th],
        [tx + tw, ty + th],
        [tx + tw, ty]
    ]).reshape(-1, 1, 2)

    templates.append({
        "name": item["objeto"],
        "distance": item["distancia"],
        "kp": kp,
        "des": des,
        "shape": img.shape,
        "path": item["ruta"],
        "obj_corners": obj_corners,
    })

if not templates:
    raise SystemExit("ERROR: no se pudo cargar ninguna plantilla util.")

start = 0
for template in templates:
    template["match_start"] = start
    start += len(template["des"])
    template["match_end"] = start
all_template_des = np.vstack([t["des"] for t in templates]).astype(np.uint8, copy=False)

print(f"Plantillas utiles cargadas: {len(templates)} de {len(PLANTILLAS)}")
print(f"Descriptores combinados: {len(all_template_des)}")

# ============================================================
# 3. FUNCIONES DE DETECCION
# ============================================================
def calcular_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def validar_geometria(projected, frame_shape):
    fh, fw = frame_shape[:2]
    if projected is None or not np.isfinite(projected).all():
        return False
    corners = projected.reshape(4, 2).astype(np.float32)
    contour = corners.reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        return False
    area = abs(cv2.contourArea(contour))
    if area < 250 or area > fw * fh * MAX_AREA_RATIO:
        return False
    if (corners[:, 0].max() < 0 or corners[:, 1].max() < 0 or
            corners[:, 0].min() >= fw or corners[:, 1].min() >= fh):
        return False
    x, y, w, h = cv2.boundingRect(contour)
    if w < 20 or h < 20:
        return False
    return max(w, h) / max(1.0, min(w, h)) <= 8.0


def suprimir_duplicados_mismo_objeto(detecciones):
    resultado = []
    for det in sorted(detecciones, key=lambda d: d["score"], reverse=True):
        duplicada = any(
            det["name"] == prev["name"] and
            calcular_iou(det["box"], prev["box"]) >= IOU_SAME_OBJECT
            for prev in resultado
        )
        if not duplicada:
            resultado.append(det)
    return resultado


def construir_deteccion(template, matches, frame_kp, frame_shape):
    buenos = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < RATIO_TEST * n.distance:
                buenos.append(m)
    if len(buenos) < MIN_MATCHES:
        return None

    unicos = {}
    for m in buenos:
        if m.trainIdx not in unicos or m.distance < unicos[m.trainIdx].distance:
            unicos[m.trainIdx] = m
    buenos = list(unicos.values())
    if len(buenos) < MIN_MATCHES:
        return None

    src = np.float32([template["kp"][m.queryIdx - template["match_start"]].pt for m in buenos]).reshape(-1, 1, 2)
    dst = np.float32([frame_kp[m.trainIdx].pt for m in buenos]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ)
    if matrix is None or mask is None:
        return None

    inliers = mask.ravel().astype(bool)
    n_inliers = int(inliers.sum())
    confidence = 100.0 * n_inliers / len(buenos)
    if n_inliers < MIN_INLIERS or confidence < MIN_INLIER_RATIO * 100:
        return None

    # Proyectamos las esquinas ajustadas del objeto (sin márgenes vacíos de la plantilla)
    projected = cv2.perspectiveTransform(template["obj_corners"], matrix)
    if not validar_geometria(projected, frame_shape):
        return None

    # Obtener el rectángulo delimitador y aplicarle un pequeño margen estético
    x, y, w, h = cv2.boundingRect(np.int32(projected))
    x = max(0, x - MARGEN_RECTANGULO)
    y = max(0, y - MARGEN_RECTANGULO)
    w += 2 * MARGEN_RECTANGULO
    h += 2 * MARGEN_RECTANGULO

    return {
        "name": template["name"],
        "distance": template["distance"],
        "score": n_inliers * (confidence / 100.0),
        "inliers": n_inliers,
        "confianza": confidence,
        "box": (x, y, w, h),
    }


def detectar_objetos(gray, frame_shape):
    frame_kp, frame_des = orb.detectAndCompute(gray, None)
    if frame_des is None or len(frame_des) < MIN_MATCHES:
        return [], len(frame_kp)

    try:
        all_matches = matcher.knnMatch(all_template_des, frame_des, k=2)
    except cv2.error:
        return [], len(frame_kp)

    detecciones = []
    for template in templates:
        subset = all_matches[template["match_start"]:template["match_end"]]
        det = construir_deteccion(template, subset, frame_kp, frame_shape)
        if det is not None:
            detecciones.append(det)
    return suprimir_duplicados_mismo_objeto(detecciones), len(frame_kp)


def detectar_desconocidos(gray, known_detections):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fh, fw = gray.shape[:2]
    min_area = max(MIN_UNKNOWN_AREA, int(fh * fw * 0.012))
    max_area = int(fh * fw * MAX_UNKNOWN_AREA_RATIO)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < MIN_UNKNOWN_WIDTH or h < MIN_UNKNOWN_HEIGHT:
            continue
        if max(w, h) / max(1.0, min(w, h)) > MAX_UNKNOWN_ASPECT:
            continue
        box = (x, y, w, h)
        if any(calcular_iou(box, d["box"]) >= UNKNOWN_OVERLAP_WITH_KNOWN for d in known_detections):
            continue
        if any(calcular_iou(box, prev) > 0.45 for prev in candidates):
            continue
        candidates.append(box)
    return candidates

# ============================================================
# 4. CAPTURA EN HILO SEPARADO
# ============================================================
class CapturaCamara:
    def __init__(self, indice):
        self.cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la camara con indice {indice}.")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS_SOLICITADOS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = Lock()
        self.frame = None
        self.ok = False
        self.running = True
        self.thread = Thread(target=self._leer, daemon=True)
        self.thread.start()

    def _leer(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
                    self.ok = True
            else:
                time.sleep(0.005)

    def obtener(self):
        with self.lock:
            if not self.ok or self.frame is None:
                return None
            return self.frame.copy()

    def cerrar(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()

# ============================================================
# 5. BUCLE PRINCIPAL
# ============================================================
try:
    camara = CapturaCamara(INDICE_CAMARA)
except RuntimeError as exc:
    raise SystemExit(str(exc))

fps = 0.0
fps_counter = 0
fps_start = time.perf_counter()
unknown_boxes_cache = []
frame_index = 0
last_detections = []
last_keypoints = 0
last_inference_ms = 0.0

try:
    while True:
        frame = camara.obtener()
        if frame is None:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.002)
            continue

        t0 = time.perf_counter()
        original_h, original_w = frame.shape[:2]
        small = cv2.resize(frame, (ANCHO, ALTO), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)

        detecciones, last_keypoints = detectar_objetos(gray, gray.shape)
        frame_index += 1
        if frame_index % FRECUENCIA_DESCONOCIDOS == 0:
            unknown_boxes_cache = detectar_desconocidos(gray, detecciones)
        unknown_boxes = unknown_boxes_cache
        last_detections = detecciones
        last_inference_ms = (time.perf_counter() - t0) * 1000.0

        output = frame.copy()
        sx, sy = original_w / float(ANCHO), original_h / float(ALTO)

        for det in detecciones:
            bx, by, bw, bh = det["box"]
            x, y = int(bx * sx), int(by * sy)
            w, h = int(bw * sx), int(bh * sy)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)
            label = f"{det['name']} {det['distance']} {det['confianza']:.0f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale, thick = 0.48, 1
            (tw, th), base = cv2.getTextSize(label, font, scale, thick)
            ly = max(0, y - th - base - 7)
            cv2.rectangle(output, (x, ly), (x + tw + 8, ly + th + base + 6), (0, 120, 0), -1)
            cv2.putText(output, label, (x + 4, ly + th + 2), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

        for ux, uy, uw, uh in unknown_boxes:
            x, y = int(ux * sx), int(uy * sy)
            w, h = int(uw * sx), int(uh * sy)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 140, 255), 2, cv2.LINE_AA)
            cv2.putText(output, "DESCONOCIDO", (x, max(18, y - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 140, 255), 1, cv2.LINE_AA)

        fps_counter += 1
        now = time.perf_counter()
        elapsed = now - fps_start
        if elapsed >= 1.0:
            measured = fps_counter / elapsed
            fps = measured if fps == 0.0 else 0.70 * fps + 0.30 * measured
            fps_counter = 0
            fps_start = now

        cv2.putText(output, f"FPS PROCESAMIENTO: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(output, f"ORB: {last_keypoints} puntos | {last_inference_ms:.0f} ms",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("ORB - 4 OBJETOS + DESCONOCIDOS (Q para salir)", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    camara.cerrar()
    cv2.destroyAllWindows()