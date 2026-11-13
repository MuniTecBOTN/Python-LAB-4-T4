import cv2
import numpy as np
import itertools
from pathlib import Path

# ==========================================
# 1. CONFIGURACIÓN Y PARÁMETROS
# ==========================================
MIN_MATCHES       = 12      # Coincidencias mínimas entre piezas
RATIO_TEST        = 0.75    # Filtro de Lowe
MIN_INLIERS       = 8       # Inliers mínimos para validar homografía
MIN_AREA          = 5000    # Área mínima para considerar una pieza
MIN_KP_OBJ        = 10      # Puntos clave mínimos en la pieza para evaluar
MAX_FALLOS_LK     = 8       # Frames máximos que un objeto puede perderse en LK
LINEA_CONTEO_X    = 0.6     # Línea virtual al 60% del ancho de la pantalla

# ==========================================
# 2. CLASES AUXILIARES
# ==========================================
class Objeto:
    _ids = itertools.count(1)
    def __init__(self, pts, bbox):
        self.id = next(Objeto._ids)
        self.pts = pts.reshape(-1, 1, 2).astype(np.float32)
        self.bbox = bbox
        self.fallos = 0
        self.contado = False
        self.categoria = None
        self.estado = "pendiente"
        self.centro_prev = self._centro()

    def _centro(self):
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)

    def actualizar(self, pts, bbox):
        self.pts = pts
        self.bbox = bbox
        self.centro_prev = self._centro()
        self.fallos = 0

    def centro(self):
        return self._centro()

def crear_orb():
    return cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8, edgeThreshold=10, fastThreshold=5)

def validar_homografia(kp_obj, kp_pl, matches, shape_obj):
    if len(matches) < MIN_MATCHES: return False, 0
    src = np.float32([kp_obj[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_pl[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or mask is None: return False, 0
    inliers = int(mask.sum())
    if inliers < MIN_INLIERS: return False, inliers
    return True, inliers

def clasificar_objeto(roi_gris, kp_obj, des_obj, plantillas, bf):
    if des_obj is None or len(kp_obj) < MIN_KP_OBJ: return "no_evaluable", None
    if roi_gris is None or roi_gris.size == 0: return "no_evaluable", None
    if cv2.Laplacian(roi_gris, cv2.CV_64F).var() < 10: return "no_evaluable", None

    mejor_inliers = 0
    mejor_cat = None
    evaluado = False

    for cat, vistas in plantillas.items():
        for vista in vistas:
            if des_obj is None or vista[1] is None: continue
            matches = bf.knnMatch(des_obj, vista[1], k=2)
            buenos = [m for m, n in matches if m.distance < RATIO_TEST * n.distance]
            if len(buenos) >= MIN_MATCHES: evaluado = True
            
            ok, inliers = validar_homografia(kp_obj, vista[0], buenos, roi_gris.shape)
            if ok and inliers > mejor_inliers:
                mejor_inliers = inliers
                mejor_cat = cat

    if mejor_cat is not None: return "conocida", mejor_cat
    if evaluado: return "desconocida", None
    return "no_evaluable", None

# ==========================================
# 3. PROGRAMA PRINCIPAL
# ==========================================
def main():
    # --- Cargar Plantillas ---
    plantillas = {}
    ruta_ref = Path("referencias")
    if not ruta_ref.exists():
        print("ERROR: No se encuentra la carpeta 'referencias'")
        return
        
    orb = crear_orb()
    for carpeta in ruta_ref.iterdir():
        if not carpeta.is_dir(): continue
        cat = carpeta.name.split("_")[-1].upper()
        plantillas[cat] = []
        for img_path in carpeta.glob("*.jpg"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                kp, des = orb.detectAndCompute(img, None)
                plantillas[cat].append((kp, des))
        print(f"[Registro] Pieza {cat}: {len(plantillas[cat])} vistas cargadas")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # --- Cámara ---
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret: return
    alto, ancho = frame.shape[:2]
    linea_x = int(ancho * LINEA_CONTEO_X)
    
    # Contadores
    conteos = {cat: 0 for cat in plantillas}
    conteos["desconocida"] = 0
    conteos["no_evaluable"] = 0

    objetos = []
    gris_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_idx = 0

    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        gris_actual = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. SEGUIMIENTO LUCAS-KANADE
        vivos = []
        for obj in objetos:
            nuevos_pts, st, _ = cv2.calcOpticalFlowPyrLK(gris_prev, gris_actual, obj.pts, None, **lk_params)
            if nuevos_pts is not None and st is not None:
                st = st.flatten() == 1
                buenos = nuevos_pts[st]
                if len(buenos) >= 6:
                    x, y, w, h = cv2.boundingRect(buenos)
                    obj.actualizar(buenos.reshape(-1, 1, 2), (x, y, w, h))
                    vivos.append(obj)
                    continue
            obj.fallos += 1
            if obj.fallos <= MAX_FALLOS_LK:
                vivos.append(obj)
        objetos = vivos

        # 2. DETECCIÓN DE NUEVOS OBJETOS (Cada 5 frames)
        if frame_idx % 5 == 0:
            blur = cv2.GaussianBlur(gris_actual, (5, 5), 0)
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Limpieza morfológica: elimina puntitos y separa objetos que se tocan
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
            
            contornos, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contornos:
                area = cv2.contourArea(c)
                x, y, w, h = cv2.boundingRect(c)
                
                # ==========================================
                # FILTROS DE SEGURIDAD EXTREMOS
                # ==========================================
                # 1. Descartar si es muy pequeño
                if area < MIN_AREA: continue
                
                # 2. Descartar si es muy grande (mesa, mano, fondo)
                if w > ancho * 0.50 or h > alto * 0.50: continue
                
                # 3. Descartar si es una franja (líneas de la banda o bordes)
                aspect_ratio = float(w) / h
                if aspect_ratio > 3.0 or aspect_ratio < 0.33: continue
                
                # 4. Descartar si está tocando el borde de la cámara
                if x <= 5 or y <= 5 or x + w >= ancho - 5 or y + h >= alto - 5: continue
                # ==========================================

                cx, cy = x + w/2, y + h/2
                
                # Evitar duplicados (si el centro está a menos de 60px de otro)
                duplicado = False
                for obj in objetos:
                    ox, oy = obj.centro()
                    if abs(ox - cx) < 60 and abs(oy - cy) < 60:
                        duplicado = True
                        break
                if duplicado: continue

                roi = gris_actual[y:y+h, x:x+w]
                kp, des = orb.detectAndCompute(roi, None)
                if kp is None or len(kp) < 4: continue

                pts = np.float32([[k.pt[0] + x, k.pt[1] + y] for k in kp]).reshape(-1, 1, 2)
                nuevo_obj = Objeto(pts, (x, y, w, h))
                estado, cat = clasificar_objeto(roi, kp, des, plantillas, bf)
                nuevo_obj.estado, nuevo_obj.categoria = estado, cat
                
                # Para depurar: imprimir qué está detectando
                print(f"Nuevo objeto -> Categoría: {cat}, Estado: {estado}, Área: {area}")
                
                objetos.append(nuevo_obj)

        # 3. EVALUAR CRUCES Y DIBUJAR
        for obj in objetos:
            # Conteo
            if not obj.contado and obj.estado in ["conocida", "desconocida"]:
                cx, _ = obj.centro()
                px, _ = obj.centro_prev
                if px < linea_x <= cx: # Cruzó la línea
                    if obj.estado == "conocida":
                        conteos[obj.categoria] += 1
                    else:
                        conteos["desconocida"] += 1
                    obj.contado = True

            # Dibujo
            x, y, w, h = obj.bbox
            if obj.contado:
                color = (150, 150, 150)
                etiqueta = f"{obj.categoria if obj.categoria else 'Desc'} (Contado)"
            elif obj.estado == "conocida":
                color, etiqueta = (0, 255, 0), f"{obj.categoria} #{obj.id}"
            elif obj.estado == "desconocida":
                color, etiqueta = (0, 0, 255), f"DESCONOCIDA #{obj.id}"
            else:
                color, etiqueta = (0, 255, 255), f"Evaluando..."

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, etiqueta, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if obj.estado == "desconocida" and not obj.contado:
                cv2.putText(frame, "!! ALERTA !!", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Interfaz
        cv2.line(frame, (linea_x, 0), (linea_x, alto), (255, 200, 0), 2)
        y_texto = 30
        cv2.putText(frame, "CONTEO", (10, y_texto), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        for k, v in conteos.items():
            y_texto += 25
            cv2.putText(frame, f"{k}: {v}", (10, y_texto), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Inspeccion ORB + LK", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
        gris_prev = gris_actual

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()