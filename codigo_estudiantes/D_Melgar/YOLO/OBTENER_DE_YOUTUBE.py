import threading
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURACIÓN GENERAL (WEBCAM)
# ==========================================
# 0 = cámara integrada principal | 1 = cámara web USB externa (si aplica)
CAMERA_INDEX = 1

# Resolución solicitada a la webcam y búfer mínimo para baja latencia
CAM_WIDTH = 1280
CAM_HEIGHT = 720

# Puedes usar "yolo11n.pt", "yolo11s.pt", "yolov8n.pt",
# o un modelo exportado como "yolo11n.onnx" / "yolo11n.engine"
MODEL_PATH = "yolo11n.pt"

# Clases COCO de interés vehicular:
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = [2, 3, 5, 7]
CLASS_NAMES_ES = {
    2: "Auto",
    3: "Moto",
    5: "Bus",
    7: "Camion",
}

# Parámetros de rendimiento y anti-lag
CONF_THRESHOLD = 0.35
IMG_SIZE = 640              # Baja a 480 si vas por CPU y necesitas más FPS
INFER_EVERY_N_FRAMES = 2    # Ejecuta YOLO cada N frames leídos para evitar saturación
LINE_Y_RATIO = 0.65         # Posición vertical de la línea de conteo (65% de la altura)


# ==========================================
# 2. LECTOR MULTIHILO ANTI-BUFFERING (WEBCAM)
# ==========================================
class WebcamReaderThread:
    """
    Lee fotogramas de la cámara web en un hilo separado y conserva SOLO el último frame.
    Evita que el búfer se sature mientras YOLO procesa el fotograma actual.
    """

    def __init__(self, camera_index: int, width: int = 1280, height: int = 720):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.frame_seq = 0
        self.running = False
        self.thread = None

    def _abrir_camara(self) -> bool:
        if self.cap is not None:
            self.cap.release()

        # CAP_DSHOW en Windows suele abrir cámaras web más rápido y con menos retraso
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            return False

        # Reduce el búfer interno para descartar frames viejos y fijar resolución
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.cap = cap
        return True

    def start(self):
        if not self._abrir_camara():
            raise RuntimeError(
                f"No se pudo abrir la cámara web con índice {self.camera_index}. "
                "Verifica que esté conectada y no esté en uso por otra aplicación."
            )
        self.running = True
        self.thread = threading.Thread(target=self._loop_lectura, daemon=True)
        self.thread.start()

    def _loop_lectura(self):
        fallos_consecutivos = 0
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.5)
                self._abrir_camara()
                continue

            ok, frame = self.cap.read()
            if not ok or frame is None:
                fallos_consecutivos += 1
                if fallos_consecutivos >= 15:
                    print("[WARN] Fallo temporal leyendo la webcam; reintentando...")
                    self._abrir_camara()
                    fallos_consecutivos = 0
                time.sleep(0.01)
                continue

            fallos_consecutivos = 0
            # Sobrescribe siempre el último frame disponible para evitar acumulación de lag
            with self.lock:
                self.latest_frame = frame
                self.frame_seq += 1

    def get_latest_frame(self):
        """Devuelve (frame_seq, frame) o (None, None)."""
        with self.lock:
            if self.latest_frame is None:
                return None, None
            return self.frame_seq, self.latest_frame

    def stop(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()


# ==========================================
# 3. CONFIGURACIÓN DE HARDWARE
# ==========================================
def preparar_modelo(model_path: str):
    """Configura dispositivo (CUDA/CPU) y precisión FP16 si hay GPU disponible."""
    has_cuda = torch.cuda.is_available()
    device = 0 if has_cuda else "cpu"
    use_half = has_cuda

    print(f"[INFO] Cargando modelo '{model_path}' en dispositivo: {device} (FP16={use_half})")
    model = YOLO(model_path)
    return model, device, use_half


# ==========================================
# 4. BUCLE PRINCIPAL DE DETECCIÓN Y CONTEO
# ==========================================
def main():
    reader = WebcamReaderThread(
        camera_index=CAMERA_INDEX,
        width=CAM_WIDTH,
        height=CAM_HEIGHT,
    )
    reader.start()

    model, device, use_half = preparar_modelo(MODEL_PATH)

    counted_ids = set()
    counts_by_class = {cls_id: 0 for cls_id in VEHICLE_CLASSES}
    prev_centers_y = {}  # track_id -> última coordenada Y conocida

    last_processed_seq = -1
    loop_counter = 0
    last_boxes_info = []
    fps_display = 0.0
    t_prev = time.perf_counter()

    print("[INFO] Prueba con webcam iniciada. Presiona 'q' para salir.")

    try:
        while True:
            seq, frame = reader.get_latest_frame()
            if frame is None or seq == last_processed_seq:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                time.sleep(0.005)
                continue

            last_processed_seq = seq
            loop_counter += 1

            h, w = frame.shape[:2]
            line_y = int(h * LINE_Y_RATIO)

            # Frame skipping: ejecuta tracking cada N frames para mantener fluidez
            run_inference = (loop_counter % INFER_EVERY_N_FRAMES == 0)

            if run_inference:
                results = model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=VEHICLE_CLASSES,
                    conf=CONF_THRESHOLD,
                    imgsz=IMG_SIZE,
                    device=device,
                    half=use_half,
                    verbose=False,
                )

                current_boxes = []
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    xyxy = boxes.xyxy.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy().astype(int)
                    confs = boxes.conf.cpu().numpy()
                    track_ids = (
                        boxes.id.cpu().numpy().astype(int)
                        if boxes.id is not None
                        else np.full(len(cls_ids), -1, dtype=int)
                    )

                    for box, cls_id, conf, tid in zip(xyxy, cls_ids, confs, track_ids):
                        x1, y1, x2, y2 = map(int, box)
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        # Conteo al cruzar la línea horizontal una sola vez por ID
                        if tid != -1:
                            prev_y = prev_centers_y.get(tid)
                            if prev_y is not None and tid not in counted_ids:
                                crossed_down = prev_y < line_y <= cy
                                crossed_up = prev_y > line_y >= cy
                                if crossed_down or crossed_up:
                                    counted_ids.add(tid)
                                    counts_by_class[cls_id] = counts_by_class.get(cls_id, 0) + 1
                            prev_centers_y[tid] = cy

                        current_boxes.append((x1, y1, x2, y2, cx, cy, cls_id, conf, tid))

                last_boxes_info = current_boxes

            # Dibujar línea de conteo y detecciones
            vis_frame = frame.copy()
            cv2.line(vis_frame, (0, line_y), (w, line_y), (0, 255, 255), 2)

            for x1, y1, x2, y2, cx, cy, cls_id, conf, tid in last_boxes_info:
                label_name = CLASS_NAMES_ES.get(cls_id, str(cls_id))
                id_txt = f"ID:{tid} " if tid != -1 else ""
                text = f"{id_txt}{label_name} {conf:.2f}"

                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.circle(vis_frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(
                    vis_frame,
                    text,
                    (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 200, 0),
                    2,
                )

            # Cálculo de FPS efectivos
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps_display = 0.9 * fps_display + 0.1 * (1.0 / dt) if fps_display > 0 else (1.0 / dt)

            # Panel HUD de conteo y rendimiento
            total_vehiculos = sum(counts_by_class.values())
            cv2.rectangle(vis_frame, (10, 10), (295, 175), (20, 20, 20), -1)
            cv2.putText(
                vis_frame,
                f"FPS: {fps_display:.1f} | Total: {total_vehiculos}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            y_offset = 62
            for cls_id in VEHICLE_CLASSES:
                nombre = CLASS_NAMES_ES[cls_id]
                cant = counts_by_class[cls_id]
                cv2.putText(
                    vis_frame,
                    f"- {nombre}s: {cant}",
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (200, 255, 200),
                    1,
                )
                y_offset += 25

            cv2.imshow("Deteccion y Conteo Vehicular - Webcam YOLO", vis_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        reader.stop()
        cv2.destroyAllWindows()
        print("[INFO] Recursos liberados correctamente.")


if __name__ == "__main__":
    main()