import os
import sys
import csv
import tempfile
import threading
import sqlite3
import uuid
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QImage, QPainter, QPen, QColor, QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QSpinBox, QListWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QMessageBox, QGroupBox, QSplitter, QLineEdit, QFileDialog, QCheckBox,
    QListWidgetItem, QScrollArea, QFrame
)





BASE_DIR = Path(__file__).resolve().parent

SOURCE_MODE = "video"
VIDEO_PATH = str(BASE_DIR / "trafico_prueba.mp4")
LOOP_VIDEO = True

CAMERA_INDEX = 1
CAM_WIDTH = 1280
CAM_HEIGHT = 720

MODEL_PATH = "yolo11n.pt"
IMG_SIZE = 768
INFER_EVERY_N_FRAMES = 1

INCLUDE_BICYCLE_AS_MOTO = True

VEHICLE_CLASSES = [1, 2, 3, 5, 7] if INCLUDE_BICYCLE_AS_MOTO else [2, 3, 5, 7]

CLASS_CONF_THRESHOLDS = {
    1: 0.20,
    2: 0.35,
    3: 0.18,
    5: 0.35,
    7: 0.35,
}
MIN_BASE_CONF = min(CLASS_CONF_THRESHOLDS.values())

CLASS_NAMES_ES = {
    1: "Motocicleta",
    2: "Auto",
    3: "Motocicleta",
    5: "Autobus",
    7: "Camion",
}

DISPLAY_ORDER = [
    ("Motocicleta", [1, 3]),
    ("Auto", [2]),
    ("Autobus", [5]),
    ("Camion", [7]),
]

LINE_COLORS = [
    (0, 255, 255), (255, 120, 0), (0, 200, 0),
    (255, 0, 180), (0, 140, 255), (200, 200, 0)
]

CLASS_BOX_COLORS = {
    1: (0, 220, 255),
    2: (40, 220, 70),
    3: (0, 220, 255),
    5: (255, 170, 0),
    7: (255, 90, 90),
}

STOPPED_MOVE_TOLERANCE_PX = 14.0
STOPPED_SECONDS_THRESHOLD = 8.0

def crear_tracker_trafico_yaml() -> str:
    yaml_content = """tracker_type: bytetrack
track_high_thresh: 0.30
track_low_thresh: 0.10
new_track_thresh: 0.25
track_buffer: 60
match_thresh: 0.80
fuse_score: True
"""
    tracker_path = os.path.join(tempfile.gettempdir(), "bytetrack_trafico_sicv.yaml")
    with open(tracker_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    return tracker_path


def mejorar_contraste_frame(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    merged = cv2.merge((l_eq, a_ch, b_ch))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def segments_intersect(a, b, c, d):
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(p, q, r):
        return (
            min(p[0], r[0]) - 1e-6 <= q[0] <= max(p[0], r[0]) + 1e-6
            and min(p[1], r[1]) - 1e-6 <= q[1] <= max(p[1], r[1]) + 1e-6
        )

    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)

    if ((o1 > 0 and o2 < 0) or (o1 < 0 and o2 > 0)) and \
       ((o3 > 0 and o4 < 0) or (o3 < 0 and o4 > 0)):
        return True
    if abs(o1) <= 1e-6 and on_segment(a, c, b):
        return True
    if abs(o2) <= 1e-6 and on_segment(a, d, b):
        return True
    if abs(o3) <= 1e-6 and on_segment(c, a, d):
        return True
    if abs(o4) <= 1e-6 and on_segment(c, b, d):
        return True
    return False


def track_crosses_polyline(previous_point, current_point, polyline):
    if previous_point is None or len(polyline) < 2:
        return False
    a = (float(previous_point[0]), float(previous_point[1]))
    b = (float(current_point[0]), float(current_point[1]))
    for p1, p2 in zip(polyline[:-1], polyline[1:]):
        c = (float(p1[0]), float(p1[1]))
        d = (float(p2[0]), float(p2[1]))
        if segments_intersect(a, b, c, d):
            return True
    return False


def normalizar_clase_vehiculo(cls_id: int) -> int:
    if INCLUDE_BICYCLE_AS_MOTO and cls_id == 1:
        return 3
    return cls_id


class IncidentStore:
    def __init__(self, base_dir=None, db_dir=None):
        self.base_dir = Path(base_dir) if base_dir else (BASE_DIR / "SIV" / "incidencias")
        self.db_dir = Path(db_dir) if db_dir else (BASE_DIR / "SIV" / "database")
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.db_dir / "sicv_incidencias.db"
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock, self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS incidencias (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    codigo TEXT UNIQUE NOT NULL,
                    fecha TEXT NOT NULL,
                    hora TEXT NOT NULL,
                    timestamp_iso TEXT NOT NULL,
                    tipo_evento TEXT NOT NULL,
                    zona TEXT NOT NULL,
                    fuente TEXT,
                    track_id INTEGER,
                    confianza REAL,
                    duracion_segundos REAL,
                    captura_original TEXT NOT NULL,
                    captura_anotada TEXT NOT NULL,
                    estado TEXT NOT NULL DEFAULT 'Pendiente',
                    observaciones TEXT DEFAULT ''
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_incidencias_fecha "
                "ON incidencias(fecha, hora)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_incidencias_tipo "
                "ON incidencias(tipo_evento)"
            )

    def registrar(
        self, frame_original, frame_anotado, tipo_evento, zona,
        fuente="", track_id=None, confianza=None,
        duracion_segundos=None, observaciones=""
    ):
        if frame_original is None or frame_anotado is None:
            raise ValueError("No hay fotogramas disponibles para guardar la incidencia.")

        now = datetime.now().astimezone()
        codigo = f"INC-{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6].upper()}"
        folder = self.base_dir / now.strftime("%Y") / now.strftime("%m") / now.strftime("%d") / codigo
        folder.mkdir(parents=True, exist_ok=True)

        original_path = folder / "original.jpg"
        annotated_path = folder / "anotada.jpg"

        if not cv2.imwrite(str(original_path), frame_original, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise IOError("No se pudo guardar la captura original.")
        if not cv2.imwrite(str(annotated_path), frame_anotado, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise IOError("No se pudo guardar la captura anotada.")

        record = {
            "codigo": codigo,
            "fecha": now.strftime("%Y-%m-%d"),
            "hora": now.strftime("%H:%M:%S"),
            "timestamp_iso": now.isoformat(timespec="seconds"),
            "tipo_evento": str(tipo_evento),
            "zona": str(zona or "Zona no especificada"),
            "fuente": str(fuente or ""),
            "track_id": track_id,
            "confianza": confianza,
            "duracion_segundos": duracion_segundos,
            "captura_original": str(original_path.resolve()),
            "captura_anotada": str(annotated_path.resolve()),
            "estado": "Pendiente",
            "observaciones": str(observaciones or ""),
        }

        with self._lock, self._connect() as conn:
            conn.execute("""
                INSERT INTO incidencias (
                    codigo, fecha, hora, timestamp_iso, tipo_evento, zona,
                    fuente, track_id, confianza, duracion_segundos,
                    captura_original, captura_anotada, estado, observaciones
                ) VALUES (
                    :codigo, :fecha, :hora, :timestamp_iso, :tipo_evento, :zona,
                    :fuente, :track_id, :confianza, :duracion_segundos,
                    :captura_original, :captura_anotada, :estado, :observaciones
                )
            """, record)

        return record

    def listar_recientes(self, limite=100, estado=None):
        query = "SELECT * FROM incidencias"
        params = []
        if estado and estado != "Todos":
            query += " WHERE estado = ?"
            params.append(estado)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(int(limite))
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def actualizar_estado(self, codigo, estado):
        permitidos = {"Pendiente", "Revisada", "Descartada"}
        if estado not in permitidos:
            raise ValueError("Estado de incidencia no válido.")
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE incidencias SET estado=? WHERE codigo=?",
                (estado, codigo)
            )

    def exportar_csv(self, destino_csv):
        registros = self.listar_recientes(limite=5000, estado="Todos")
        campos = [
            "codigo", "fecha", "hora", "tipo_evento", "zona",
            "fuente", "estado", "observaciones", "captura_original", "captura_anotada"
        ]
        with open(destino_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(registros)
        return len(registros)


class VideoWorker(QThread):
    frame_ready = Signal(QImage)
    counts_ready = Signal(dict)
    status_ready = Signal(str)
    error_ready = Signal(str)
    incident_ready = Signal(dict)

    def __init__(
        self,
        source_mode="video",
        video_path="",
        camera_index=0,
        loop_video=True,
        model_path=MODEL_PATH,
        img_size=IMG_SIZE,
        moto_conf=0.18,
        enhance_contrast=False,
        incident_store=None,
    ):
        super().__init__()
        self.source_mode = source_mode
        self.video_path = video_path
        self.camera_index = camera_index
        self.loop_video = loop_video
        self.model_path = model_path
        self.img_size = img_size
        self.moto_conf = moto_conf
        self.enhance_contrast = enhance_contrast
        self.incident_store = incident_store
        self._latest_frames_lock = threading.Lock()
        self.latest_original_frame = None
        self.latest_annotated_frame = None

        self._running = True
        self._lock = threading.Lock()
        self._lines = []
        self._model = None
        self.counts = defaultdict(lambda: defaultdict(int))
        self.counted_pairs = set()
        self.previous_points = {}
        self.frame_number = 0
        self.auto_detect_stopped = True
        self.track_motion_state = {}
        self.reported_stopped_tracks = set()

    def obtener_ultimos_fotogramas(self):
        with self._latest_frames_lock:
            original = None if self.latest_original_frame is None else self.latest_original_frame.copy()
            anotado = None if self.latest_annotated_frame is None else self.latest_annotated_frame.copy()
        return original, anotado

    def registrar_incidencia(
        self, tipo_evento, zona, track_id=None, confianza=None,
        duracion_segundos=None, observaciones=""
    ):
        if self.incident_store is None:
            raise RuntimeError("El registro de incidencias no está inicializado.")
        original, anotado = self.obtener_ultimos_fotogramas()
        fuente = self.video_path if self.source_mode == "video" else f"Webcam índice {self.camera_index}"
        record = self.incident_store.registrar(
            original, anotado, tipo_evento, zona, fuente,
            track_id, confianza, duracion_segundos, observaciones
        )
        self.incident_ready.emit(record)
        return record

    def set_lines(self, lines):
        with self._lock:
            self._lines = [
                {"id": item["id"], "name": item["name"],
                 "points": [tuple(p) for p in item["points"]]}
                for item in lines
            ]

    def set_moto_conf(self, value: float):
        with self._lock:
            self.moto_conf = float(value)

    def set_enhance_contrast(self, enabled: bool):
        with self._lock:
            self.enhance_contrast = bool(enabled)

    def set_auto_detect_stopped(self, enabled: bool):
        with self._lock:
            self.auto_detect_stopped = bool(enabled)

    def reset_counts_data(self):
        with self._lock:
            self.counts.clear()
            self.counted_pairs.clear()
            self.previous_points.clear()
            self.track_motion_state.clear()
            self.reported_stopped_tracks.clear()

    def stop(self):
        self._running = False
        self.wait(3000)

    def _abrir_fuente(self):
        if self.source_mode == "video":
            if not self.video_path or not os.path.exists(self.video_path):
                return None, f"Selecciona o especifica un archivo de video válido: {self.video_path}"
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                return None, f"No se pudo abrir el archivo de video: {self.video_path}"
            return cap, None

        if self.source_mode == "rtsp":
            if not self.stream_url:
                return None, "Especifica una URL RTSP o HTTP válida para la cámara."
            if sys.platform.startswith("win"):
                cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(self.stream_url)
            if not cap.isOpened():
                return None, f"No se pudo conectar al flujo de la cámara: {self.stream_url}"
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap, None

        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(self.camera_index)
        else:
            cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            return None, (
                f"No se pudo abrir la cámara con índice {self.camera_index}. "
                "Prueba otro índice (0 o 1) y verifica que ninguna otra "
                "aplicación esté usando la cámara."
            )

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        return cap, None
    
    def run(self):
        cap = None
        try:
            self.status_ready.emit(f"Cargando modelo {self.model_path}...")
            self._model = YOLO(self.model_path)
            tracker_yaml = crear_tracker_trafico_yaml()

            device = 0 if torch.cuda.is_available() else "cpu"
            use_half = torch.cuda.is_available()
            self.status_ready.emit(
                f"Modelo listo ({self.model_path} @ {self.img_size}px | {'GPU' if use_half else 'CPU'})."
            )

            cap, err = self._abrir_fuente()
            if err:
                self.error_ready.emit(err)
                return

            is_video_file = (self.source_mode == "video")
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_delay_ms = int(1000 / fps) if (is_video_file and fps and fps > 1) else 15

            if is_video_file:
                self.status_ready.emit(f"Reproduciendo video: {self.video_path}")
            else:
                self.status_ready.emit("Cámara conectada. Conteo en ejecución.")

            while self._running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if is_video_file and self.loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.reset_counts_data()
                        continue
                    if is_video_file:
                        self.status_ready.emit("Video finalizado.")
                        break
                    self.msleep(10)
                    continue

                self.frame_number += 1
                frame_original = frame.copy()
                h, w = frame.shape[:2]

                with self._lock:
                    lines_snapshot = [
                        {"id": ln["id"], "name": ln["name"],
                         "points": list(ln["points"])}
                        for ln in self._lines
                    ]
                    current_moto_conf = self.moto_conf
                    apply_clahe = self.enhance_contrast

                infer_frame = mejorar_contraste_frame(frame) if apply_clahe else frame

                thresholds = dict(CLASS_CONF_THRESHOLDS)
                thresholds[1] = current_moto_conf
                thresholds[3] = current_moto_conf
                base_conf = min(thresholds.values())

                detections = []
                if self.frame_number % INFER_EVERY_N_FRAMES == 0:
                    results = self._model.track(
                        source=infer_frame,
                        persist=True,
                        tracker=tracker_yaml,
                        classes=VEHICLE_CLASSES,
                        conf=base_conf,
                        iou=0.50,
                        imgsz=self.img_size,
                        device=device,
                        half=use_half,
                        verbose=False,
                    )

                    if results and results[0].boxes is not None:
                        boxes = results[0].boxes
                        if boxes.id is not None:
                            xyxy = boxes.xyxy.cpu().numpy()
                            cls_ids = boxes.cls.cpu().numpy().astype(int)
                            confs = boxes.conf.cpu().numpy()
                            track_ids = boxes.id.cpu().numpy().astype(int)

                            for box, raw_cls_id, conf, track_id in zip(
                                xyxy, cls_ids, confs, track_ids
                            ):
                                raw_cls_id = int(raw_cls_id)
                                conf = float(conf)
                                min_req = thresholds.get(raw_cls_id, 0.35)
                                if conf < min_req:
                                    continue

                                cls_id = normalizar_clase_vehiculo(raw_cls_id)
                                x1, y1, x2, y2 = map(int, box)

                                if (x2 - x1) < 10 or (y2 - y1) < 10:
                                    continue

                                anchor = (int((x1 + x2) / 2), int(y2))
                                detections.append(
                                    (x1, y1, x2, y2, anchor, cls_id, conf, int(track_id))
                                )

                                now_mono = time.monotonic()
                                is_stopped_now = False
                                stopped_elapsed = 0.0

                                with self._lock:
                                    state = self.track_motion_state.get(int(track_id))
                                    if state is None:
                                        self.track_motion_state[int(track_id)] = {
                                            "ref_point": anchor,
                                            "since": now_mono,
                                            "cls_id": cls_id,
                                            "conf": conf,
                                        }
                                    else:
                                        ref_x, ref_y = state["ref_point"]
                                        dist = float(np.hypot(anchor[0] - ref_x, anchor[1] - ref_y))
                                        state["cls_id"] = cls_id
                                        state["conf"] = conf
                                        if dist > STOPPED_MOVE_TOLERANCE_PX:
                                            state["ref_point"] = anchor
                                            state["since"] = now_mono
                                        else:
                                            stopped_elapsed = now_mono - state["since"]
                                            if stopped_elapsed >= STOPPED_SECONDS_THRESHOLD:
                                                is_stopped_now = True

                                    auto_stopped_enabled = self.auto_detect_stopped
                                    already_reported = int(track_id) in self.reported_stopped_tracks
                                    if is_stopped_now and auto_stopped_enabled and not already_reported:
                                        self.reported_stopped_tracks.add(int(track_id))

                                    previous = self.previous_points.get(int(track_id))
                                    for line in lines_snapshot:
                                        line_id = line["id"]
                                        pair = (int(track_id), line_id)
                                        pixel_line = [
                                            (int(px * w), int(py * h))
                                            for px, py in line["points"]
                                        ]
                                        if pair not in self.counted_pairs and \
                                           track_crosses_polyline(previous, anchor, pixel_line):
                                            self.counted_pairs.add(pair)
                                            self.counts[line_id][cls_id] += 1
                                    self.previous_points[int(track_id)] = anchor

                                if is_stopped_now and auto_stopped_enabled and not already_reported:
                                    annotated_preview = frame.copy()
                                    cv2.rectangle(annotated_preview, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                    cv2.putText(
                                        annotated_preview,
                                        f"VEHICULO DETENIDO ID:{track_id} ({stopped_elapsed:.1f}s)",
                                        (x1, max(22, y1 - 24)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.55,
                                        (0, 0, 255),
                                        2,
                                    )
                                    fuente_txt = (
                                        self.video_path
                                        if self.source_mode == "video"
                                        else f"Webcam índice {self.camera_index}"
                                    )
                                    if self.incident_store is not None:
                                        record = self.incident_store.registrar(
                                            frame_original=frame_original,
                                            frame_anotado=annotated_preview,
                                            tipo_evento="Vehículo detenido",
                                            zona=f"Carril / zona cámara (ID {track_id})",
                                            fuente=fuente_txt,
                                            track_id=int(track_id),
                                            confianza=round(conf, 3),
                                            duracion_segundos=round(stopped_elapsed, 1),
                                            observaciones=(
                                                f"Detección automática: {CLASS_NAMES_ES.get(cls_id, cls_id)} "
                                                f"sin desplazamiento mayor a {STOPPED_MOVE_TOLERANCE_PX:.0f}px "
                                                f"durante {stopped_elapsed:.1f}s."
                                            ),
                                        )
                                        self.incident_ready.emit(record)

                for index, line in enumerate(lines_snapshot):
                    color = LINE_COLORS[index % len(LINE_COLORS)]
                    pts = [(int(x * w), int(y * h)) for x, y in line["points"]]
                    if len(pts) >= 2:
                        for p1, p2 in zip(pts[:-1], pts[1:]):
                            cv2.line(frame, p1, p2, color, 3, cv2.LINE_AA)
                    for point in pts:
                        cv2.circle(frame, point, 5, color, -1)
                    if pts:
                        cv2.putText(
                            frame, line["name"], (pts[0][0] + 8, pts[0][1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                        )

                for x1, y1, x2, y2, anchor, cls_id, conf, track_id in detections:
                    box_color = CLASS_BOX_COLORS.get(cls_id, (40, 220, 70))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.circle(frame, anchor, 4, (0, 0, 255), -1)
                    label = f"ID:{track_id} {CLASS_NAMES_ES.get(cls_id, cls_id)} {conf:.2f}"
                    cv2.putText(
                        frame, label, (x1, max(18, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, box_color, 2
                    )

                with self._lock:
                    total = sum(
                        sum(class_counts.values()) for class_counts in self.counts.values()
                    )
                    motos_total = sum(
                        self.counts[ln["id"]][3] for ln in lines_snapshot
                    )
                    snapshot = {
                        line["id"]: {
                            "name": line["name"],
                            "total": sum(self.counts[line["id"]].values()),
                            "classes": dict(self.counts[line["id"]]),
                        }
                        for line in lines_snapshot
                    }

                cv2.rectangle(frame, (8, 8), (430, 47), (25, 25, 25), -1)
                cv2.putText(
                    frame, f"TOTAL: {total}  |  MOTOS: {motos_total}", (18, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2
                )

                with self._latest_frames_lock:
                    self.latest_original_frame = frame_original
                    self.latest_annotated_frame = frame.copy()

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = QImage(
                    rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888
                ).copy()
                self.frame_ready.emit(image)
                self.counts_ready.emit(snapshot)

                if is_video_file:
                    self.msleep(min(frame_delay_ms, 50))

        except Exception as exc:
            self.error_ready.emit(f"Error durante el procesamiento: {exc}")
        finally:
            if cap is not None:
                cap.release()
            self.status_ready.emit("Fuente detenida.")


class VideoCanvas(QLabel):
    point_added = Signal(float, float)

    def __init__(self):
        super().__init__("Inicia el video o la cámara para mostrar la imagen")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 270)
        self.setStyleSheet("background:#171b22; color:#d7dde5; border:1px solid #394150;")
        self.current_image = None
        self.pending_points = []

    def set_image(self, image):
        self.current_image = image
        self.update()

    def set_pending_points(self, points):
        self.pending_points = list(points)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.current_image is None:
            return
        painter = QPainter(self)
        scaled = self.current_image.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawImage(x, y, scaled)

        if self.pending_points:
            pen = QPen(QColor(255, 80, 80), 3)
            painter.setPen(pen)
            for px, py in self.pending_points:
                sx = x + int(px * scaled.width())
                sy = y + int(py * scaled.height())
                painter.drawEllipse(sx - 5, sy - 5, 10, 10)
            if len(self.pending_points) >= 2:
                for a, b in zip(self.pending_points[:-1], self.pending_points[1:]):
                    painter.drawLine(
                        x + int(a[0] * scaled.width()),
                        y + int(a[1] * scaled.height()),
                        x + int(b[0] * scaled.width()),
                        y + int(b[1] * scaled.height())
                    )
        painter.end()

    def mousePressEvent(self, event):
        if self.current_image is None or event.button() != Qt.MouseButton.LeftButton:
            return
        iw, ih = self.current_image.width(), self.current_image.height()
        if iw <= 0 or ih <= 0:
            return
        scale = min(self.width() / iw, self.height() / ih)
        dw, dh = int(iw * scale), int(ih * scale)
        if dw <= 0 or dh <= 0:
            return
        ox, oy = (self.width() - dw) // 2, (self.height() - dh) // 2
        px, py = event.position().x(), event.position().y()
        if not (ox <= px <= ox + dw and oy <= py <= oy + dh):
            return
        nx = min(1.0, max(0.0, (px - ox) / dw))
        ny = min(1.0, max(0.0, (py - oy) / dh))
        self.point_added.emit(nx, ny)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SICV - Conteo inteligente de vehículos")
        self.resize(1120, 680)
        self.worker = None
        self.lines = []
        self.pending_points = []
        self.next_line_id = 1
        self.incident_store = IncidentStore()
        self._last_incident_popup = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 6)
        root.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("SICV | Sistema Inteligente de Conteo Vehicular")
        title.setStyleSheet("font-size:17px; font-weight:700;")
        header.addWidget(title)
        header.addStretch()
        self.status_label = QLabel("Fuente detenida")
        self.status_label.setStyleSheet("color:#a66b00; font-weight:600;")
        header.addWidget(self.status_label)
        root.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        self.canvas = VideoCanvas()
        self.canvas.point_added.connect(self.add_point)
        splitter.addWidget(self.canvas)

        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setFrameShape(QFrame.Shape.NoFrame)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_scroll.setMinimumWidth(330)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(4, 0, 8, 4)
        side_layout.setSpacing(8)

        source_group = QGroupBox("Fuente de video / cámara")
        source_layout = QGridLayout(source_group)
        source_layout.setContentsMargins(8, 10, 8, 8)
        source_layout.setVerticalSpacing(5)

        source_layout.addWidget(QLabel("Modo:"), 0, 0)
        self.mode_source_combo = QComboBox()
        self.mode_source_combo.addItem("Archivo de video", "video")
        self.mode_source_combo.addItem("Cámara web", "webcam")
        idx_mode = 0 if SOURCE_MODE == "video" else 1
        self.mode_source_combo.setCurrentIndex(idx_mode)
        self.mode_source_combo.currentIndexChanged.connect(self._on_source_mode_changed)
        source_layout.addWidget(self.mode_source_combo, 0, 1, 1, 2)

        source_layout.addWidget(QLabel("Video:"), 1, 0)
        self.video_path_edit = QLineEdit(VIDEO_PATH)
        source_layout.addWidget(self.video_path_edit, 1, 1)
        self.browse_button = QPushButton("Buscar...")
        self.browse_button.clicked.connect(self.browse_video)
        source_layout.addWidget(self.browse_button, 1, 2)

        source_layout.addWidget(QLabel("Índice cam:"), 2, 0)
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 10)
        self.camera_spin.setValue(CAMERA_INDEX)
        source_layout.addWidget(self.camera_spin, 2, 1, 1, 2)

        self.start_button = QPushButton("Iniciar")
        self.start_button.clicked.connect(self.start_source)
        self.stop_button = QPushButton("Detener")
        self.stop_button.clicked.connect(self.stop_source)
        self.stop_button.setEnabled(False)
        source_layout.addWidget(self.start_button, 3, 0, 1, 2)
        source_layout.addWidget(self.stop_button, 3, 2)

        side_layout.addWidget(source_group)

        detect_group = QGroupBox("Ajustes de detección (Motos y precisión)")
        detect_layout = QGridLayout(detect_group)
        detect_layout.setContentsMargins(8, 10, 8, 8)
        detect_layout.setVerticalSpacing(5)

        detect_layout.addWidget(QLabel("Modelo:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItem("yolo11s.pt (Recomendado motos)", "yolo11s.pt")
        self.model_combo.addItem("yolo11n.pt (Más rápido)", "yolo11n.pt")
        self.model_combo.addItem("yolo11m.pt (Alta precisión)", "yolo11m.pt")
        detect_layout.addWidget(self.model_combo, 0, 1)

        detect_layout.addWidget(QLabel("Resolución:"), 1, 0)
        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItem("960 px (Mejor para motos)", 960)
        self.imgsz_combo.addItem("1280 px (Motos lejanas / GPU)", 1280)
        self.imgsz_combo.addItem("640 px (Más rápido en CPU)", 640)
        detect_layout.addWidget(self.imgsz_combo, 1, 1)

        detect_layout.addWidget(QLabel("Sensibilidad motos:"), 2, 0)
        self.moto_conf_combo = QComboBox()
        self.moto_conf_combo.addItem("Alta (conf >= 0.15)", 0.15)
        self.moto_conf_combo.addItem("Balanceada (conf >= 0.18)", 0.18)
        self.moto_conf_combo.addItem("Estricta (conf >= 0.25)", 0.25)
        self.moto_conf_combo.setCurrentIndex(1)
        self.moto_conf_combo.currentIndexChanged.connect(self._on_moto_conf_changed)
        detect_layout.addWidget(self.moto_conf_combo, 2, 1)

        self.clahe_check = QCheckBox("Realzar contraste (sombras / poca luz)")
        self.clahe_check.stateChanged.connect(self._on_clahe_changed)
        detect_layout.addWidget(self.clahe_check, 3, 0, 1, 2)

        side_layout.addWidget(detect_group)

        draw_group = QGroupBox("Trazado de líneas")
        draw_layout = QVBoxLayout(draw_group)
        draw_layout.setContentsMargins(8, 10, 8, 8)
        draw_layout.setSpacing(5)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Línea recta (2 clics)", "straight")
        self.mode_combo.addItem("Línea poligonal (varios puntos)", "polyline")
        draw_layout.addWidget(QLabel("Tipo de trazo:"))
        draw_layout.addWidget(self.mode_combo)
        draw_layout.addWidget(QLabel("Haz clic sobre el video para colocar puntos."))

        draw_btns = QHBoxLayout()
        self.finish_button = QPushButton("Finalizar trazo")
        self.finish_button.clicked.connect(self.finish_line)
        self.finish_button.setEnabled(False)
        self.cancel_button = QPushButton("Cancelar trazo")
        self.cancel_button.clicked.connect(self.cancel_line)
        self.cancel_button.setEnabled(False)
        draw_btns.addWidget(self.finish_button)
        draw_btns.addWidget(self.cancel_button)
        draw_layout.addLayout(draw_btns)
        side_layout.addWidget(draw_group)

        lines_group = QGroupBox("Líneas configuradas")
        lines_layout = QVBoxLayout(lines_group)
        lines_layout.setContentsMargins(8, 10, 8, 8)
        lines_layout.setSpacing(5)
        self.lines_list = QListWidget()
        self.lines_list.setMaximumHeight(85)
        lines_layout.addWidget(self.lines_list)
        self.delete_button = QPushButton("Eliminar línea seleccionada")
        self.delete_button.clicked.connect(self.delete_selected_line)
        lines_layout.addWidget(self.delete_button)
        side_layout.addWidget(lines_group)

        stats_group = QGroupBox("Conteo por línea")
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setContentsMargins(8, 10, 8, 8)
        self.stats_label = QLabel("Aún no hay cruces registrados.")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        side_layout.addWidget(stats_group)

        incidents_group = QGroupBox("Centro de incidencias")
        incidents_layout = QVBoxLayout(incidents_group)
        incidents_layout.setContentsMargins(8, 10, 8, 8)
        incidents_layout.setSpacing(5)

        self.incident_type_combo = QComboBox()
        self.incident_type_combo.addItems([
            "Vehículo detenido", "Sentido contrario",
            "Objeto en carretera", "Invasión de carril", "Otro"
        ])
        incidents_layout.addWidget(QLabel("Tipo de incidencia:"))
        incidents_layout.addWidget(self.incident_type_combo)

        self.incident_zone_edit = QLineEdit("Zona de cámara / sin carril asignado")
        incidents_layout.addWidget(QLabel("Zona o carril:"))
        incidents_layout.addWidget(self.incident_zone_edit)

        self.auto_stopped_check = QCheckBox("Detectar vehículo detenido automáticamente")
        self.auto_stopped_check.setChecked(True)
        self.auto_stopped_check.stateChanged.connect(self._on_auto_stopped_changed)
        incidents_layout.addWidget(self.auto_stopped_check)

        self.capture_test_button = QPushButton("Registrar captura de prueba")
        self.capture_test_button.setToolTip(
            "Guarda el fotograma actual como incidencia de prueba. "
            "No simula una detección automática."
        )
        self.capture_test_button.clicked.connect(self.registrar_captura_prueba)
        self.capture_test_button.setEnabled(False)
        incidents_layout.addWidget(self.capture_test_button)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Estado:"))
        self.incident_filter_combo = QComboBox()
        self.incident_filter_combo.addItems(["Todos", "Pendiente", "Revisada", "Descartada"])
        self.incident_filter_combo.currentTextChanged.connect(self.cargar_historial_incidencias)
        filter_row.addWidget(self.incident_filter_combo, 1)
        incidents_layout.addLayout(filter_row)

        self.incidents_list = QListWidget()
        self.incidents_list.setMinimumHeight(110)
        self.incidents_list.setMaximumHeight(160)
        self.incidents_list.itemSelectionChanged.connect(self.mostrar_detalle_incidencia)
        incidents_layout.addWidget(QLabel("Historial reciente:"))
        incidents_layout.addWidget(self.incidents_list)

        self.incident_detail_label = QLabel("Selecciona una incidencia para ver sus datos.")
        self.incident_detail_label.setWordWrap(True)
        incidents_layout.addWidget(self.incident_detail_label)

        incident_actions = QHBoxLayout()
        self.open_capture_button = QPushButton("Abrir captura")
        self.open_capture_button.clicked.connect(self.abrir_captura_seleccionada)
        self.open_capture_button.setEnabled(False)
        incident_actions.addWidget(self.open_capture_button)

        self.review_button = QPushButton("Marcar revisada")
        self.review_button.clicked.connect(self.marcar_incidencia_revisada)
        self.review_button.setEnabled(False)
        incident_actions.addWidget(self.review_button)
        incidents_layout.addLayout(incident_actions)

        self.export_csv_button = QPushButton("Exportar incidencias a CSV")
        self.export_csv_button.clicked.connect(self.exportar_incidencias_csv)
        incidents_layout.addWidget(self.export_csv_button)

        side_layout.addWidget(incidents_group)

        self.reset_button = QPushButton("Reiniciar conteos")
        self.reset_button.clicked.connect(self.reset_counts)
        side_layout.addWidget(self.reset_button)
        side_layout.addStretch()

        side_scroll.setWidget(side)
        splitter.addWidget(side_scroll)
        splitter.setSizes([750, 350])

        footer = QLabel(
            "Nota: las motocicletas se resaltan en color cian/amarillo con umbral "
            "de confianza independiente para evitar perderlas."
        )
        footer.setStyleSheet("color:#687386; padding:2px 4px; font-size:11px;")
        root.addWidget(footer)

        self._on_source_mode_changed()
        self.cargar_historial_incidencias()

    def _on_source_mode_changed(self):
        is_video = (self.mode_source_combo.currentData() == "video")
        self.video_path_edit.setEnabled(is_video)
        self.browse_button.setEnabled(is_video)
        self.camera_spin.setEnabled(not is_video)

    def _on_moto_conf_changed(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.set_moto_conf(self.moto_conf_combo.currentData())

    def _on_clahe_changed(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.set_enhance_contrast(self.clahe_check.isChecked())

    def _on_auto_stopped_changed(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.set_auto_detect_stopped(self.auto_stopped_check.isChecked())

    def browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar video de prueba",
            str(BASE_DIR),
            "Archivos de video (*.mp4 *.avi *.mov *.mkv);;Todos los archivos (*.*)"
        )
        if file_path:
            self.video_path_edit.setText(file_path)

    def start_source(self):
        if self.worker is not None and self.worker.isRunning():
            return
        mode = self.mode_source_combo.currentData()
        video_path = self.video_path_edit.text().strip()
        cam_idx = self.camera_spin.value()
        model_path = self.model_combo.currentData()
        img_size = self.imgsz_combo.currentData()
        moto_conf = self.moto_conf_combo.currentData()
        enhance_contrast = self.clahe_check.isChecked()

        self.worker = VideoWorker(
            source_mode=mode,
            video_path=video_path,
            camera_index=cam_idx,
            loop_video=LOOP_VIDEO,
            model_path=model_path,
            img_size=img_size,
            moto_conf=moto_conf,
            enhance_contrast=enhance_contrast,
            incident_store=self.incident_store,
        )
        self.worker.frame_ready.connect(self.canvas.set_image)
        self.worker.counts_ready.connect(self.update_stats)
        self.worker.status_ready.connect(self.status_label.setText)
        self.worker.error_ready.connect(self.show_error)
        self.worker.incident_ready.connect(self.on_incident_created)
        self.worker.set_lines(self.lines)
        self.worker.set_auto_detect_stopped(self.auto_stopped_check.isChecked())
        self.worker.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.capture_test_button.setEnabled(True)

    def stop_source(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.capture_test_button.setEnabled(False)

    def registrar_captura_prueba(self):
        if self.worker is None or not self.worker.isRunning():
            QMessageBox.information(self, "Sin fuente activa", "Inicia la webcam o un video antes de registrar una captura.")
            return
        tipo = self.incident_type_combo.currentText()
        zona = self.incident_zone_edit.text().strip() or "Zona no especificada"
        try:
            record = self.worker.registrar_incidencia(
                tipo_evento=tipo,
                zona=zona,
                observaciones="Registro de prueba creado manualmente por el operador."
            )
            self.status_label.setText(f"Captura guardada: {record['codigo']}")
        except Exception as exc:
            QMessageBox.critical(self, "No se pudo registrar", str(exc))

    def on_incident_created(self, record):
        self.cargar_historial_incidencias()
        for row in range(self.incidents_list.count()):
            item = self.incidents_list.item(row)
            data = item.data(Qt.ItemDataRole.UserRole)
            if data and data.get("codigo") == record.get("codigo"):
                self.incidents_list.setCurrentItem(item)
                break

        QApplication.beep()
        if self._last_incident_popup is not None:
            self._last_incident_popup.close()
            self._last_incident_popup = None

        popup = QMessageBox(self)
        popup.setIcon(QMessageBox.Icon.Warning)
        popup.setWindowTitle("Nueva incidencia SICV")
        popup.setText(
            f"{record['tipo_evento']}\n"
            f"Código: {record['codigo']}\n"
            f"Fecha y hora: {record['fecha']} {record['hora']}\n"
            f"Zona: {record['zona']}"
        )
        popup.setStandardButtons(QMessageBox.StandardButton.Ok)
        popup.setModal(False)
        popup.show()
        self._last_incident_popup = popup

    def cargar_historial_incidencias(self):
        estado_filtro = self.incident_filter_combo.currentText() if hasattr(self, "incident_filter_combo") else "Todos"
        self.incidents_list.clear()
        for record in self.incident_store.listar_recientes(100, estado=estado_filtro):
            label = f"{record['codigo']} | {record['tipo_evento']} [{record['estado']}]"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, record)
            self.incidents_list.addItem(item)

    def _registro_seleccionado(self):
        items = self.incidents_list.selectedItems()
        return items[0].data(Qt.ItemDataRole.UserRole) if items else None

    def mostrar_detalle_incidencia(self):
        record = self._registro_seleccionado()
        if not record:
            self.incident_detail_label.setText("Selecciona una incidencia para ver sus datos.")
            self.open_capture_button.setEnabled(False)
            self.review_button.setEnabled(False)
            return
        self.incident_detail_label.setText(
            f"<b>{record['codigo']}</b><br>"
            f"Evento: {record['tipo_evento']}<br>"
            f"Fecha: {record['fecha']} · Hora: {record['hora']}<br>"
            f"Zona: {record['zona']}<br>"
            f"Fuente: {record.get('fuente', '')}<br>"
            f"Estado: {record['estado']}"
        )
        self.open_capture_button.setEnabled(os.path.exists(record["captura_original"]))
        self.review_button.setEnabled(record["estado"] == "Pendiente")

    def abrir_captura_seleccionada(self):
        record = self._registro_seleccionado()
        if record:
            path = record.get("captura_anotada") if os.path.exists(record.get("captura_anotada", "")) else record["captura_original"]
            if path and os.path.exists(path):
                QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            else:
                QMessageBox.warning(self, "Captura no encontrada", path or "Ruta no disponible")

    def marcar_incidencia_revisada(self):
        record = self._registro_seleccionado()
        if not record:
            return
        self.incident_store.actualizar_estado(record["codigo"], "Revisada")
        self.cargar_historial_incidencias()
        self.incident_detail_label.setText(
            f"La incidencia {record['codigo']} fue marcada como revisada."
        )

    def exportar_incidencias_csv(self):
        default_name = str(BASE_DIR / f"incidencias_sicv_{datetime.now():%Y%m%d_%H%M%S}.csv")
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar incidencias a CSV",
            default_name,
            "Archivos CSV (*.csv)"
        )
        if not dest:
            return
        try:
            total = self.incident_store.exportar_csv(dest)
            QMessageBox.information(self, "Exportación completada", f"Se exportaron {total} incidencias a:\n{dest}")
        except Exception as exc:
            QMessageBox.critical(self, "Error al exportar", str(exc))

    def add_point(self, x, y):
        self.pending_points.append((x, y))
        self.canvas.set_pending_points(self.pending_points)
        self.finish_button.setEnabled(True)
        self.cancel_button.setEnabled(True)

        if self.mode_combo.currentData() == "straight" and len(self.pending_points) == 2:
            self.finish_line()

    def finish_line(self):
        if len(self.pending_points) < 2:
            QMessageBox.information(self, "Trazo incompleto", "Coloca al menos dos puntos.")
            return
        line_id = self.next_line_id
        self.next_line_id += 1
        line = {
            "id": line_id,
            "name": f"Línea {line_id}",
            "points": list(self.pending_points),
        }
        self.lines.append(line)
        self.lines_list.addItem(f"{line['name']} ({len(line['points'])} puntos)")
        self.pending_points.clear()
        self.canvas.set_pending_points([])
        self.finish_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.sync_lines()

    def cancel_line(self):
        self.pending_points.clear()
        self.canvas.set_pending_points([])
        self.finish_button.setEnabled(False)
        self.cancel_button.setEnabled(False)

    def delete_selected_line(self):
        index = self.lines_list.currentRow()
        if index < 0:
            return
        del self.lines[index]
        self.lines_list.takeItem(index)
        self.sync_lines()

    def sync_lines(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.set_lines(self.lines)

    def update_stats(self, snapshot):
        if not snapshot:
            self.stats_label.setText("Dibuja una línea para comenzar a contar.")
            return
        rows = []
        grand_total = 0
        for _, info in snapshot.items():
            grand_total += info["total"]
            details = []
            for label_name, cls_list in DISPLAY_ORDER:
                amount = sum(info["classes"].get(c, 0) for c in cls_list)
                if amount > 0:
                    details.append(f"{label_name}: {amount}")
            rows.append(
                f"<b>{info['name']}: {info['total']}</b><br>"
                + (", ".join(details) if details else "Sin cruces")
            )
        self.stats_label.setText(
            f"<b>Total general: {grand_total}</b><hr>" + "<br><br>".join(rows)
        )

    def reset_counts(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.reset_counts_data()
            self.stats_label.setText("Conteos reiniciados en vivo.")
        else:
            self.stats_label.setText("Conteos reiniciados. Inicia la fuente.")

    def show_error(self, message):
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Problema", message)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.capture_test_button.setEnabled(False)

    def closeEvent(self, event):
        self.stop_source()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()