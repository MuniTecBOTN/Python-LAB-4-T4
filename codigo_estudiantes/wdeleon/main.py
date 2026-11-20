#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - Sistema de inspección visual de piezas en banda transportadora.

Tecnologías: Python 3 + OpenCV (ORB, BFMatcher/Hamming, prueba de Lowe,
homografía + RANSAC, flujo óptico Lucas-Kanade) y contador por línea virtual.

Estructura esperada en disco:
    main.py
    plantillas/objeto_A/referencia_1.jpg ... (4 o más imágenes por categoría)
    plantillas/objeto_B/...
    plantillas/objeto_C/...
    evidencias/            (se crea automáticamente)

Uso:
    python main.py                       # cámara 0
    python main.py --source 1            # cámara 1
    python main.py --source video.mp4    # archivo de video (pruebas)
    python main.py --headless --max-frames 600 --source video.mp4

Teclas: q/ESC salir | c recalibrar banda vacía | s captura | r reiniciar
contadores | m ver máscara de segmentación.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("inspeccion")

# Etiquetas internas de clasificación
LABEL_UNKNOWN = "UNKNOWN"   # evaluada y NO coincide con ninguna categoría
LABEL_NE = "NE"             # no evaluable (evidencia insuficiente o ambigua)

# Colores BGR
COLOR_OK = (0, 200, 0)       # verde: pieza conocida
COLOR_ALERT = (0, 0, 255)    # rojo: pieza desconocida
COLOR_WARN = (0, 170, 255)   # ámbar: no evaluable
COLOR_TEXT = (255, 255, 255)


# =============================================================================
# 1. CONFIGURACIÓN (todos los parámetros ajustables en un solo lugar)
# =============================================================================
@dataclass
class Config:
    # --- Rutas y registro de referencias ---
    templates_dir: str = "plantillas"
    evidence_dir: str = "evidencias"
    categories: Tuple[str, ...] = ("A", "B", "C")
    folder_pattern: str = "objeto_{cat}"
    image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
    expected_refs_per_category: int = 4
    min_ref_size_px: int = 64          # lado mínimo aceptable de una referencia
    min_ref_keypoints: int = 40        # mínimo de puntos ORB para usar una referencia
    auto_crop_reference: bool = True   # recorta y endereza la pieza (fondo uniforme)

    # --- Captura ---
    source: object = 0                 # índice de cámara o ruta de video
    request_width: int = 1280
    request_height: int = 720
    proc_width: int = 640              # resolución de procesamiento (ancho)
    max_read_failures: int = 30        # lecturas fallidas consecutivas permitidas

    # --- Zona de inspección y línea virtual (fracciones) ---
    roi: Tuple[float, float, float, float] = (0.05, 0.10, 0.95, 0.90)  # x0,y0,x1,y1
    count_axis: str = "x"              # "x": línea vertical | "y": línea horizontal
    count_pos: float = 0.55            # posición de la línea dentro de la ROI (0-1)
    flow_sign: int = +1                # +1: avanza hacia x/y crecientes; -1: inverso

    # --- Localización por segmentación de banda (color Lab) ---
    calib_frames: int = 30             # fotogramas de banda vacía para calibrar
    min_color_dist: float = 18.0       # umbral mínimo de distancia al color de banda
    threshold_margin: float = 1.5      # margen sobre el percentil 99.5 del ruido
    lightness_weight: float = 0.5      # peso de L (reduce efecto de sombras)
    morph_kernel: int = 5
    min_area_frac: float = 0.006       # área mínima de pieza (fracción de ROI)
    max_area_frac: float = 0.45        # área máxima (evita detectar toda la banda)
    border_margin_px: int = 3          # contacto con el borde de la ROI => parcial

    # --- ORB ---
    orb_nfeatures: int = 800
    orb_scale_factor: float = 1.2
    orb_nlevels: int = 8
    orb_fast_threshold: int = 10
    orb_edge_threshold: int = 15
    orb_patch_size: int = 31
    norm_size: int = 256               # lado largo normalizado (referencia y escena)
    use_clahe: bool = True
    scene_pad_px: int = 6

    # --- Comparación (BFMatcher + Lowe) ---
    lowe_ratio: float = 0.75
    max_hamming: int = 64              # descarta correspondencias muy lejanas (de 256 bits)

    # --- Validación geométrica (RANSAC) ---
    ransac_reproj_thresh: float = 3.0
    ransac_max_iters: int = 2000
    ransac_confidence: float = 0.995
    geo_area_ratio: Tuple[float, float] = (0.55, 1.8)
    geo_min_angle: float = 55.0
    geo_max_angle: float = 125.0
    geo_side_ratio: float = 1.7
    geo_center_tol: float = 0.30       # fracción de norm_size
    geo_margin: float = 0.20           # fracción de norm_size

    # --- Criterios de decisión ---
    min_good_matches: int = 15
    min_good_ratio: float = 0.04       # coincidencias / min(kp_ref, kp_escena)
    min_inliers: int = 12
    min_inlier_ratio: float = 0.50
    max_reproj_error: float = 3.0      # px (promedio sobre inliers, escala normalizada)
    weak_inliers: int = 9              # entre weak y min => evidencia parcial (ambigua)
    ambiguity_ratio: float = 0.80      # 2da categoría > 80% del score => ambiguo
    min_scene_keypoints: int = 30
    min_sharpness: float = 40.0        # varianza del Laplaciano dentro de la pieza

    # --- Votación temporal (estabilidad de la categoría) ---
    vote_window: int = 7
    confirm_votes: int = 3
    eval_interval: int = 2             # cada cuántos fotogramas reevaluar con ORB
    max_evals_per_frame: int = 2
    max_evals_per_track: int = 40

    # --- Lucas-Kanade ---
    lk_win: Tuple[int, int] = (21, 21)
    lk_levels: int = 3
    lk_max_corners: int = 40
    lk_quality_level: float = 0.01
    lk_min_distance: int = 5
    lk_fb_error: float = 1.5           # error forward-backward máximo (px)
    lk_ransac: float = 3.0
    lk_min_quality: float = 0.35       # puntos vivos / puntos iniciales
    lk_min_points: int = 4

    # --- Asociación y ciclo de vida de identificadores ---
    assoc_gate_px: float = 40.0
    assoc_area_ratio: Tuple[float, float] = (0.4, 2.5)
    max_lost_frames: int = 25
    show_lost_frames: int = 8

    # --- Interfaz ---
    display_scale: float = 1.5
    draw_homography: bool = False
    audible_alert: bool = False


# =============================================================================
# 2. UTILIDADES
# =============================================================================
def read_image(path: Path) -> Optional[np.ndarray]:
    """Lee una imagen (compatible con rutas con acentos en Windows)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError) as exc:
        log.error("No se pudo leer %s: %s", path, exc)
        return None


def make_orb(cfg: Config):
    return cv2.ORB_create(
        nfeatures=cfg.orb_nfeatures, scaleFactor=cfg.orb_scale_factor,
        nlevels=cfg.orb_nlevels, edgeThreshold=cfg.orb_edge_threshold,
        patchSize=cfg.orb_patch_size, fastThreshold=cfg.orb_fast_threshold)


def keypoints_to_array(kps) -> np.ndarray:
    return np.array([k.pt for k in kps], dtype=np.float32).reshape(-1, 2)


# =============================================================================
# 3. REGISTRO DE REFERENCIAS (12 imágenes -> 3 categorías)
# =============================================================================
@dataclass
class ReferenceImage:
    category: str
    name: str
    pts: np.ndarray              # (N,2) puntos clave
    desc: np.ndarray             # (N,32) descriptores ORB
    corners: np.ndarray          # (4,1,2) esquinas de la referencia normalizada
    size: Tuple[int, int]


class ReferenceRegistry:
    """Carga, valida y precalcula descriptores ORB de todas las referencias."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.orb = make_orb(cfg)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self.refs: Dict[str, List[ReferenceImage]] = {c: [] for c in cfg.categories}

    def load(self) -> bool:
        cfg = self.cfg
        base = Path(cfg.templates_dir)
        if not base.is_dir():
            log.error("No existe la carpeta de plantillas: %s", base.resolve())
            return False
        ok = True
        for cat in cfg.categories:
            folder = base / cfg.folder_pattern.format(cat=cat)
            if not folder.is_dir():
                log.error("Falta la carpeta de la categoría %s: %s", cat, folder)
                ok = False
                continue
            files = sorted(p for p in folder.iterdir()
                           if p.suffix.lower() in cfg.image_exts)
            if len(files) < cfg.expected_refs_per_category:
                log.warning("Categoría %s: %d imágenes (se esperaban %d)",
                            cat, len(files), cfg.expected_refs_per_category)
            for f in files:
                ref = self._build_reference(cat, f)
                if ref is not None:
                    self.refs[cat].append(ref)
            usable = len(self.refs[cat])
            log.info("Categoría %s: %d/%d referencias utilizables (%s)", cat, usable,
                     len(files), ", ".join(str(r.pts.shape[0]) + " kp" for r in self.refs[cat]))
            if usable == 0:
                log.error("Categoría %s sin referencias utilizables.", cat)
                ok = False
        return ok

    def all_refs(self) -> List[ReferenceImage]:
        return [r for refs in self.refs.values() for r in refs]

    def _build_reference(self, cat: str, path: Path) -> Optional[ReferenceImage]:
        cfg = self.cfg
        img = read_image(path)
        if img is None:
            log.error("[%s] imagen inválida o ilegible: %s", cat, path.name)
            return None
        h, w = img.shape[:2]
        if min(h, w) < cfg.min_ref_size_px:
            log.error("[%s] %s demasiado pequeña (%dx%d)", cat, path.name, w, h)
            return None
        if cfg.auto_crop_reference:
            img = self._autocrop_deskew(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scale = cfg.norm_size / max(gray.shape)
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)
        if cfg.use_clahe:
            gray = self.clahe.apply(gray)
        kps, desc = self.orb.detectAndCompute(gray, None)
        if desc is None or len(kps) < cfg.min_ref_keypoints:
            log.warning("[%s] %s descartada: solo %d puntos ORB (mín. %d)",
                        cat, path.name, 0 if desc is None else len(kps), cfg.min_ref_keypoints)
            return None
        gh, gw = gray.shape
        corners = np.float32([[0, 0], [gw, 0], [gw, gh], [0, gh]]).reshape(-1, 1, 2)
        return ReferenceImage(cat, f"{cat}/{path.name}", keypoints_to_array(kps),
                              desc, corners, (gw, gh))

    @staticmethod
    def _autocrop_deskew(img: np.ndarray) -> np.ndarray:
        """Recorta la pieza sobre fondo uniforme y la endereza (rect. mínimo)."""
        h, w = img.shape[:2]
        border = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]]).astype(np.float32)
        bg = np.median(border, axis=0)
        dist = np.linalg.norm(img.astype(np.float32) - bg, axis=2)
        d8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, m = cv2.threshold(cv2.GaussianBlur(d8, (5, 5), 0), 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return img
        c = max(cnts, key=cv2.contourArea)
        frac = cv2.contourArea(c) / float(h * w)
        if frac < 0.05 or frac > 0.98:
            return img
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect).astype(np.float32)
        w0 = float(np.linalg.norm(box[0] - box[1]))
        h0 = float(np.linalg.norm(box[1] - box[2]))
        if w0 < 32 or h0 < 32:
            return img
        dst = np.array([[0, 0], [w0, 0], [w0, h0], [0, h0]], np.float32)
        M = cv2.getPerspectiveTransform(box, dst)
        return cv2.warpPerspective(img, M, (int(round(w0)), int(round(h0))))


# =============================================================================
# 4. LOCALIZACIÓN DE OBJETOS (segmentación de la banda por color)
# =============================================================================
@dataclass
class Candidate:
    contour: np.ndarray                 # (N,1,2) int32, coordenadas del fotograma
    rect: Tuple                         # minAreaRect
    box: np.ndarray                     # (4,2) float32
    bbox: Tuple[int, int, int, int]
    area: float
    centroid: np.ndarray                # (2,) float32
    partial: bool                       # toca el borde de la zona de inspección

    @property
    def long_side(self) -> float:
        return float(max(self.rect[1]))

    def local_mask(self, x0: int, y0: int, w: int, h: int, dilate: int = 0) -> np.ndarray:
        m = np.zeros((h, w), np.uint8)
        cv2.drawContours(m, [self.contour], -1, 255, -1, offset=(-x0, -y0))
        if dilate > 0:
            m = cv2.dilate(m, np.ones((2 * dilate + 1,) * 2, np.uint8))
        return m


class BeltSegmenter:
    """
    Modelo de color de la banda vacía (mediana en Lab) + distancia por píxel.
    Se eligió frente a MOG2 porque la banda está en movimiento (el fondo no es
    estático) y frente a Otsu global porque el umbral se calibra con el ruido
    real de la banda.
    """

    def __init__(self, cfg: Config, roi_px: Tuple[int, int, int, int]):
        self.cfg = cfg
        self.x0, self.y0, self.x1, self.y1 = roi_px
        self.reset()
        self.last_mask: Optional[np.ndarray] = None

    def reset(self):
        self._samples: List[np.ndarray] = []
        self.median: Optional[np.ndarray] = None
        self.threshold: float = 0.0

    @property
    def ready(self) -> bool:
        return self.median is not None

    @property
    def calib_progress(self) -> float:
        return min(1.0, len(self._samples) / max(1, self.cfg.calib_frames))

    def _to_lab(self, bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    def _dist(self, lab: np.ndarray) -> np.ndarray:
        w = np.array([self.cfg.lightness_weight, 1.0, 1.0], np.float32)
        return np.sqrt((((lab - self.median) * w) ** 2).sum(axis=-1))

    def feed_calibration(self, frame: np.ndarray) -> None:
        roi = cv2.GaussianBlur(frame[self.y0:self.y1, self.x0:self.x1], (5, 5), 0)
        self._samples.append(self._to_lab(roi)[::4, ::4].reshape(-1, 3))
        if len(self._samples) >= self.cfg.calib_frames:
            data = np.concatenate(self._samples)
            self.median = np.median(data, axis=0).astype(np.float32)
            noise = np.percentile(self._dist(data), 99.5)
            self.threshold = max(self.cfg.min_color_dist, self.cfg.threshold_margin * float(noise))
            self._samples = []
            log.info("Banda calibrada: color Lab=%s, umbral=%.1f", self.median.round(1), self.threshold)

    def segment(self, frame: np.ndarray) -> List[Candidate]:
        cfg = self.cfg
        roi = cv2.GaussianBlur(frame[self.y0:self.y1, self.x0:self.x1], (5, 5), 0)
        mask = ((self._dist(self._to_lab(roi)) > self.threshold).astype(np.uint8)) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_kernel,) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        self.last_mask = mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rh, rw = mask.shape
        roi_area = float(rh * rw)
        m = cfg.border_margin_px
        out: List[Candidate] = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < cfg.min_area_frac * roi_area or area > cfg.max_area_frac * roi_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            partial = x <= m or y <= m or x + w >= rw - m or y + h >= rh - m
            c_frame = (c + np.array([[[self.x0, self.y0]]], dtype=c.dtype)).astype(np.int32)
            rect = cv2.minAreaRect(c_frame)
            mom = cv2.moments(c_frame)
            if mom["m00"] <= 0:
                continue
            centroid = np.array([mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]], np.float32)
            out.append(Candidate(c_frame, rect, cv2.boxPoints(rect).astype(np.float32),
                                 (x + self.x0, y + self.y0, w, h), area, centroid, partial))
        return out


# =============================================================================
# 5. CLASIFICACIÓN ORB + BFMatcher + Lowe + Homografía/RANSAC
# =============================================================================
@dataclass
class SceneFeatures:
    pts: np.ndarray
    desc: Optional[np.ndarray]
    x0: int
    y0: int
    scale: float
    size: Tuple[int, int]
    box_norm: np.ndarray
    sharpness: float


@dataclass
class RefMatch:
    category: str
    ref_name: str
    n_good: int = 0
    good_ratio: float = 0.0
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    reproj_err: float = float("inf")
    geometry_ok: bool = False
    geo_reason: str = "sin homografía"
    passes: bool = False
    weak: bool = False
    score: float = 0.0
    polygon: Optional[np.ndarray] = None   # polígono proyectado (coords. del fotograma)


@dataclass
class EvalResult:
    label: str
    reason: str = ""
    best: Optional[RefMatch] = None
    elapsed_ms: float = 0.0


class OrbClassifier:
    def __init__(self, cfg: Config, registry: ReferenceRegistry):
        self.cfg = cfg
        self.registry = registry
        self.orb = make_orb(cfg)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)   # distancia de Hamming
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    # ---- preparación de la escena (región candidata normalizada) ----
    def _prepare_scene(self, gray: np.ndarray, cand: Candidate) -> Optional[SceneFeatures]:
        cfg = self.cfg
        x, y, w, h = cand.bbox
        H, W = gray.shape
        pad = cfg.scene_pad_px
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        if x1 - x0 < 8 or y1 - y0 < 8 or cand.long_side < 8:
            return None
        crop = gray[y0:y1, x0:x1]
        mask = cand.local_mask(x0, y0, x1 - x0, y1 - y0, dilate=2)
        # Nitidez: varianza del Laplaciano dentro de la pieza
        lap = cv2.Laplacian(crop, cv2.CV_64F)
        inside = mask > 0
        sharp = float(lap[inside].var()) if inside.any() else 0.0
        s = float(np.clip(cfg.norm_size / cand.long_side, 0.2, 6.0))
        crop_n = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        mask_n = cv2.resize(mask, (crop_n.shape[1], crop_n.shape[0]), interpolation=cv2.INTER_NEAREST)
        if cfg.use_clahe:
            crop_n = self.clahe.apply(crop_n)
        kps, desc = self.orb.detectAndCompute(crop_n, mask_n)
        box_norm = (cand.box - np.array([x0, y0], np.float32)) * s
        return SceneFeatures(keypoints_to_array(kps), desc, x0, y0, s,
                             (crop_n.shape[1], crop_n.shape[0]), box_norm.astype(np.float32), sharp)

    # ---- validación geométrica de la homografía ----
    def _geometry_ok(self, poly: np.ndarray, scene: SceneFeatures) -> Tuple[bool, str]:
        cfg = self.cfg
        if not np.all(np.isfinite(poly)):
            return False, "polígono no finito"
        poly32 = poly.astype(np.float32)
        box_area = cv2.contourArea(scene.box_norm)
        if box_area < 1:
            return False, "región degenerada"
        ratio = cv2.contourArea(poly32) / box_area
        if not (cfg.geo_area_ratio[0] <= ratio <= cfg.geo_area_ratio[1]):
            return False, f"área proyectada {ratio:.2f}"
        if not cv2.isContourConvex(poly32.reshape(-1, 1, 2)):
            return False, "polígono no convexo"
        sides = [float(np.linalg.norm(poly[(i + 1) % 4] - poly[i])) for i in range(4)]
        if min(sides) < 1e-3:
            return False, "lado nulo"
        for a, b in ((0, 2), (1, 3)):
            r = sides[a] / sides[b]
            if not (1 / cfg.geo_side_ratio <= r <= cfg.geo_side_ratio):
                return False, "deformación de lados opuestos"
        for i in range(4):
            v1, v2 = poly[i - 1] - poly[i], poly[(i + 1) % 4] - poly[i]
            cosang = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
            ang = math.degrees(math.acos(cosang))
            if not (cfg.geo_min_angle <= ang <= cfg.geo_max_angle):
                return False, f"ángulo {ang:.0f}°"
        if np.linalg.norm(poly.mean(0) - scene.box_norm.mean(0)) > cfg.geo_center_tol * cfg.norm_size:
            return False, "centro fuera de la región"
        W, H = scene.size
        m = cfg.geo_margin * cfg.norm_size
        if (poly[:, 0].min() < -m or poly[:, 1].min() < -m
                or poly[:, 0].max() > W + m or poly[:, 1].max() > H + m):
            return False, "fuera de los límites"
        return True, "ok"

    # ---- comparación contra una referencia ----
    def _compare(self, ref: ReferenceImage, scene: SceneFeatures) -> RefMatch:
        cfg = self.cfg
        r = RefMatch(ref.category, ref.name)
        if scene.desc is None or len(scene.desc) < 2:
            return r
        knn = self.matcher.knnMatch(ref.desc, scene.desc, k=2)
        good = [p[0] for p in knn if len(p) == 2
                and p[0].distance < cfg.lowe_ratio * p[1].distance
                and p[0].distance <= cfg.max_hamming]
        r.n_good = len(good)
        r.good_ratio = len(good) / max(1, min(len(ref.pts), len(scene.pts)))
        if len(good) < 4:
            return r
        src = ref.pts[[m.queryIdx for m in good]].reshape(-1, 1, 2)
        dst = scene.pts[[m.trainIdx for m in good]].reshape(-1, 1, 2)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, cfg.ransac_reproj_thresh,
                                    maxIters=cfg.ransac_max_iters, confidence=cfg.ransac_confidence)
        if H is None or inl is None:
            return r
        inl = inl.ravel().astype(bool)
        r.n_inliers = int(inl.sum())
        r.inlier_ratio = r.n_inliers / len(good)
        if r.n_inliers < 4:
            return r
        proj = cv2.perspectiveTransform(src[inl], H)
        r.reproj_err = float(np.mean(np.linalg.norm(proj - dst[inl], axis=2)))
        poly = cv2.perspectiveTransform(ref.corners, H).reshape(4, 2)
        r.geometry_ok, r.geo_reason = self._geometry_ok(poly, scene)
        r.polygon = poly / scene.scale + np.array([scene.x0, scene.y0], np.float32)
        r.passes = (r.geometry_ok and r.n_good >= cfg.min_good_matches
                    and r.good_ratio >= cfg.min_good_ratio
                    and r.n_inliers >= cfg.min_inliers
                    and r.inlier_ratio >= cfg.min_inlier_ratio
                    and r.reproj_err <= cfg.max_reproj_error)
        r.weak = r.n_inliers >= cfg.weak_inliers
        if r.passes:
            r.score = r.n_inliers * r.inlier_ratio * max(0.0, 1.0 - 0.5 * r.reproj_err / cfg.max_reproj_error)
        return r

    # ---- evaluación completa de una región candidata ----
    def evaluate(self, gray: np.ndarray, cand: Candidate) -> EvalResult:
        t0 = time.perf_counter()
        cfg = self.cfg

        def done(label, reason="", best=None):
            return EvalResult(label, reason, best, (time.perf_counter() - t0) * 1000)

        # 1) ¿Existe evidencia suficiente para evaluar? (calidad de la región)
        if cand.partial:
            return done(LABEL_NE, "pieza parcialmente fuera de la zona")
        scene = self._prepare_scene(gray, cand)
        if scene is None:
            return done(LABEL_NE, "región demasiado pequeña")
        if scene.sharpness < cfg.min_sharpness:
            return done(LABEL_NE, f"desenfoque/poca textura ({scene.sharpness:.0f})")
        if scene.desc is None or len(scene.pts) < cfg.min_scene_keypoints:
            return done(LABEL_NE, "pocas características ORB")

        # 2) Comparar contra las 12 referencias (4 por categoría)
        matches = [self._compare(ref, scene) for ref in self.registry.all_refs()]

        # 3) Mejor resultado por categoría (la categoría gana por su mejor referencia)
        best_by_cat: Dict[str, RefMatch] = {}
        for m in matches:
            cur = best_by_cat.get(m.category)
            if cur is None or (m.score, m.n_inliers) > (cur.score, cur.n_inliers):
                best_by_cat[m.category] = m
        passing = sorted((m for m in best_by_cat.values() if m.passes),
                         key=lambda m: m.score, reverse=True)

        if passing:
            if len(passing) > 1 and passing[1].score > cfg.ambiguity_ratio * passing[0].score:
                return done(LABEL_NE, f"ambiguo entre {passing[0].category} y {passing[1].category}",
                            passing[0])
            return done(passing[0].category, "coincidencia válida", passing[0])

        top = max(matches, key=lambda m: (m.n_inliers, m.n_good), default=None)
        if any(m.weak for m in matches):
            return done(LABEL_NE, "evidencia parcial (ambigua)", top)
        # Evaluación válida y sin coincidencias ni evidencia parcial: desconocida
        return done(LABEL_UNKNOWN, "sin coincidencia con ninguna categoría", top)


# =============================================================================
# 6. SEGUIMIENTO LUCAS-KANADE
# =============================================================================
class Track:
    def __init__(self, track_id: int, cand: Candidate, cfg: Config):
        self.cfg = cfg
        self.id = track_id
        self.contour = cand.contour
        self.polygon = cand.box.copy()
        self.center = cand.centroid.copy()
        self.center_prev = cand.centroid.copy()
        self.velocity = np.zeros(2, np.float32)
        self.long_side = cand.long_side
        self.area = cand.area
        self.area_reliable = not cand.partial
        self.points: Optional[np.ndarray] = None
        self.n_points_init = 0
        self.quality = 0.0
        self.lost = 0
        self.age = 0
        self.matched_cand: Optional[Candidate] = None
        # Clasificación
        self.votes: Deque[str] = deque(maxlen=cfg.vote_window)
        self.decision: Optional[str] = None     # A/B/C/UNKNOWN una vez confirmada
        self.reason = "evaluando"
        self.evaluating = False         # True si ya hubo votos válidos (A/B/C/UNKNOWN)
        self.last_eval = -999
        self.eval_count = 0
        self.best: Optional[RefMatch] = None
        # Conteo
        self.prev_side: Optional[float] = None
        self.crossed = False
        self.counted = False

    @property
    def tracking_ok(self) -> bool:
        return self.points is not None and len(self.points) >= self.cfg.lk_min_points \
            and self.quality >= self.cfg.lk_min_quality

    def init_points(self, gray: np.ndarray, cand: Candidate) -> None:
        """(Re)inicializa puntos con goodFeaturesToTrack dentro de la pieza."""
        x, y, w, h = cand.bbox
        H, W = gray.shape
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        mask = cand.local_mask(x0, y0, x1 - x0, y1 - y0)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
        pts = cv2.goodFeaturesToTrack(gray[y0:y1, x0:x1], self.cfg.lk_max_corners,
                                      self.cfg.lk_quality_level, self.cfg.lk_min_distance,
                                      mask=mask, blockSize=7)
        if pts is None or len(pts) == 0:
            self.points, self.n_points_init, self.quality = None, 0, 0.0
            return
        self.points = (pts + np.array([[[x0, y0]]], np.float32)).astype(np.float32)
        self.n_points_init = len(self.points)
        self.quality = 1.0

    def lk_update(self, prev_gray: np.ndarray, gray: np.ndarray) -> None:
        """Flujo óptico piramidal con verificación adelante-atrás y RANSAC afín."""
        cfg = self.cfg
        self.center_prev = self.center.copy()
        self.matched_cand = None
        if self.points is None or len(self.points) == 0:
            self.quality = 0.0
            return
        lk = dict(winSize=cfg.lk_win, maxLevel=cfg.lk_levels,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        p0 = self.points
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk)
        if p1 is None:
            self.points, self.quality = None, 0.0
            return
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **lk)
        fb = np.linalg.norm(p0 - p0r, axis=2).ravel()
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < cfg.lk_fb_error)
        if ok.sum() >= 3:
            M, inl = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC,
                                                 ransacReprojThreshold=cfg.lk_ransac)
            if M is not None and inl is not None:
                M = M.astype(np.float32)
                idx = np.flatnonzero(ok)[inl.ravel().astype(bool)]
                self.polygon = cv2.transform(self.polygon.reshape(-1, 1, 2), M).reshape(-1, 2)
                self.center = cv2.transform(self.center.reshape(1, 1, 2), M).reshape(2)
                self.points = p1[idx]
                self.quality = len(self.points) / max(1, self.n_points_init)
                return
        self.points, self.quality = None, 0.0

    def coast(self) -> None:
        """Si LK falla, se extrapola la posición con la última velocidad conocida."""
        self.polygon = self.polygon + self.velocity
        self.center = self.center + self.velocity

    def apply_candidate(self, cand: Candidate, gray: np.ndarray) -> None:
        """Corrige la posición con la segmentación y reinicializa puntos si hace falta."""
        disp = cand.centroid - self.center_prev
        self.velocity = 0.7 * self.velocity + 0.3 * disp.astype(np.float32)
        self.contour, self.polygon, self.center = cand.contour, cand.box.copy(), cand.centroid.copy()
        self.long_side = cand.long_side
        if not cand.partial:
            self.area, self.area_reliable = cand.area, True
        self.matched_cand = cand
        self.lost = 0
        if (not self.tracking_ok) or self.quality < 0.5:
            self.init_points(gray, cand)


# =============================================================================
# 7. CONTADOR POR LÍNEA VIRTUAL
# =============================================================================
class LineCounter:
    KEYS = ("A", "B", "C", LABEL_UNKNOWN)

    def __init__(self, cfg: Config, roi_px: Tuple[int, int, int, int], evidence_dir: Path):
        self.cfg = cfg
        x0, y0, x1, y1 = roi_px
        self.axis = 0 if cfg.count_axis == "x" else 1
        lo, hi = (x0, x1) if self.axis == 0 else (y0, y1)
        self.pos = lo + cfg.count_pos * (hi - lo)
        self.sign = 1 if cfg.flow_sign >= 0 else -1
        self.counts: Dict[str, int] = {k: 0 for k in self.KEYS}
        self.events: List[Tuple[str, int, int, str]] = []
        self.evidence_dir = evidence_dir

    def side(self, center: np.ndarray) -> float:
        return float(center[self.axis] - self.pos) * self.sign

    def update(self, track: Track, frame_idx: int) -> None:
        s = self.side(track.center)
        if track.prev_side is not None and track.prev_side < 0 <= s:
            track.crossed = True                       # cruce en el sentido esperado
        track.prev_side = s
        # Se cuenta UNA sola vez, y solo con categoría confirmada
        if track.crossed and not track.counted and track.decision is not None:
            self.counts[track.decision] += 1
            track.counted = True
            self.events.append((datetime.now().isoformat(timespec="milliseconds"),
                                frame_idx, track.id, track.decision))
            log.info("CONTEO: ID%d -> %s | %s", track.id, track.decision, self.counts)

    def reset(self):
        self.counts = {k: 0 for k in self.KEYS}

    def save_events(self) -> Optional[Path]:
        if not self.events:
            return None
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        path = self.evidence_dir / "eventos_conteo.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "fotograma", "id_seguimiento", "categoria"])
            w.writerows(self.events)
        return path


# =============================================================================
# 8. SISTEMA DE INSPECCIÓN (integra todo el flujo)
# =============================================================================
class InspectionSystem:
    def __init__(self, cfg: Config, registry: ReferenceRegistry, frame_shape: Tuple[int, int]):
        self.cfg = cfg
        h, w = frame_shape
        x0, y0 = int(cfg.roi[0] * w), int(cfg.roi[1] * h)
        x1, y1 = int(cfg.roi[2] * w), int(cfg.roi[3] * h)
        self.roi_px = (x0, y0, x1, y1)
        self.segmenter = BeltSegmenter(cfg, self.roi_px)
        self.classifier = OrbClassifier(cfg, registry)
        self.evidence_dir = Path(cfg.evidence_dir)
        self.counter = LineCounter(cfg, self.roi_px, self.evidence_dir)
        self.tracks: List[Track] = []
        self.prev_gray: Optional[np.ndarray] = None
        self.frame_idx = 0
        self._next_id = 1
        self.timing = {"seg": 0.0, "lk": 0.0, "orb": 0.0, "total": 0.0}
        self.totals: List[float] = []
        self.show_mask = False
        self.alert_events = 0

    # ------------------------------------------------------------------
    def recalibrate(self):
        self.segmenter.reset()
        self.tracks.clear()
        log.info("Recalibrando banda: mantenga la banda vacía.")

    def _associate(self, cands: List[Candidate]):
        cfg = self.cfg
        pairs = []
        for ti, t in enumerate(self.tracks):
            gate = max(cfg.assoc_gate_px, 0.6 * t.long_side) * (1 + 0.25 * t.lost)
            for ci, c in enumerate(cands):
                d = float(np.linalg.norm(t.center - c.centroid))
                if d > gate:
                    continue
                if t.area_reliable and not c.partial:
                    r = c.area / max(t.area, 1.0)
                    if not (cfg.assoc_area_ratio[0] <= r <= cfg.assoc_area_ratio[1]):
                        continue
                pairs.append((d / gate, ti, ci))
        pairs.sort()
        used_t, used_c, result = set(), set(), []
        for _, ti, ci in pairs:
            if ti in used_t or ci in used_c:
                continue
            used_t.add(ti)
            used_c.add(ci)
            result.append((ti, ci))
        new_c = [ci for ci in range(len(cands)) if ci not in used_c]
        return result, used_t, new_c

    def _vote(self, track: Track, res: EvalResult):
        track.reason = res.reason
        track.best = res.best
        track.last_eval = self.frame_idx
        track.eval_count += 1
        if res.label == LABEL_NE:
            return
        track.votes.append(res.label)
        label, n = Counter(track.votes).most_common(1)[0]
        track.evaluating = True          # hay evidencia válida: se está confirmando
        track.reason = f"confirmando {label} ({n}/{self.cfg.confirm_votes})"
        if n >= self.cfg.confirm_votes:
            track.decision = label       # categoría bloqueada: no cambia después
            track.reason = "confirmada"
            if label == LABEL_UNKNOWN:
                self.alert_events += 1
                log.warning("ALERTA: pieza desconocida (ID%d)", track.id)
                if self.cfg.audible_alert:
                    print("\a", end="", flush=True)

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        t_start = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out = frame.copy()
        self.frame_idx += 1

        if not self.segmenter.ready:               # fase de calibración
            self.segmenter.feed_calibration(frame)
            self._draw_zone(out)
            msg = f"CALIBRANDO BANDA VACIA {int(self.segmenter.calib_progress * 100)}%"
            cv2.putText(out, msg, (20, out.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, COLOR_WARN, 2, cv2.LINE_AA)
            self.prev_gray = gray
            return out

        # --- Lucas-Kanade: actualizar tracks existentes ---
        t0 = time.perf_counter()
        if self.prev_gray is not None:
            for t in self.tracks:
                t.lk_update(self.prev_gray, gray)
                t.age += 1
                if not t.tracking_ok:
                    t.coast()
        t_lk = (time.perf_counter() - t0) * 1000

        # --- Localización de candidatos (segmentación) ---
        t0 = time.perf_counter()
        cands = self.segmenter.segment(frame)
        t_seg = (time.perf_counter() - t0) * 1000

        # --- Asociación seguimiento <-> detección (incluye recuperación) ---
        pairs, matched_t, new_c = self._associate(cands)
        t0 = time.perf_counter()
        for ti, ci in pairs:
            self.tracks[ti].apply_candidate(cands[ci], gray)
        for ti, t in enumerate(self.tracks):
            if ti not in matched_t:
                t.lost += 1
        for ci in new_c:
            t = Track(self._next_id, cands[ci], cfg)
            self._next_id += 1
            t.init_points(gray, cands[ci])
            t.matched_cand = cands[ci]
            self.tracks.append(t)
        t_lk += (time.perf_counter() - t0) * 1000

        # --- Reconocimiento ORB (solo tracks sin categoría confirmada) ---
        t0 = time.perf_counter()
        pending = [t for t in self.tracks if t.decision is None and t.matched_cand is not None
                   and t.eval_count < cfg.max_evals_per_track
                   and self.frame_idx - t.last_eval >= cfg.eval_interval]
        pending.sort(key=lambda t: t.eval_count)
        for t in pending[:cfg.max_evals_per_frame]:
            self._vote(t, self.classifier.evaluate(gray, t.matched_cand))
        t_orb = (time.perf_counter() - t0) * 1000

        # --- Contador ---
        for t in self.tracks:
            self.counter.update(t, self.frame_idx)

        # --- Limpieza de identificadores ---
        x0, y0, x1, y1 = self.roi_px
        keep = []
        for t in self.tracks:
            inside = x0 - 5 <= t.center[0] <= x1 + 5 and y0 - 5 <= t.center[1] <= y1 + 5
            if t.lost <= cfg.max_lost_frames and inside:
                keep.append(t)
        self.tracks = keep

        self.prev_gray = gray
        self._draw(out)
        total = (time.perf_counter() - t_start) * 1000
        self.totals.append(total)
        a = 0.1
        for k, v in (("seg", t_seg), ("lk", t_lk), ("orb", t_orb), ("total", total)):
            self.timing[k] = (1 - a) * self.timing[k] + a * v if self.timing[k] else v
        return out

    # ------------------------------------------------------------------
    # Dibujo de interfaz
    # ------------------------------------------------------------------
    def _draw_zone(self, img: np.ndarray):
        x0, y0, x1, y1 = self.roi_px
        cv2.rectangle(img, (x0, y0), (x1, y1), (160, 160, 160), 1)
        p = int(round(self.counter.pos))
        if self.counter.axis == 0:
            cv2.line(img, (p, y0), (p, y1), (0, 255, 255), 2)
            cv2.arrowedLine(img, (p - 20 * self.counter.sign, y0 + 15), (p + 20 * self.counter.sign, y0 + 15),
                            (0, 255, 255), 2, tipLength=0.4)
        else:
            cv2.line(img, (x0, p), (x1, p), (0, 255, 255), 2)
            cv2.arrowedLine(img, (x0 + 15, p - 20 * self.counter.sign), (x0 + 15, p + 20 * self.counter.sign),
                            (0, 255, 255), 2, tipLength=0.4)

    @staticmethod
    def _label(img, text, org, color):
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x, y = int(org[0]), int(max(th + 4, org[1]))
        cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + bl), color, -1)
        cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def _draw(self, img: np.ndarray):
        cfg = self.cfg
        self._draw_zone(img)
        unknown_visible = False
        for t in self.tracks:
            visible = t.matched_cand is not None
            if not visible and t.lost > cfg.show_lost_frames:
                continue
            if not visible and not t.tracking_ok:
                color, txt = COLOR_WARN, f"ID{t.id} NO EVALUABLE: seguimiento perdido"
            elif t.decision in self.counter.KEYS and t.decision != LABEL_UNKNOWN:
                color, txt = COLOR_OK, f"ID{t.id} PIEZA {t.decision} OK"
            elif t.decision == LABEL_UNKNOWN:
                color, txt = COLOR_ALERT, f"ID{t.id} PIEZA NO IDENTIFICADA!"
                unknown_visible = unknown_visible or visible
            elif t.evaluating and t.reason.startswith("confirmando"):
                color, txt = COLOR_WARN, f"ID{t.id} EVALUANDO ({t.reason})"
            else:
                color, txt = COLOR_WARN, f"ID{t.id} NO EVALUABLE: {t.reason}"
            if t.counted:
                txt += " [contada]"
            if visible:
                cv2.drawContours(img, [t.contour], -1, color, 2)
            else:
                cv2.polylines(img, [t.polygon.astype(np.int32).reshape(-1, 1, 2)], True, color, 1)
            if cfg.draw_homography and t.best is not None and t.best.polygon is not None:
                cv2.polylines(img, [t.best.polygon.astype(np.int32).reshape(-1, 1, 2)], True, (255, 255, 0), 1)
            if t.points is not None:
                for p in t.points.reshape(-1, 2):
                    cv2.circle(img, (int(p[0]), int(p[1])), 2, (255, 0, 255), -1)
            top = t.polygon[:, 1].min()
            self._label(img, txt, (t.polygon[:, 0].min(), top - 6), color)

        # Alerta visual parpadeante
        if unknown_visible and (self.frame_idx // 6) % 2 == 0:
            cv2.rectangle(img, (0, 0), (img.shape[1], 34), COLOR_ALERT, -1)
            cv2.putText(img, "!!! PIEZA NO IDENTIFICADA !!!", (img.shape[1] // 2 - 170, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2, cv2.LINE_AA)

        # Panel de contadores (siempre visible)
        c = self.counter.counts
        lines = [(f"Piezas A: {c['A']}", COLOR_OK), (f"Piezas B: {c['B']}", COLOR_OK),
                 (f"Piezas C: {c['C']}", COLOR_OK), (f"Desconocidas: {c[LABEL_UNKNOWN]}", COLOR_ALERT)]
        panel = img.copy()
        cv2.rectangle(panel, (6, 40), (190, 40 + 22 * len(lines) + 50), (30, 30, 30), -1)
        cv2.addWeighted(panel, 0.6, img, 0.4, 0, img)
        for i, (txt, col) in enumerate(lines):
            cv2.putText(img, txt, (14, 62 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
        fps = 1000.0 / self.timing["total"] if self.timing["total"] else 0.0
        cv2.putText(img, f"FPS: {fps:.1f}  t: {self.timing['total']:.1f} ms", (14, 62 + 22 * len(lines)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(img, f"seg {self.timing['seg']:.1f} | lk {self.timing['lk']:.1f} | orb {self.timing['orb']:.1f} ms",
                    (14, 62 + 22 * len(lines) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1, cv2.LINE_AA)


# =============================================================================
# 9. CAPTURA Y BUCLE PRINCIPAL
# =============================================================================
def open_capture(cfg: Config) -> Optional[cv2.VideoCapture]:
    src = cfg.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        log.error("No se pudo abrir la fuente de video: %s", src)
        cap.release()
        return None
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.request_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.request_height)
    return cap


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def run(cfg: Config, headless: bool, max_frames: int, save_video: Optional[str]) -> int:
    registry = ReferenceRegistry(cfg)
    if not registry.load():
        log.error("Registro de referencias incompleto. Revise la carpeta '%s'.", cfg.templates_dir)
        return 2
    log.info("Referencias cargadas: %d imágenes en %d categorías.",
             len(registry.all_refs()), len(cfg.categories))

    cap = open_capture(cfg)
    if cap is None:
        return 3
    ok, first = cap.read()
    if not ok or first is None:
        log.error("La fuente se abrió pero no entrega fotogramas.")
        cap.release()
        return 3
    first = resize_to_width(first, cfg.proc_width)
    system = InspectionSystem(cfg, registry, first.shape[:2])
    writer = None
    fails = 0
    loop_times: List[float] = []
    win = "Inspeccion visual - banda transportadora"
    try:
        if save_video:
            fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
            writer = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps_src,
                                     (first.shape[1], first.shape[0]))
        frame = first
        while True:
            t_loop = time.perf_counter()
            out = system.process(frame)
            if writer is not None:
                writer.write(out)
            if not headless:
                show = out
                if system.show_mask and system.segmenter.last_mask is not None:
                    x0, y0, x1, y1 = system.roi_px
                    show = out.copy()
                    show[y0:y1, x0:x1][system.segmenter.last_mask > 0] = (255, 128, 0)
                if cfg.display_scale != 1.0:
                    show = cv2.resize(show, None, fx=cfg.display_scale, fy=cfg.display_scale)
                cv2.imshow(win, show)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                elif key == ord("c"):
                    system.recalibrate()
                elif key == ord("r"):
                    system.counter.reset()
                elif key == ord("m"):
                    system.show_mask = not system.show_mask
                elif key == ord("s"):
                    Path(cfg.evidence_dir).mkdir(parents=True, exist_ok=True)
                    p = Path(cfg.evidence_dir) / f"captura_{datetime.now():%Y%m%d_%H%M%S}.png"
                    cv2.imwrite(str(p), out)
                    log.info("Captura guardada: %s", p)
            loop_times.append((time.perf_counter() - t_loop) * 1000)
            if max_frames and system.frame_idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if isinstance(cfg.source, str) or fails > cfg.max_read_failures:
                    log.info("Fin de la fuente de video o fallo de lectura.")
                    break
                continue
            fails = 0
            frame = resize_to_width(frame, cfg.proc_width)
    except KeyboardInterrupt:
        log.info("Interrumpido por el usuario.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()
        path = system.counter.save_events()
        if path:
            log.info("Eventos de conteo guardados en %s", path)
        report(system, loop_times)
    return 0


def report(system: InspectionSystem, loop_times: List[float]) -> None:
    """Resumen de rendimiento MEDIDO en esta ejecución (no valores supuestos)."""
    print("\n=== RESUMEN DE EJECUCIÓN ===")
    print("Contadores:", system.counter.counts)
    print("Alertas de pieza desconocida confirmadas:", system.alert_events)
    if system.totals:
        t = np.array(system.totals)
        print(f"Fotogramas procesados (tras calibración): {len(t)}")
        print(f"Tiempo de procesamiento por fotograma: media={t.mean():.1f} ms, "
              f"mediana={np.median(t):.1f} ms, p95={np.percentile(t, 95):.1f} ms")
    if loop_times:
        lt = np.array(loop_times)
        print(f"Bucle completo (incluye captura/visualización): media={lt.mean():.1f} ms "
              f"-> {1000.0 / lt.mean():.1f} FPS")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Inspección visual de piezas con ORB + Lucas-Kanade")
    ap.add_argument("--source", default="0", help="índice de cámara o ruta de video")
    ap.add_argument("--templates", default=None, help="carpeta de plantillas")
    ap.add_argument("--evidence", default=None, help="carpeta de evidencias")
    ap.add_argument("--headless", action="store_true", help="sin ventanas (pruebas)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--save-video", default=None, help="guarda el video anotado")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = Config()
    cfg.source = int(args.source) if args.source.isdigit() else args.source
    if args.templates:
        cfg.templates_dir = args.templates
    if args.evidence:
        cfg.evidence_dir = args.evidence
    return run(cfg, args.headless, args.max_frames, args.save_video)


if __name__ == "__main__":
    sys.exit(main())
