import cv2
import numpy as np
import time


# =====================================================
# 1. CONFIGURACIÓN GENERAL
# =====================================================

REFERENCE_PATH = "imagenes_deteccion/1ss.jpg"

CAMERA_INDEX = 1
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ORB
ORB_FEATURES = 1500
ORB_SCALE_FACTOR = 1.2
ORB_LEVELS = 8
ORB_EDGE_THRESHOLD = 31
ORB_PATCH_SIZE = 31
ORB_FAST_THRESHOLD = 15

# Matcher
RATIO_THRESHOLD = 0.72
MIN_MATCH_COUNT = 20

# Homografía
RANSAC_THRESHOLD = 4.0
MIN_INLIER_RATIO = 0.55

# Suavizado temporal
SMOOTHING = 0.30
MAX_LOST_FRAMES = 5

# Procesamiento
USE_CLAHE = False
DRAW_ALL_KEYPOINTS = False

WINDOW_NAME = "DETECCION ORB OPTIMIZADA"


# =====================================================
# 2. PREPROCESAMIENTO
# =====================================================

def preprocess_image(image, clahe=None):

    if image is None:
        raise ValueError("La imagen recibida es None")

    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image

    if clahe is not None:
        gray = clahe.apply(gray)

    return gray


# =====================================================
# 3. INICIALIZAR ORB
# =====================================================

def create_orb():

    return cv2.ORB_create(
        nfeatures=ORB_FEATURES,
        scaleFactor=ORB_SCALE_FACTOR,
        nlevels=ORB_LEVELS,
        edgeThreshold=ORB_EDGE_THRESHOLD,
        patchSize=ORB_PATCH_SIZE,
        fastThreshold=ORB_FAST_THRESHOLD,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE
    )


# =====================================================
# 4. EXTRAER CARACTERÍSTICAS DE REFERENCIA
# =====================================================

def load_reference(path, orb, clahe=None):

    image = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            f"No se pudo cargar: {path}"
        )

    if clahe is not None:
        image = clahe.apply(image)

    keypoints, descriptors = orb.detectAndCompute(
        image,
        None
    )

    if descriptors is None or len(keypoints) < MIN_MATCH_COUNT:
        raise ValueError(
            "La imagen de referencia no contiene "
            "suficientes características"
        )

    return image, keypoints, descriptors


# =====================================================
# 5. CONFIGURAR MATCHER
# =====================================================

def create_matcher():

    # ORB utiliza descriptores binarios.
    # NORM_HAMMING es la distancia apropiada.

    return cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False
    )


# =====================================================
# 6. FILTRAR COINCIDENCIAS
# =====================================================

def find_good_matches(
    matcher,
    descriptors_reference,
    descriptors_frame
):

    if descriptors_frame is None:
        return []

    if len(descriptors_frame) < 2:
        return []

    raw_matches = matcher.knnMatch(
        descriptors_reference,
        descriptors_frame,
        k=2
    )

    candidates = []

    # Lowe's ratio test

    for pair in raw_matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < RATIO_THRESHOLD * n.distance:
            candidates.append(m)

    # Evitar que varios descriptores de referencia
    # utilicen el mismo descriptor del fotograma.

    best_matches = {}

    for match in candidates:

        train_id = match.trainIdx

        previous = best_matches.get(train_id)

        if previous is None or match.distance < previous.distance:
            best_matches[train_id] = match

    good_matches = list(best_matches.values())

    return good_matches


# =====================================================
# 7. VALIDAR GEOMETRÍA
# =====================================================

def validate_geometry(
    projected,
    frame_shape,
    src_inliers,
    dst_inliers,
    reference_shape
):

    frame_h, frame_w = frame_shape[:2]
    ref_h, ref_w = reference_shape[:2]

    if projected is None:
        return False

    polygon = np.asarray(
        projected,
        dtype=np.float32
    ).reshape(4, 2)

    if not np.isfinite(polygon).all():
        return False

    # Convexidad

    contour = polygon.reshape(-1, 1, 2)

    if not cv2.isContourConvex(contour):
        return False

    # Área del objeto proyectado

    area = abs(cv2.contourArea(contour))

    if area < 300:
        return False

    if area > frame_w * frame_h * 0.85:
        return False

    # Límites de la imagen

    if (
        np.any(polygon[:, 0] < -frame_w * 0.20)
        or np.any(polygon[:, 0] > frame_w * 1.20)
        or np.any(polygon[:, 1] < -frame_h * 0.20)
        or np.any(polygon[:, 1] > frame_h * 1.20)
    ):
        return False

    # Longitud mínima de los lados

    sides = np.linalg.norm(
        polygon - np.roll(polygon, -1, axis=0),
        axis=1
    )

    if np.min(sides) < 20:
        return False

    # Distribución espacial de inliers

    if len(src_inliers) < 4 or len(dst_inliers) < 4:
        return False

    src_hull = cv2.convexHull(
        src_inliers.astype(np.float32).reshape(-1, 1, 2)
    )

    dst_hull = cv2.convexHull(
        dst_inliers.astype(np.float32).reshape(-1, 1, 2)
    )

    src_area = cv2.contourArea(src_hull)
    dst_area = cv2.contourArea(dst_hull)

    reference_area = ref_w * ref_h

    if src_area / reference_area < 0.02:
        return False

    if dst_area / area < 0.05:
        return False

    return True


# =====================================================
# 8. CALCULAR HOMOGRAFÍA
# =====================================================

def calculate_homography(
    good_matches,
    kp_reference,
    kp_frame,
    reference_shape,
    frame_shape
):

    if len(good_matches) < MIN_MATCH_COUNT:
        return None, [], 0.0, None

    # Construcción de matrices NumPy

    src_pts = np.float32([
        kp_reference[m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    dst_pts = np.float32([
        kp_frame[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    # RANSAC

    matrix, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        RANSAC_THRESHOLD,
        maxIters=2000,
        confidence=0.995
    )

    if matrix is None or mask is None:
        return None, [], 0.0, None

    if not np.isfinite(matrix).all():
        return None, [], 0.0, None

    if np.linalg.matrix_rank(matrix) < 3:
        return None, [], 0.0, None

    inliers = mask.ravel().astype(bool)

    valid_matches = [
        match
        for match, valid in zip(good_matches, inliers)
        if valid
    ]

    confidence = (
        len(valid_matches) / len(good_matches)
    )

    if (
        len(valid_matches) < MIN_MATCH_COUNT
        or confidence < MIN_INLIER_RATIO
    ):
        return None, [], confidence, None

    # Coordenadas de los puntos válidos

    src_inliers = src_pts[inliers].reshape(-1, 2)
    dst_inliers = dst_pts[inliers].reshape(-1, 2)

    # Esquinas de la imagen de referencia

    ref_h, ref_w = reference_shape[:2]

    corners = np.float32([
        [0, 0],
        [ref_w - 1, 0],
        [ref_w - 1, ref_h - 1],
        [0, ref_h - 1]
    ]).reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(
        corners,
        matrix
    )

    # Validación geométrica

    if not validate_geometry(
        projected,
        frame_shape,
        src_inliers,
        dst_inliers,
        reference_shape
    ):
        return None, [], confidence, None

    return (
        matrix,
        valid_matches,
        confidence,
        projected.reshape(4, 2)
    )


# =====================================================
# 9. SUAVIZAR CONTORNO
# =====================================================

def smooth_polygon(previous, current):

    if previous is None:
        return current.copy()

    return (
        SMOOTHING * current
        + (1.0 - SMOOTHING) * previous
    )


# =====================================================
# 10. VISUALIZACIÓN
# =====================================================

def draw_information(
    frame,
    detected,
    corners,
    confidence,
    good_count,
    valid_count,
    lost_frames,
    fps,
    keypoints=None
):

    output = frame.copy()

    # Dibujar todos los puntos solamente si se solicita

    if DRAW_ALL_KEYPOINTS and keypoints is not None:

        output = cv2.drawKeypoints(
            output,
            keypoints,
            None,
            color=(255, 0, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        )

    if detected and corners is not None:

        polygon = np.int32(corners)

        cv2.polylines(
            output,
            [polygon],
            True,
            (0, 255, 0),
            3,
            cv2.LINE_AA
        )

        x, y, w, h = cv2.boundingRect(polygon)

        center_x = x + w // 2
        center_y = y + h // 2

        cv2.circle(
            output,
            (center_x, center_y),
            6,
            (0, 255, 255),
            -1
        )

        cv2.rectangle(
            output,
            (10, 10),
            (390, 180),
            (25, 25, 25),
            -1
        )

        information = [
            ("OBJETO ENCONTRADO", (0, 255, 0)),
            (f"Inliers: {confidence * 100:.1f}%", (255, 255, 255)),
            (f"Coincidencias: {good_count}", (255, 255, 255)),
            (f"Puntos validos: {valid_count}", (0, 255, 0)),
            (f"Centro: X={center_x} Y={center_y}", (255, 255, 255)),
            (f"Tamano: {w} x {h}", (255, 255, 255))
        ]

        for i, (message, color) in enumerate(information):

            cv2.putText(
                output,
                message,
                (25, 38 + i * 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
                cv2.LINE_AA
            )

    else:

        status = (
            "RECUPERANDO OBJETO"
            if lost_frames <= MAX_LOST_FRAMES
            else "BUSCANDO OBJETO..."
        )

        cv2.rectangle(
            output,
            (10, 10),
            (350, 80),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            output,
            status,
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"Coincidencias: {good_count}",
            (25, 67),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        if corners is not None and lost_frames <= MAX_LOST_FRAMES:

            cv2.polylines(
                output,
                [np.int32(corners)],
                True,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

    cv2.putText(
        output,
        f"FPS: {fps:.1f}",
        (output.shape[1] - 130, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    return output


# =====================================================
# 11. PROGRAMA PRINCIPAL
# =====================================================

def main():

    cap = None

    try:

        # Inicializar CLAHE opcional

        clahe = None

        if USE_CLAHE:

            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )

        # Crear ORB y cargar referencia

        orb = create_orb()

        reference, kp_reference, des_reference = load_reference(
            REFERENCE_PATH,
            orb,
            clahe
        )

        # Crear matcher

        matcher = create_matcher()

        # Abrir cámara

        cap = cv2.VideoCapture(CAMERA_INDEX)

        if not cap.isOpened():
            raise RuntimeError(
                "No se pudo abrir la cámara"
            )

        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            CAMERA_WIDTH
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            CAMERA_HEIGHT
        )

        # Estado del detector

        smooth_corners = None
        lost_frames = 0

        previous_time = time.perf_counter()
        fps_smooth = 0.0

        print("Detector ORB iniciado.")
        print("Presiona Q para salir.")

        # =========================================
        # BUCLE DE PROCESAMIENTO
        # =========================================

        while True:

            ret, frame = cap.read()

            if not ret:
                print("Error al capturar fotograma.")
                break

            # Preprocesamiento

            gray = preprocess_image(
                frame,
                clahe
            )

            # Extraer características

            kp_frame, des_frame = orb.detectAndCompute(
                gray,
                None
            )

            detected = False
            confidence = 0.0
            good_matches = []
            valid_matches = []

            new_corners = None

            # =====================================
            # COINCIDENCIAS
            # =====================================

            if des_frame is not None:

                good_matches = find_good_matches(
                    matcher,
                    des_reference,
                    des_frame
                )

            # =====================================
            # HOMOGRAFÍA
            # =====================================

            if len(good_matches) >= MIN_MATCH_COUNT:

                (
                    matrix,
                    valid_matches,
                    confidence,
                    projected
                ) = calculate_homography(
                    good_matches,
                    kp_reference,
                    kp_frame,
                    reference.shape,
                    frame.shape
                )

                if projected is not None:

                    detected = True
                    new_corners = projected

            # =====================================
            # SUAVIZADO TEMPORAL
            # =====================================

            if detected:

                smooth_corners = smooth_polygon(
                    smooth_corners,
                    new_corners
                )

                lost_frames = 0

            else:

                lost_frames += 1

                if lost_frames > MAX_LOST_FRAMES:
                    smooth_corners = None

            # =====================================
            # FPS
            # =====================================

            current_time = time.perf_counter()

            elapsed = current_time - previous_time

            previous_time = current_time

            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            # Suavizar lectura de FPS

            if fps_smooth == 0:
                fps_smooth = fps
            else:
                fps_smooth = (
                    0.2 * fps
                    + 0.8 * fps_smooth
                )

            # =====================================
            # VISUALIZACIÓN
            # =====================================

            output = draw_information(
                frame,
                detected,
                smooth_corners,
                confidence,
                len(good_matches),
                len(valid_matches),
                lost_frames,
                fps_smooth,
                kp_frame
            )

            cv2.imshow(
                WINDOW_NAME,
                output
            )

            # Salir con Q o ESC

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    except (FileNotFoundError, ValueError, RuntimeError) as error:

        print(f"Error: {error}")

    except cv2.error as error:

        print(f"Error de OpenCV: {error}")

    finally:

        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()

        print("Recursos liberados.")


# =====================================================
# 12. EJECUCIÓN
# =====================================================

if __name__ == "__main__":
    main()