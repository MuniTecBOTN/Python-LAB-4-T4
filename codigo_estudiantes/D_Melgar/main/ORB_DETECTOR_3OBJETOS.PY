
import cv2
import numpy as np
import time


query_img = cv2.imread("main/plantillas/objeto_1_CA.jpg", cv2.IMREAD_GRAYSCALE)

query_img2 = cv2.imread("main/plantillas/objeto_2_CB.jpg", cv2.IMREAD_GRAYSCALE)

query_img3 = cv2.imread("main/plantillas/objeto_3_CC.jpg", cv2.IMREAD_GRAYSCALE)

if query_img is None:
    print("Error: No se pudo cargar 1ss_.jpg")
    raise SystemExit

if query_img2 is None:
    print("Error: No se pudo cargar 2ss_.jpg")
    raise SystemExit

if query_img3 is None:
    print("Error: No se pudo cargar 3ss_.jpg")
    raise SystemExit

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

query_img = clahe.apply(query_img)

query_img2 = clahe.apply(query_img2)

query_img3 = clahe.apply(query_img3)

orb = cv2.ORB_create(
    nfeatures=1500,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=15,
    patchSize=31,
    fastThreshold=10
)

kp1, des1 = orb.detectAndCompute(query_img, None)
kp2, des2 = orb.detectAndCompute(query_img2, None)
kp3, des3 = orb.detectAndCompute(query_img3, None)

if des1 is None or len(kp1) < 15:
    print("No hay suficientes puntos en la imagen de referencia")
    raise SystemExit
if des2 is None or len(kp2) < 15:
    print("No hay suficientes puntos en la imagen de referencia 2")
    raise SystemExit

if des3 is None or len(kp3) < 15:
    print("No hay suficientes puntos en la imagen de referencia 3")
    raise SystemExit

reference_points = cv2.drawKeypoints(
    query_img,
    kp1,
    None,
    color=(255, 0, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

reference_points2 = cv2.drawKeypoints(
    query_img2,
    kp2,
    None,
    color=(255, 0, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

reference_points3 = cv2.drawKeypoints(
    query_img3,
    kp3,
    None,
    color=(255, 0, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

FLANN_INDEX_LSH = 6

index_params = dict(
    algorithm=FLANN_INDEX_LSH,
    table_number=12,
    key_size=20,
    multi_probe_level=2
)

search_params = dict(checks=100)

flann = cv2.FlannBasedMatcher(
    index_params,
    search_params
)


cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)


MIN_MATCH_COUNT = 20
MIN_INLIER_RATIO = 0.55

SMOOTHING = 0.30
MAX_LOST_FRAMES = 3

smooth_corners = None
lost_frames = 0

previous_time = time.perf_counter()


def validar_geometria(projected, frame_shape, src_inliers,
                      dst_inliers, reference_shape):

    frame_h, frame_w = frame_shape[:2]
    ref_h, ref_w = reference_shape[:2]

    if projected is None:
        return False

    if not np.isfinite(projected).all():
        return False

    polygon = projected.reshape(4, 2).astype(np.float32)

    if not cv2.isContourConvex(polygon.reshape(-1, 1, 2)):
        return False

    area = abs(cv2.contourArea(polygon))

    if area < 300:
        return False

    if area > frame_w * frame_h * 0.85:
        return False

    if (
        np.any(polygon[:, 0] < -frame_w * 0.20)
        or np.any(polygon[:, 0] > frame_w * 1.20)
        or np.any(polygon[:, 1] < -frame_h * 0.20)
        or np.any(polygon[:, 1] > frame_h * 1.20)
    ):
        return False

    sides = [
        np.linalg.norm(polygon[i] - polygon[(i + 1) % 4])
        for i in range(4)
    ]

    if min(sides) < 20:
        return False

    src_hull = cv2.convexHull(
        src_inliers.reshape(-1, 1, 2)
    )

    src_area = cv2.contourArea(src_hull)
    reference_area = ref_w * ref_h

    if src_area / reference_area < 0.02:
        return False

    dst_hull = cv2.convexHull(
        dst_inliers.reshape(-1, 1, 2)
    )

    dst_area = cv2.contourArea(dst_hull)

    if dst_area / area < 0.05:
        return False

    return True



while True:

    ret, frame = cap.read()

    if not ret:
        print("No se pudo recibir imagen de la cámara")
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Mejorar contraste también en la cámara
    gray = clahe.apply(gray)

    kp2, des2 = orb.detectAndCompute(gray, None)

    good_matches = []
    valid_matches = []

    detected = False
    confidence = 0
    matrix = None


    if des2 is not None and len(des2) >= 2:

        matches = flann.knnMatch(
            des1,
            des2,
            k=2
        )

        for pair in matches:

            if len(pair) == 2:

                m, n = pair

                # Filtro más estricto
                if m.distance < 0.72 * n.distance:
                    good_matches.append(m)

    if len(good_matches) >= MIN_MATCH_COUNT:

        src_pts = np.float32([
            kp1[m.queryIdx].pt
            for m in good_matches
        ]).reshape(-1, 1, 2)

        dst_pts = np.float32([
            kp2[m.trainIdx].pt
            for m in good_matches
        ]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            4.0
        )

        if matrix is not None and mask is not None:

            inliers = mask.ravel().astype(bool)

            valid_matches = [
                m for m, valid in zip(good_matches, inliers)
                if valid
            ]

            confidence = (
                len(valid_matches) / len(good_matches)
            ) * 100

            if (
                len(valid_matches) >= MIN_MATCH_COUNT
                and confidence >= MIN_INLIER_RATIO * 100
            ):

                h, w = query_img.shape

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

                src_inliers = src_pts[inliers].reshape(-1, 2)
                dst_inliers = dst_pts[inliers].reshape(-1, 2)

                geometry_ok = validar_geometria(
                    projected,
                    frame.shape,
                    src_inliers,
                    dst_inliers,
                    query_img.shape
                )

                if geometry_ok:

                    detected = True

                    new_corners = projected.reshape(4, 2)

                    if smooth_corners is None:

                        smooth_corners = new_corners.copy()

                    else:

                        smooth_corners = (
                            SMOOTHING * new_corners
                            + (1 - SMOOTHING) * smooth_corners
                        )

                    lost_frames = 0


    if not detected:
        lost_frames += 1


    output = cv2.drawKeypoints(
        frame,
        kp2,
        None,
        color=(255, 0, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )


    if detected:

        matched_kp2 = [
            kp2[m.trainIdx]
            for m in valid_matches
        ]

        output = cv2.drawKeypoints(
            output,
            matched_kp2,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        polygon = np.int32(smooth_corners)

        cv2.polylines(
            output,
            [polygon],
            True,
            (0, 255, 0),
            3,
            cv2.LINE_AA
        )

        x, y, box_w, box_h = cv2.boundingRect(polygon)

        center_x = x + box_w // 2
        center_y = y + box_h // 2

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
            (420, 230),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            output,
            "OBJETO ENCONTRADO",
            (25, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            output,
            f"Confianza: {confidence:.1f}%",
            (25, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.putText(
            output,
            f"Proporcion inliers: {confidence:.1f}%",
            (25, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.putText(
            output,
            f"Coincidencias: {len(good_matches)}",
            (25, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.putText(
            output,
            f"Puntos validos: {len(valid_matches)}",
            (25, 138),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1
        )

        cv2.putText(
            output,
            f"Centro: X={center_x} Y={center_y}",
            (25, 168),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.putText(
            output,
            f"Tamano: {box_w} x {box_h}",
            (25, 198),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

    else:

        cv2.rectangle(
            output,
            (10, 10),
            (370, 85),
            (25, 25, 25),
            -1
        )

        if lost_frames <= MAX_LOST_FRAMES and smooth_corners is not None:
            status = "RECUPERANDO OBJETO"
            color = (0, 255, 255)
        else:
            status = "BUSCANDO OBJETO..."
            color = (0, 0, 255)

        cv2.putText(
            output,
            status,
            (25, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

        cv2.putText(
            output,
            f"Coincidencias: {len(good_matches)}",
            (25, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )


    if not detected and smooth_corners is not None:

        if lost_frames <= MAX_LOST_FRAMES:

            cv2.polylines(
                output,
                [np.int32(smooth_corners)],
                True,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        else:

            smooth_corners = None

    current_time = time.perf_counter()

    elapsed = current_time - previous_time

    fps = 1 / elapsed if elapsed > 0 else 0

    previous_time = current_time

    cv2.putText(
        output,
        f"FPS: {fps:.1f}",
        (output.shape[1] - 130, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "DETECCION MEJORADA",
        output
    )

    cv2.imshow(
        "IMAGEN DE REFERENCIA",
        reference_points
    )
    
    cv2.imshow(
        "IMAGEN DE REFERENCIA 2",
        reference_points2
    )

    cv2.imshow(
        "IMAGEN DE REFERENCIA 3",
        reference_points3
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
