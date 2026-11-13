
import cv2
import numpy as np
from ultralytics import YOLO

# =====================================
# 1. CONFIGURACION DE OBJETOS ORB
# =====================================

objetos = {
    "Objeto principal": "referencias/qs.jpg",
    "Botella": "referencias/botella.jpg",
    "Libro": "referencias/libro.jpg",
    "Celular": "referencias/celular.jpg"
}

orb = cv2.ORB_create(nfeatures=2000)

referencias = {}

for nombre, ruta in objetos.items():

    imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

    if imagen is None:
        print(f"No se pudo cargar: {ruta}")
        continue

    kp, des = orb.detectAndCompute(imagen, None)

    if des is not None and len(kp) >= 10:
        referencias[nombre] = {
            "imagen": imagen,
            "keypoints": kp,
            "descriptores": des
        }

print("Objetos ORB cargados:", list(referencias.keys()))

# =====================================
# 2. CONFIGURAR FLANN
# =====================================

FLANN_INDEX_LSH = 6

index_params = dict(
    algorithm=FLANN_INDEX_LSH,
    table_number=6,
    key_size=12,
    multi_probe_level=1
)

search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(
    index_params,
    search_params
)

# =====================================
# 3. CARGAR YOLO
# =====================================

modelo = YOLO("yolo11n.pt")

# =====================================
# 4. ABRIR CAMARA
# =====================================

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("No se pudo abrir la camara")
    exit()

MIN_MATCHES = 15

# =====================================
# 5. PROCESAMIENTO
# =====================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gris = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    kp2, des2 = orb.detectAndCompute(gris, None)

    # ---------------------------------
    # DETECCION ORB DE VARIOS OBJETOS
    # ---------------------------------

    if des2 is not None and len(des2) >= 2:

        for nombre, datos in referencias.items():

            des1 = datos["descriptores"]
            kp1 = datos["keypoints"]
            imagen_ref = datos["imagen"]

            matches = flann.knnMatch(
                des1,
                des2,
                k=2
            )

            buenos = []

            for par in matches:

                if len(par) == 2:

                    m, n = par

                    if m.distance < 0.75 * n.distance:
                        buenos.append(m)

            if len(buenos) < MIN_MATCHES:
                continue

            src = np.float32([
                kp1[m.queryIdx].pt for m in buenos
            ]).reshape(-1, 1, 2)

            dst = np.float32([
                kp2[m.trainIdx].pt for m in buenos
            ]).reshape(-1, 1, 2)

            matriz, mascara = cv2.findHomography(
                src,
                dst,
                cv2.RANSAC,
                3.0
            )

            if matriz is None or mascara is None:
                continue

            inliers = mascara.ravel().astype(bool)

            validos = [
                m for m, valido in zip(buenos, inliers)
                if valido
            ]

            calidad = len(validos) / len(buenos)

            if len(validos) < MIN_MATCHES or calidad < 0.55:
                continue

            h, w = imagen_ref.shape

            esquinas = np.float32([
                [0, 0],
                [0, h - 1],
                [w - 1, h - 1],
                [w - 1, 0]
            ]).reshape(-1, 1, 2)

            proyectadas = cv2.perspectiveTransform(
                esquinas,
                matriz
            )

            area = abs(cv2.contourArea(proyectadas))

            if (
                not np.isfinite(proyectadas).all()
                or not cv2.isContourConvex(
                    proyectadas.astype(np.float32)
                )
                or area < 100
                or area > frame.shape[0] * frame.shape[1] * 0.90
            ):
                continue

            poligono = np.int32(proyectadas)

            cv2.polylines(
                frame,
                [poligono],
                True,
                (0, 255, 0),
                3
            )

            porcentaje = calidad * 100

            x, y = poligono[0][0]

            cv2.putText(
                frame,
                f"{nombre} - {porcentaje:.1f}%",
                (int(x), max(25, int(y) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # ---------------------------------
    # DETECCION YOLO
    # ---------------------------------

    resultados = modelo(
        frame,
        conf=0.50,
        verbose=False
    )

    for resultado in resultados:

        for caja in resultado.boxes:

            x1, y1, x2, y2 = map(
                int,
                caja.xyxy[0].tolist()
            )

            confianza = float(caja.conf[0])

            clase_id = int(caja.cls[0])

            nombre_clase = modelo.names[clase_id]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            etiqueta = (
                f"{nombre_clase} {confianza * 100:.1f}%"
            )

            cv2.putText(
                frame,
                etiqueta,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

    # ---------------------------------
    # MOSTRAR RESULTADO
    # ---------------------------------

    cv2.imshow(
        "SISTEMA DE DETECCION MULTIPLE",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
