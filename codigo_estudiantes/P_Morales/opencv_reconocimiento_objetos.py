import cv2
import numpy as np

# 1. Cargar la fotografía de referencia
objeto = cv2.imread("objeto.jpg", cv2.IMREAD_GRAYSCALE)

if objeto is None:
    print("No se encontro objeto.jpg")
    exit()

objeto_gris = cv2.cvtColor(objeto, cv2.COLOR_GRAY2BGR)

# 2. Configuración avanzada de ORB optimizada para objetos lejanos y pequeños
orb = cv2.ORB_create(
    nfeatures=1500,
    scaleFactor=1.15,  # Pasos de escala más suaves entre niveles
    nlevels=10,        # Más niveles en la pirámide para detectar tamaños pequeños/lejanos
    fastThreshold=12   # Sensibilidad equilibrada para capturar detalles a distancia
)

kp_objeto, des_objeto = orb.detectAndCompute(objeto, None)
print("Puntos de referencia:", len(kp_objeto))

imagen_puntos = cv2.drawKeypoints(
    objeto_gris,
    kp_objeto,
    None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
cap = cv2.VideoCapture(0)

print("Presiona Q para salir")

# Umbral reducido para permitir la detección cuando el objeto es pequeño/lejano
MIN_COINCIDENCIAS = 10 

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray, None)

    puntos_coincidentes = []

    if des_frame is not None and des_objeto is not None and len(des_objeto) > 1:
        matches = bf.knnMatch(
            des_objeto,
            des_frame,
            k=2
        )

        # Test de Lowe
        buenas = []
        for par in matches:
            if len(par) == 2:
                m, n = par
                if m.distance < 0.75 * n.distance:
                    buenas.append(m)

        # Validación geométrica con RANSAC
        if len(buenas) > 4:
            src_pts = np.float32([kp_objeto[m.queryIdx].pt for m in buenas]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in buenas]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if mask is not None:
                matchesMask = mask.ravel().tolist()
                for i, match in enumerate(buenas):
                    if matchesMask[i]:
                        punto = kp_frame[match.trainIdx]
                        puntos_coincidentes.append(punto)
        else:
            puntos_coincidentes = []

    # Evaluar estado según el nuevo umbral para distancias
    if len(puntos_coincidentes) >= MIN_COINCIDENCIAS:
        estado = "OBJETO RECONOCIDO"
        color = (0, 255, 0)
    else:
        estado = "OBJETO NO RECONOCIDO"
        color = (0, 0, 255)

    # Visualización limpia de los puntos coincidentes sobre la pieza
    resultado = cv2.drawKeypoints(
        frame,
        puntos_coincidentes,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.putText(
        resultado,
        estado,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.putText(
        resultado,
        f"Puntos similares: {len(puntos_coincidentes)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("Reconocimiento ORB por puntos", resultado)
    cv2.imshow("Objeto de referencia", imagen_puntos)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()