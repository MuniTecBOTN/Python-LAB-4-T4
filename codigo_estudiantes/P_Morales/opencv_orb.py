import cv2

cap = cv2.VideoCapture(0)

orb = cv2.ORB_create(nfeatures=500)

print("Presiona la tecla 'q' para salir.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    keypoints, descriptors = orb.detectAndCompute(gray, None)

    img_keypoints = cv2.drawKeypoints(
        frame, keypoints, None, color=(0, 255, 0), 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imshow('Deteccion ORB en Tiempo Real', img_keypoints)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
