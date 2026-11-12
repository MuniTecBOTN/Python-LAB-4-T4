import cv2

camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camara.isOpened():
    print("ERROR: No se pudo abrir la camara")
    exit()

print("CAMARA ABIERTA")

while True:

    correcto, frame = camara.read()

    if not correcto:
        print("ERROR: No se pudo recibir imagen")
        break

    cv2.imshow("PRUEBA DE CAMARA", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

camara.release()
cv2.destroyAllWindows()