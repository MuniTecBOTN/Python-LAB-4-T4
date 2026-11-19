
import cv2

print("Version OpenCV:", cv2.__version__)

for backend_nombre, backend in [
    ("MSMF", cv2.CAP_MSMF),
    ("DSHOW", cv2.CAP_DSHOW)
]:
    print(f"\nProbando backend: {backend_nombre}")

    camara = cv2.VideoCapture(0, backend)

    if not camara.isOpened():
        print("No se pudo abrir la camara.")
        camara.release()
        continue

    try:
        correcto, frame = camara.read()

        if correcto and frame is not None and frame.size > 0:
            print("Camara funcionando correctamente.")
            print("Resolucion:", frame.shape)

            cv2.imshow("PRUEBA DE CAMARA", frame)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
        else:
            print("La camara se abre, pero no entrega imagen.")

    except cv2.error as error:
        print("Error de OpenCV al leer:", error)

    finally:
        camara.release()

cv2.destroyAllWindows()
print("\nPrueba terminada.")
