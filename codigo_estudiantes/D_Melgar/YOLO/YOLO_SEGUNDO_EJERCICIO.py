import cv2
from ultralytics import YOLO

modelo = YOLO("yolo11n.pt")

camara = cv2.VideoCapture(1)

while True:
    ret, frame = camara.read()

    if not ret:
        print("No se pudo capturar el fotograma de la cámara.")
        break

    resultados = modelo(frame, verbose=False)
    cantidad_personas = 0

    for caja in resultados[0].boxes:
        class_id = int(caja.cls[0])
        confianza = float(caja.conf[0])
        nombre = modelo.names[class_id]

        if nombre == "person":
            cantidad_personas += 1

        print(f"Clase: {class_id} ({nombre}) | Confianza: {confianza:.2f}")

    print(f"Total de personas: {cantidad_personas}")
    print("-------------------------------")

    frame_detectado = resultados[0].plot()

    cv2.putText(
        frame_detectado,
        f"Personas: {cantidad_personas}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Detecciones", frame_detectado)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()