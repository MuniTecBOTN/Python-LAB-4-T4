import cv2
import numpy as np


ID_CAMARA = 0
UMBRAL_PERSONA = 0.35


def detectar_personas(frame, detector):
    """Detecta personas completas sin usar fotografías ni plantillas."""
    cajas, pesos = detector.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    candidatas = []
    for caja, peso in zip(cajas, pesos):
        confianza = float(peso[0] if hasattr(peso, "__len__") else peso)
        if confianza >= UMBRAL_PERSONA:
            candidatas.append([int(v) for v in caja])

    if not candidatas:
        return []

    indices = cv2.dnn.NMSBoxes(candidatas, [1.0] * len(candidatas), 0.5, 0.4)
    if len(indices) == 0:
        return []
    return [candidatas[int(i)] for i in indices]

def procesar_tiempo_real():
    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cap = cv2.VideoCapture(ID_CAMARA)
    if not cap.isOpened():
        print(f"Error: No se puede acceder a la cámara.")
        return

    cap.set(cv2.CAP_PROP_FPS, 30)

    COLOR_VERDE = (0, 255, 0)
    
    print("\n--- DETECTOR DE PERSONAS EN TIEMPO REAL INICIADO ---")
    print("Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_visualizacion = frame.copy()
        personas = detectar_personas(frame, detector)

        for bx, by, bw, bh in personas:
            cv2.rectangle(
                frame_visualizacion,
                (bx, by),
                (bx + bw, by + bh),
                COLOR_VERDE,
                2
            )
            cv2.putText(
                frame_visualizacion,
                "Persona",
                (bx, max(20, by - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_VERDE,
                2
            )

        cv2.putText(
            frame_visualizacion,
            f"Personas detectadas: {len(personas)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        cv2.imshow("Detector de Personas - Tiempo Real", frame_visualizacion)
        
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    procesar_tiempo_real()