import cv2
import numpy as np
from keras.models import load_model

# ==========================================
# CONFIGURACIÓN SIMPLE Y CONFIABLE
# ==========================================
MODELO_PATH = "keras_model.h5"
ETIQUETAS_PATH = "labels.txt"
ID_CAMARA = 0
CONFIANZA_MINIMA = 0.80  # 80% de seguridad mínima

# Cargar modelo y etiquetas
modelo = load_model(MODELO_PATH, compile=False)
with open(ETIQUETAS_PATH, "r", encoding="utf-8") as f:
    clases = [line.strip().split(" ", 1)[1] for line in f.readlines()]

cap = cv2.VideoCapture(ID_CAMARA)
print("--- SISTEMA INICIADO (Presiona 'q' para salir) ---")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Preparar imagen para la IA (224x224 píxeles normalizados)
    img_ia = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    img_array = np.asarray(img_ia, dtype=np.float32).reshape(1, 224, 224, 3)
    img_array = (img_array / 127.5) - 1

    # 2. Predecir qué objeto está frente a la cámara
    prediccion = modelo.predict(img_array, verbose=0)
    indice = np.argmax(prediccion)
    nombre_objeto = clases[indice]
    confianza = prediccion[0][indice]

    # 3. Mostrar resultado claro en pantalla (Ignora si solo ve el Fondo)
    if nombre_objeto.lower() != "fondo":
        if confianza >= CONFIANZA_MINIMA:
            # VERDE: Objeto reconocido
            color = (0, 255, 0)
            texto = f"Objeto reconocido: {nombre_objeto} ({confianza*100:.0f}%)"
            cv2.rectangle(frame, (20, 20), (620, 70), color, -1)
            cv2.putText(frame, texto, (30, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        else:
            # ROJO: Objeto no reconocido
            color = (0, 0, 255)
            texto = "Objeto no reconocido"
            cv2.rectangle(frame, (20, 20), (400, 70), color, -1)
            cv2.putText(frame, texto, (30, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.imshow("Detector Inteligente", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
