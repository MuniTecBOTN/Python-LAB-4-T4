
import cv2
import numpy as np
import os
from ultralytics import YOLO


# ==========================================
# CONFIGURACION DEL SISTEMA
# ==========================================

ID_CAMARA = 1

MODELO_YOLO = "yolov8n.pt"

CONFIANZA_MINIMA = 0.50

CLASE_PERSONA = 0

TAMANO_IMAGEN = 640

COLOR_VERDE = (0, 255, 0)
COLOR_AZUL = (255, 0, 0)
COLOR_ROJO = (0, 0, 255)
COLOR_BLANCO = (255, 255, 255)

GROSOR_RECTANGULO = 2


# ==========================================
# FASE 1: CARGAR MODELO YOLO
# ==========================================

def cargar_descriptores_plantillas():

    print("--- FASE 1: CARGANDO MODELO YOLO ---")

    try:

        modelo = YOLO(MODELO_YOLO)

        print("Modelo YOLO cargado correctamente.")

        return modelo

    except Exception as error:

        print(f"Error al cargar YOLO: {error}")

        return None


# ==========================================
# FASE 2: PREPROCESAMIENTO
# ==========================================

def preprocesamiento_antidesenfoque_banda(imagen):

    if imagen is None:

        return None

    # Mantener la imagen original en color

    return imagen


# ==========================================
# FASE 3: DETECCION Y SEGUIMIENTO
# ==========================================

def clasificar_region(region, plantillas):

    resultados = plantillas.track(
        source=region,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONFIANZA_MINIMA,
        imgsz=TAMANO_IMAGEN,
        classes=[CLASE_PERSONA],
        verbose=False
    )

    personas_detectadas = []

    for resultado in resultados:

        cajas = resultado.boxes

        if cajas is None or len(cajas) == 0:
            continue

        # Obtener identificadores de seguimiento

        identificadores = cajas.id

        coordenadas = cajas.xyxy.cpu().numpy()

        confianzas = cajas.conf.cpu().numpy()

        if identificadores is not None:

            ids = identificadores.int().cpu().tolist()

        else:

            ids = [None] * len(coordenadas)

        for caja, confianza, persona_id in zip(
            coordenadas,
            confianzas,
            ids
        ):

            x1, y1, x2, y2 = map(int, caja)

            personas_detectadas.append({

                "caja": (x1, y1, x2, y2),

                "confianza": float(confianza),

                "id": persona_id

            })

    return personas_detectadas


# ==========================================
# FASE 4: SEGUIMIENTO Y CONTEO
# ==========================================

def actualizar_conteo(personas, ids_registrados):

    total_personas = len(personas)

    for persona in personas:

        persona_id = persona["id"]

        if persona_id is not None:

            ids_registrados.add(persona_id)

    total_unicas = len(ids_registrados)

    return total_personas, total_unicas


# ==========================================
# FASE 5: DIBUJAR DETECCIONES
# ==========================================

def dibujar_detecciones(frame, personas):

    for persona in personas:

        x1, y1, x2, y2 = persona["caja"]

        confianza = persona["confianza"]

        persona_id = persona["id"]

        # Dibujar rectangulo

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            COLOR_VERDE,
            GROSOR_RECTANGULO
        )

        # Mostrar ID de seguimiento

        if persona_id is not None:

            etiqueta = f"Persona ID: {persona_id}"

        else:

            etiqueta = "Persona"

        # Mostrar confianza

        etiqueta += f" | {confianza * 100:.1f}%"

        cv2.putText(
            frame,
            etiqueta,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_VERDE,
            2
        )

    return frame


# ==========================================
# FASE 6: MOSTRAR ESTADISTICAS
# ==========================================

def mostrar_informacion(
    frame,
    total_personas,
    total_unicas
):

    # Panel de informacion

    cv2.rectangle(
        frame,
        (10, 10),
        (430, 125),
        (30, 30, 30),
        -1
    )

    # Personas visibles

    cv2.putText(
        frame,
        f"Personas actuales: {total_personas}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        COLOR_BLANCO,
        2
    )

    # Personas unicas registradas

    cv2.putText(
        frame,
        f"Total de personas: {total_unicas}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        COLOR_VERDE,
        2
    )

    # Estado

    if total_personas > 0:

        estado = "DETECCION ACTIVA"

        color_estado = COLOR_VERDE

    else:

        estado = "SIN PERSONAS"

        color_estado = COLOR_ROJO

    cv2.putText(
        frame,
        estado,
        (20, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color_estado,
        2
    )

    return frame


# ==========================================
# FASE 7: PROCESAMIENTO EN TIEMPO REAL
# ==========================================

def procesar_tiempo_real():

    plantillas = cargar_descriptores_plantillas()

    if plantillas is None:

        print("Error: No se pudo cargar YOLO.")

        return

    cap = cv2.VideoCapture(ID_CAMARA)

    if not cap.isOpened():

        print("Error: No se puede acceder a la camara.")

        return

    cap.set(cv2.CAP_PROP_FPS, 30)


    ids_registrados = set()

    print("\n--- SISTEMA YOLO + BYTETRACK INICIADO ---")

    print("Presione Q para finalizar.")

    try:

        while True:

            ret, frame = cap.read()

            if not ret:

                print("Error al obtener imagen.")

                break

            frame_visualizacion = frame.copy()



            imagen_actual = preprocesamiento_antidesenfoque_banda(
                frame
            )


            personas_detectadas = clasificar_region(
                imagen_actual,
                plantillas
            )


            total_personas, total_unicas = actualizar_conteo(
                personas_detectadas,
                ids_registrados
            )


            frame_visualizacion = dibujar_detecciones(
                frame_visualizacion,
                personas_detectadas
            )


            frame_visualizacion = mostrar_informacion(
                frame_visualizacion,
                total_personas,
                total_unicas
            )


            cv2.imshow(
                "Sistema Industrial YOLOv8 + ByteTrack",
                frame_visualizacion
            )

            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord('q'):

                break

    finally:

        cap.release()

        cv2.destroyAllWindows()

        print("\n--- RESUMEN DEL SISTEMA ---")

        print(f"Personas unicas registradas: {len(ids_registrados)}")

        print("Sistema finalizado correctamente.")



if __name__ == "__main__":

    procesar_tiempo_real()
