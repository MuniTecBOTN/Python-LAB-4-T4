import cv2
import numpy as np

# 1. Inicializar la cámara (0 es la cámara integrada)
import cv2
import numpy as np

# 1. Inicializar la cámara usando DirectShow para evitar el error de Windows
camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # <-- REEMPLAZA ESTA LÍNEA (LÍNEA 5)

print("Presiona la tecla 'ESC' para cerrar el programa.")

while camara.isOpened():
    # Aquí continúa todo el resto del código del bucle while...






    # Capturar el fotograma actual de la cámara
    exito, fotograma = camara.read()
    
    if not exito:
        print("No se pudo obtener el fotograma de la cámara.")
        break
    
    # --- PROCESAMIENTO DE IMAGEN EN TIEMPO REAL ---
    
    # Convertir el fotograma a escala de grises
    gris = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)
    
    # Aplicar binarización (ajusta el 127 según la luz de tu entorno)
    _, binaria = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)
    
    # Encontrar los contornos en el fotograma binarizado
    contornos, jerarquia = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Obtener las dimensiones del fotograma para crear el fondo azul
    alto, ancho, canales = fotograma.shape
    
    # Crear el lienzo limpio de COLOR AZUL -> BGR: (255, 0, 0)
    imagen_resultado = np.ones((alto, ancho, canales), dtype="uint8") * np.array([255, 0, 0], dtype="uint8")
    
    # --- DIBUJAR LOS OBJETOS Y LAS ETIQUETAS ---
    for i, c in enumerate(contornos):
        # Calcular el área del objeto
        area = cv2.contourArea(c)
        
        # Ignorar contornos extremadamente pequeños (ruido visual de la cámara)
        if area > 100:
            # Obtener las coordenadas del cuadrado delimitador
            x, y, w, h = cv2.boundingRect(c)
            
            # DIBUJAR CUADRADO RELLENO AMARILLO -> BGR: (0, 255, 255)
            cv2.rectangle(imagen_resultado, (x, y), (x + w, y + h), (0, 255, 255), -1)
            
            # Definir la etiqueta con el número y el área
            etiqueta = f"Obj {i+1} | Area: {int(area)}"
            
            # DIBUJAR LETRAS EN NEGRO -> BGR: (0, 0, 0)
            cv2.putText(imagen_resultado, etiqueta, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
    # --- MOSTRAR LAS VENTANAS ---
    cv2.imshow("Camara Original (Entrada)", fotograma)
    cv2.imshow("Filtro Binario", binaria)
    cv2.imshow("Fondo Azul con Objetos Amarillos", imagen_resultado)
    
    # Romper el bucle si se presiona la tecla ESC (código ASCII 27)
    if cv2.waitKey(1) == 27:
        break

# Liberar los recursos de la cámara y cerrar las ventanas de OpenCV
camara.release()
cv2.destroyAllWindows()
