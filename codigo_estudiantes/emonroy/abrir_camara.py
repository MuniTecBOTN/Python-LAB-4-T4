import cv2
import numpy as np

# 1. Inicializar la cámara usando DirectShow
camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Presiona la tecla 'ESC' para cerrar el programa.")

while camara.isOpened():
    exito, fotograma = camara.read()
    
    if not exito:
        print("No se pudo obtener el fotograma de la cámara.")
        break
        
    # --- BLOQUE DE PROCESAMIENTO DE IMAGEN ---
    
    # A. Convertir el fotograma actual a escala de grises
    gris = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)
    
    # B. Umbral Adaptativo para detectar bordes perfectos en tiempo real
    binaria = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)
    
    # C. Encontrar las coordenadas de los contornos
    contornos, jerarquia = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # --- BLOQUE DE DIBUJO Y ETIQUETADO DIRECTO ---
    for i, c in enumerate(contornos):
        area = cv2.contourArea(c)
        
        # Filtro de ruido: ignoramos contornos muy pequeños
        if area > 150:
            # Obtener las coordenadas para la etiqueta de texto
            x, y, w, h = cv2.boundingRect(c)
            
            # CAMBIO CLAVE: Dibujamos la silueta verde encendido DIRECTAMENTE sobre el fotograma real
            # Cambié el último parámetro a '2' para que sea una línea de contorno y no rellene la pantalla
            cv2.drawContours(fotograma, [c], -1, (0, 255, 0), 2)
            
            # Estructurar el texto de la etiqueta
            etiqueta = f"Obj {i+1} | Area: {int(area)}"
            
            # Escribir las letras rojas DIRECTAMENTE sobre el fotograma real para que contrasten
            cv2.putText(fotograma, etiqueta, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
    # --- VENTANAS DE SALIDA ---
    # Ahora solo mostramos la cámara original modificada con los dibujos encima
    cv2.imshow("Deteccion de Contornos en Tiempo Real", fotograma)
    
    # Escuchar el teclado. Si se presiona ESC (código 27), rompe el bucle
    if cv2.waitKey(1) == 27:
        break

camara.release()
cv2.destroyAllWindows()
