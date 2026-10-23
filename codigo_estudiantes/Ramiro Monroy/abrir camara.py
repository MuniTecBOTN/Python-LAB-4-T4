import cv2
# Activar la camara web
camara = cv2.VideoCapture(0)
while camara.isOpened():
    # capturar un fotograma
    exito, fotograma = camara.read()
    if not exito:
        break
    #Mostrar las imagenes
    cv2.imshow("camara original", fotograma)
    # Salir al presionar ESC
    if cv2.waitKey(1) == 27:
        break
# Liberar la Camara
camara.realease()
cv2.destroyAllWindows()





