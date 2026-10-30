import cv2

# Importa la camara web
camara = cv2.VideoCapture(0)
camara.set(cv2.CAP_PROP_FPS, 45)
fps = camara.get(cv2.CAP_PROP_FPS)
while camara.isOpened():
    # Captura un fotograma
    exito, fotograma = camara.read()
    if not exito:
        break

    altura_imagen, ancho_imagen, canales_color = fotograma.shape

    print(f"INF_IMAGEN: {altura_imagen}x{ancho_imagen} - {canales_color}")

    imagen_grises = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)

    _, imagen_binaria = cv2.threshold(
        imagen_grises,
        100,
        255,
        cv2.THRESH_BINARY_INV,
    )

    contornos, _ = cv2.findContours(
        imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )


    for index, contorno in enumerate(contornos):
        area = cv2.contourArea(contorno)
        if area < 10000:
            continue
        x_pi, y_pi, ancho_caja, alto_caja = cv2.boundingRect(contorno)

        cv2.rectangle(
            fotograma,
            (x_pi - 10, y_pi - 10),
            (x_pi + ancho_caja + 10, y_pi + alto_caja + 10),
            (127, 255, 22),
            2,
        )

        print(f"\nContorno #{index}")
        print(f"AREA: {area} px")
        for punto in contorno:
            print(f"{punto}")

        cv2.drawContours(fotograma, contornos, index, (0, 255, 0), 1)

        etiqueta_y = y_pi - 40

        if etiqueta_y <= 0:
            etiqueta_y = y_pi + alto_caja + 45

        cv2.putText(
            fotograma,
            "No. Objeto: #" + str(index) + f"\nArea: " + str(area),
            (x_pi, max(etiqueta_y - 10, -10)),
            cv2.FONT_HERSHEY_COMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
    # Mostrar las imagenes
    cv2.imshow("camara_original", fotograma)
    cv2.imshow("camara_binaria", imagen_binaria)
    # salir al presionar ESC
    if cv2.waitKey(1) == 27:
        break

# Liberar la camara
camara.release()
cv2.destroyAllWindows()
