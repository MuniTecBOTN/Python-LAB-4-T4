
import cv2

camara = cv2.VideoCapture(0)

while camara.isOpened():
    exito, fotograma = camara.read()
    if not exito:
        break

    altura_imagen, ancho_imagen, canales_color = fotograma.shape
    print(f"INF_IMAGEN: {altura_imagen}x{ancho_imagen} - {canales_color}")

    imagen_grises = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)

    # Reducir ruido sin cambiar la estructura original
    imagen_grises = cv2.GaussianBlur(imagen_grises, (5, 5), 0)

    _, imagen_binaria = cv2.threshold(
        imagen_grises,
        127,
        255,
        cv2.THRESH_BINARY_INV
    )

    contornos, _ = cv2.findContours(
        imagen_binaria,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    escala_fuente = min(ancho_imagen, altura_imagen) * 0.001
    grosor_fuente = max(1, round(escala_fuente * 2))

    for index, contorno in enumerate(contornos):
        area = cv2.contourArea(contorno)

        if area < 10000:
            continue

        x_pi, y_pi, ancho_caja, alto_caja = cv2.boundingRect(contorno)

        # Caracteristicas geometricas del objeto
        if alto_caja == 0 or ancho_caja == 0:
            continue

        proporcion = max(
            ancho_caja / alto_caja,
            alto_caja / ancho_caja
        )

        area_rectangulo = ancho_caja * alto_caja
        rectangularidad = area / area_rectangulo

        perimetro = cv2.arcLength(contorno, True)

        if perimetro > 0:
            circularidad = (4 * 3.1416 * area) / (perimetro ** 2)
        else:
            circularidad = 0

        # Clasificacion aproximada por forma
        if proporcion >= 2.0 and rectangularidad >= 0.55:
            objeto = "Telefono / Control"

        elif rectangularidad >= 0.80 and proporcion < 1.4:
            objeto = "Objeto rectangular"

        elif circularidad >= 0.75:
            objeto = "Objeto circular"

        else:
            objeto = "Objeto irregular"

        cv2.rectangle(
            fotograma,
            (x_pi - 10, y_pi - 10),
            (x_pi + ancho_caja + 10, y_pi + alto_caja + 10),
            (123, 255, 77),
            2,
        )

        cv2.drawContours(
            fotograma,
            [contorno],
            -1,
            (255, 26, 200),
            1,
        )

        y_etiqueta_objeto = y_pi - 20
        x_etiqueta_objeto = x_pi - 10

        if y_etiqueta_objeto <= 0:
            y_etiqueta_objeto = y_pi + alto_caja + 45

        cv2.putText(
            fotograma,
            f"{objeto} #{index}",
            (x_etiqueta_objeto, y_etiqueta_objeto),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 26, 200),
            1,
        )

        etiqueta_area = f"AREA: {area:.2f} px2"
        posicion_area = (x_pi, y_pi + alto_caja + 25)

        cv2.putText(
            fotograma,
            etiqueta_area,
            posicion_area,
            cv2.FONT_HERSHEY_SIMPLEX,
            escala_fuente,
            (255, 26, 200),
            grosor_fuente,
        )

        print(f"\nCONTORNO #{index}")
        print(f"AREA: {area:.2f} px2")
        print(f"PROPORCION: {proporcion:.2f}")
        print(f"RECTANGULARIDAD: {rectangularidad:.2f}")
        print(f"CIRCULARIDAD: {circularidad:.2f}")
        print(f"OBJETO ESTIMADO: {objeto}")

    cv2.imshow("Camara original", fotograma)
    cv2.imshow("Imagen binaria", imagen_binaria)

    if cv2.waitKey(1) == 27:
        break

camara.release()
cv2.destroyAllWindows()
