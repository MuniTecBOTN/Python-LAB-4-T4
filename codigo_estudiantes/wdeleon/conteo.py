import cv2

class Contador:
    def __init__(self, categorias, linea_x, sentido=+1):
        """
        linea_x: posición X de la línea virtual de conteo (vertical).
        sentido: +1 si las piezas avanzan de izquierda a derecha, -1 al contrario.
        """
        self.linea_x = linea_x
        self.sentido = sentido
        self.conteos = {c: 0 for c in categorias}
        self.conteos["desconocida"] = 0
        self.conteos["no_evaluable"] = 0

    def evaluar_cruce(self, objetos):
        """Debe llamarse cada frame después de actualizar el tracking."""
        for obj in objetos:
            if obj.contado:
                continue
            cx, _ = obj.centro()
            px, _ = obj.centro_prev
            cruzó = (px - self.linea_x) * self.sentido <= 0 < (cx - self.linea_x) * self.sentido
            if not cruzó:
                continue
            # Solo se cuenta si el objeto ya fue clasificado
            if obj.estado in ("conocida", "desconocida", "no_evaluable"):
                clave = obj.categoria if obj.estado == "conocida" else obj.estado
                if clave not in self.conteos:
                    self.conteos["desconocida"] += 1
                else:
                    self.conteos[clave] += 1
                obj.contado = True

    def dibujar_linea(self, frame):
        h = frame.shape[0]
        cv2.line(frame, (self.linea_x, 0), (self.linea_x, h), (255, 200, 0), 2)

    def dibujar_panel(self, frame):
        y = 25
        cv2.putText(frame, "CONTEO", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        for k, v in self.conteos.items():
            y += 25
            cv2.putText(frame, f"{k}: {v}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)