import cv2
import numpy as np

# Crear ventana
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

points = []
count = 0
circle_positions = []  # Lista para almacenar las posiciones anteriores del círculo

def get_mouse_points(event, x, y, flags, param):
    global points, count
    if event == cv2.EVENT_LBUTTONDOWN and count < 4:
        points.append((x, y))
        count += 1
        print("Punto agregado: ", x, y)

cv2.setMouseCallback('Video', get_mouse_points)

# Parámetros para la detección de círculos
circle_params = {
    'dp': 1.2,
    'minDist': 50,
    'param1': 100,
    'param2': 40,
    'minRadius': 10,
    'maxRadius': 30
}

# Inicializar capturador de video
cap = cv2.VideoCapture(1)  # Cambiar el número a 0 para usar la cámara predeterminada

while True:

    # Leer el fotograma del video
    ret, frame = cap.read()

    # Dibujar los puntos seleccionados en el fotograma
    for point in points:
        cv2.circle(frame, point, 5, (0, 255, 0), -1)

    # Mostrar el fotograma original en la ventana 'Video'
    cv2.imshow('Video', frame)

    # Esperar la tecla 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Verificar si se han seleccionado 4 puntos
    if len(points) == 4:
        print("Aplicando transformación homográfica...")
        # Convertir la lista de puntos a un arreglo NumPy
        points = np.array(points)

        w = frame.shape[1]  # Ancho del fotograma
        h = frame.shape[0]  # Alto del fotograma
        front_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

        # Calcular la matriz homográfica
        H, _ = cv2.findHomography(points, front_points)

        print("Matriz Homográfica:")
        print(H)

        # Corregir distorsión no lineal
        img_undistorted = cv2.undistort(frame, np.eye(3), np.zeros(5))

        # Aplicar la transformación al fotograma corregido
        img_frontal = cv2.warpPerspective(img_undistorted, H, (w, h))

        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(img_frontal, cv2.COLOR_BGR2GRAY)

        # Aplicar desenfoque
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detección de círculos
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            **circle_params
        )

        # Dibujar círculos detectados y marcar trayectoria
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Dibujar círculo detectado
                cv2.circle(img_frontal, (x, y), r, (0, 255, 0), 2)

                # Agregar la posición del círculo a la lista de posiciones anteriores
                circle_positions.append((x, y))

                # Dibujar la trayectoria del círculo con una línea roja continua
                for i in range(1, len(circle_positions)):
                    cv2.line(img_frontal, circle_positions[i - 1], circle_positions[i], (0, 0, 255), 2)
        else:
            # Dibujar la trayectoria del círculo con una línea roja continua, aunque el círculo no esté presente
            for i in range(1, len(circle_positions)):
                cv2.line(img_frontal, circle_positions[i - 1], circle_positions[i], (0, 0, 255), 2)

        # Mostrar el fotograma frontal con los círculos detectados y la trayectoria marcada
        cv2.imshow('Detected Circles', img_frontal)

    # Esperar la tecla 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

