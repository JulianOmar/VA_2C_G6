import cv2
import numpy as np

# Valores de calibración
camera_matrix = np.array([[622.18, 0., 335.15], [0., 623.24, 229.76], [0., 0., 1.]], dtype=np.float32)
dist_coeffs = np.array([-0.34109674, 0.47033725, 0., 0., -1.06358352], dtype=np.float32)

# Variables globales para el modo de clic
clicked_points = []
homography = None
mode = 'visualization'

# Tamaño de la vista frontal deseada
width, height = 300, 300
dst_points = np.array([[0,0],[width,0],[width,height],[0,height]], dtype=np.float32)

def draw_grid(img, H, rows=3, cols=3):
    for i in range(rows+1):
        for j in range(cols+1):
            pt = np.array([[j/cols*width, i/rows*height]], dtype=np.float32)
            pt = cv2.perspectiveTransform(np.array([pt]), H)[0][0]
            pt = tuple(pt.astype(int))
            cv2.circle(img, pt, 3, (0,255,0), -1)
    # Opcional: líneas horizontales y verticales
    for i in range(1, rows):
        start = cv2.perspectiveTransform(np.array([[[0, i/rows*height]]], dtype=np.float32), H)[0][0]
        end = cv2.perspectiveTransform(np.array([[[width, i/rows*height]]], dtype=np.float32), H)[0][0]
        cv2.line(img, tuple(start.astype(int)), tuple(end.astype(int)), (0,255,0), 1)
    for j in range(1, cols):
        start = cv2.perspectiveTransform(np.array([[[j/cols*width, 0]]], dtype=np.float32), H)[0][0]
        end = cv2.perspectiveTransform(np.array([[[j/cols*width, height]]], dtype=np.float32), H)[0][0]
        cv2.line(img, tuple(start.astype(int)), tuple(end.astype(int)), (0,255,0), 1)

def mouse_callback(event, x, y, flags, param):
    global clicked_points, homography, mode
    if mode == 'manual' and event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        if len(clicked_points) == 4:
            pts = np.array(clicked_points, dtype=np.float32)
            homography = cv2.getPerspectiveTransform(pts, dst_points)
            clicked_points = []
            mode = 'visualization'

cap = cv2.VideoCapture(2)
cv2.namedWindow("Webcam")
cv2.setMouseCallback("Webcam", mouse_callback)

qr_detector = cv2.QRCodeDetector()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Corregir distorsión antes de cualquier procesamiento
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    display_frame = undistorted_frame.copy()

    # Dibujar grilla si hay homografía
    if homography is not None:
        draw_grid(display_frame, homography)

    cv2.imshow("Webcam", display_frame)

    # Mostrar perspectiva homográfica
    if homography is not None:
        frontal = cv2.warpPerspective(undistorted_frame, homography, (width, height))
        cv2.imshow("Frontal", frontal)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # Modo QR
        mode = 'qr'
        data, pts, _ = qr_detector.detectAndDecode(frame)
        if pts is not None:
            pts = pts[0].astype(np.float32)
            homography = cv2.getPerspectiveTransform(pts, dst_points)
        mode = 'visualization'

    elif key == ord('h'):
        # Modo manual
        mode = 'manual'
        clicked_points = []

    elif key == ord('q'):  # ESC para salir
        break


cap.release()
cv2.destroyAllWindows()

