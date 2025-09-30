import cv2
import numpy as np

# Valores de calibración
#camera_matrix = np.array([[622.18, 0., 335.15], [0., 623.24, 229.76], [0., 0., 1.]], dtype=np.float32)
#dist_coeffs = np.array([-0.34109674, 0.47033725, 0., 0., -1.06358352], dtype=np.float32)

# Variables globales para el modo de clic
clicked_points = []
homography = None
mode = 'visualization'

# Tamaño de la vista frontal deseada
width, height = 300, 300
dst_points = np.array([[0,0],[width,0],[width,height],[0,height]], dtype=np.float32)

def draw_grid_original(img, H, rows=3, cols=3):
    """
    Dibuja una grilla en la imagen original usando la homografía inversa
    """
    if H is None:
        return
    H_inv = np.linalg.inv(H)

    for i in range(rows+1):
        for j in range(cols+1):
            # Coordenadas en el plano frontal (0 a width, 0 a height)
            pt_frontal = np.array([[j/cols*width, i/rows*height]], dtype=np.float32)
            pt_frontal = np.array([pt_frontal])
            # Transformar al plano original
            pt_orig = cv2.perspectiveTransform(pt_frontal, H_inv)[0][0]
            pt_orig = tuple(pt_orig.astype(int))
            cv2.circle(img, pt_orig, 3, (0,255,0), -1)

    # Líneas horizontales
    for i in range(rows+1):
        start = np.array([[[0, i/rows*height]]], dtype=np.float32)
        end = np.array([[[width, i/rows*height]]], dtype=np.float32)
        start_orig = cv2.perspectiveTransform(start, H_inv)[0][0]
        end_orig = cv2.perspectiveTransform(end, H_inv)[0][0]
        cv2.line(img, tuple(start_orig.astype(int)), tuple(end_orig.astype(int)), (0,255,0), 1)

    # Líneas verticales
    for j in range(cols+1):
        start = np.array([[[j/cols*width, 0]]], dtype=np.float32)
        end = np.array([[[j/cols*width, height]]], dtype=np.float32)
        start_orig = cv2.perspectiveTransform(start, H_inv)[0][0]
        end_orig = cv2.perspectiveTransform(end, H_inv)[0][0]
        cv2.line(img, tuple(start_orig.astype(int)), tuple(end_orig.astype(int)), (0,255,0), 1)

def draw_grid_frontal(frontal_img, rows=3, cols=3):
    """
    Dibuja una grilla directamente sobre la imagen frontal (sin transformación)
    """
    h, w = frontal_img.shape[:2]
    for i in range(1, rows):
        y = int(i * h / rows)
        cv2.line(frontal_img, (0, y), (w, y), (0,255,0), 1)
    for j in range(1, cols):
        x = int(j * w / cols)
        cv2.line(frontal_img, (x, 0), (x, h), (0,255,0), 1)

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
    #undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    #display_frame = undistorted_frame.copy()

    # Dibujar puntos marcados manualmente
    if mode == 'manual' and len(clicked_points) > 0:
        for pt in clicked_points:
            cv2.circle(frame, tuple(pt), 5, (0,0,255), -1)

    # Dibujar grilla en la imagen original
    if homography is not None:
        draw_grid_original(frame, homography)

    cv2.imshow("Webcam", frame)

    # Mostrar perspectiva homográfica
    if homography is not None:
        frontal = cv2.warpPerspective(frame, homography, (width, height))
        draw_grid_frontal(frontal)  # Dibujar grilla en la imagen frontal
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