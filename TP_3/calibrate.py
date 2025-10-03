import numpy as np
import cv2 as cv

print("""
      Uso:
      Space: Tomar una foto
      C: Calibrar con fotografías tomadas y guardar en calibrate.cfg
      ESC: Salir
      """)

ESC = chr(27)
cv.namedWindow("Detecciones", cv.WINDOW_NORMAL)
cv.namedWindow("Camara", cv.WINDOW_NORMAL)
defaultPrintOptions = np.get_printoptions()

chessBoard = (9,6)
gradualDarkness = 0.90
cornersubpixTerminationCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
imgPoints = []
objPoints = []

# Chessboard 3D points
chessboardPointCloud3D = np.zeros((chessBoard[0]*chessBoard[1],3), np.float32)
chessboardPointCloud3D[:,:2] = np.mgrid[0:chessBoard[0],0:chessBoard[1]].T.reshape(-1,2)

cam = cv.VideoCapture(2)
width = cam.get(cv.CAP_PROP_FRAME_WIDTH)
height = cam.get(cv.CAP_PROP_FRAME_HEIGHT)
print("Resolucion Camara:", width, " x ", height)
newSize = (640, int(640 * height / width))
imBlack = np.zeros(newSize[::-1]+(3,), dtype=np.uint8)

while True:
    ret, im = cam.read()
    if ret:
        imLowRes = cv.resize(im, newSize)
        imGrayLowRes = cv.cvtColor(imLowRes, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(imGrayLowRes, chessBoard, None) 
        if ret:
            cv.drawChessboardCorners(imLowRes, chessBoard, corners, ret)
    
    cv.imshow('Camara', imLowRes)

    key = cv.waitKey(33)
    if key>=0:
        key = chr(key)
        print(key)
        match key:
            case ' ':
                # Repite la detección en alta resolución y la registra
                imGray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
                ret, precisionCorners = cv.findChessboardCorners(imGray, chessBoard, None)
                if ret:
                    precisionCorners = cv.cornerSubPix(imGray, precisionCorners, (11,11), (-1,-1), cornersubpixTerminationCriteria)
                    imgPoints.append(precisionCorners)
                    objPoints.append(chessboardPointCloud3D)

                    # Anota en baja resolución
                    imBlack = cv.convertScaleAbs(imBlack, alpha=gradualDarkness, beta=0)
                    cv.drawChessboardCorners(imBlack, chessBoard, corners, ret)
                    cv.imshow("Detecciones", imBlack)

                    print(len(imgPoints), "fotografías tomadas.")

            case 'c':
                # Calibra
                ret, K, distCoef, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, im.shape[:2][::-1], None, None, flags=cv.CALIB_ZERO_TANGENT_DIST)

                # Guarda resultados en archivo
                coef_list = distCoef.flatten().tolist()
                k_list = K.tolist()
                with open("calibrate.cfg", "w") as f:
                    f.write(f"coeficientes_distorsion = {coef_list}\n")
                    f.write(f"matriz_k = {k_list}\n")
                print("Resultados guardados en calibrate.cfg")

            case ESC:
                print("Terminando.")
                break
