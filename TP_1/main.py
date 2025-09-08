
import cv2 as cv
import os

if os.name == "nt":
  webcam = cv.VideoCapture(0, cv.CAP_DSHOW) # WINDOWS
elif os.name == "posix":
  webcam = cv.VideoCapture(0) # LINUX / OS

frame_count = 0
skip = 10

cv.namedWindow("Manual")
cv.createTrackbar("Umbral", "Manual", 127, 255, lambda x: None)
cv.createTrackbar("Morfologico", "Manual", 3, 20, lambda x: None)
cv.createTrackbar("Area", "Manual", 500, 5000, lambda x: None)

def umbral_manual(frame):
  t = cv.getTrackbarPos("Umbral", "Manual")
  _, th_manual = cv.threshold(gray, t, 255, cv.THRESH_BINARY_INV)
  return th_manual

def umbral_otsu(frame):
  _, th_otsu = cv.threshold(frame, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
  return th_otsu

def umbral_triangle(frame):
  _, th_triangle = cv.threshold(frame, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_TRIANGLE)
  return th_triangle

def operaciones_morfologicas(frame):
  ksize = cv.getTrackbarPos("Morfologico", "Manual")
  if ksize < 1: ksize = 1
  kernel = cv.getStructuringElement(cv.MORPH_RECT, (ksize, ksize))

  op_clean = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)
  op_clean = cv.morphologyEx(op_clean, cv.MORPH_CLOSE, kernel)

  return op_clean

def buscar_contorno(frame):
  contornos, _ = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
  min_area = cv.getTrackbarPos("Area", "Manual")
  contornos_filtrados = [cnt for cnt in contornos if cv.contourArea(cnt) > min_area]
  return contornos_filtrados

while True:
  frame_count += 1
  if frame_count % skip != 0:
    continue

  ret, frame = webcam.read()

  if not ret: break

  gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

  frame_thresold = umbral_manual(gray)

  frame_morfo = operaciones_morfologicas(frame_thresold)

  contornos = buscar_contorno(frame_morfo)

  for contorno in contornos:
    cv.drawContours(frame, [contorno], -1, (0,255,0), 2)
  
    x, y, w, h = cv.boundingRect(contorno)
    cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv.putText(frame, f"{contorno}", (x, y-10),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

  cv.imshow("Manual", frame)
  
  if cv.waitKey(30) & 0xFF == ord('q'):
    break

webcam.release()
cv.destroyAllWindows()