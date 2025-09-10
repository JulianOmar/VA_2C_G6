
import cv2 as cv
import numpy as np
import os
import csv

C = [] # PASAR A IMAGENES, UNA IMAGEN POR FIGURA
X = []
Y = []

if os.name == "nt":
  webcam = cv.VideoCapture(0, cv.CAP_DSHOW) # WINDOWS
elif os.name == "posix":
  webcam = cv.VideoCapture(0) # LINUX / OS

cv.namedWindow("Manual")
cv.createTrackbar("Umbral", "Manual", 40, 255, lambda x: None)
cv.createTrackbar("Morfologico", "Manual", 3, 20, lambda x: None)
cv.createTrackbar("Area", "Manual", 300, 5000, lambda x: None)
cv.createTrackbar("Umbral_Match", "Manual", 0, 100, lambda x: None)

def umbral_manual(frame):
  t = cv.getTrackbarPos("Umbral", "Manual")
  _, th_manual = cv.threshold(frame, t, 255, cv.THRESH_BINARY_INV)
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

def buscar_contornos(frame):
  contornos, _ = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
  min_area = cv.getTrackbarPos("Area", "Manual")
  contornos_filtrados = [cnt for cnt in contornos if cv.contourArea(cnt) > min_area]
  return contornos_filtrados

def guardar_dataset():
  with open('data.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    for x_row, y_val in zip(X, Y):
        writer.writerow(x_row + [y_val])

while True:
  ret, frame = webcam.read()

  if not ret: break

  gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

  frame_thresold = umbral_manual(gray)

  frame_morfo = operaciones_morfologicas(frame_thresold)

  contornos = buscar_contornos(frame_morfo)
  
  print(len(contornos))

  for contorno in contornos:
    #cv.drawContours(frame, [contorno], -1, (0,255,0), 2)
  
    x, y, w, h = cv.boundingRect(contorno)

    umbral_match = cv.getTrackbarPos("Umbral_Match", "Manual") / 100

    #if C:
    mejor_coincidencia = None
    menor_distancia = float('inf')
    for index, contorno_ref in enumerate(C):
      distancia = cv.matchShapes(contorno, contorno_ref, cv.CONTOURS_MATCH_I1, 0.0)
      if distancia < menor_distancia:
          menor_distancia = distancia
          mejor_coincidencia = Y[index]

    if menor_distancia < umbral_match:
      cv.putText(frame, f"{mejor_coincidencia}", (x, y-10),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
      cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
      cv.putText(frame, "Desconocido", (x, y-10),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
      cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #else:
    #  cv.putText(frame, "Sin informacion", (x, y-10),
    #              cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

  cv.imshow("Debug", frame_morfo)
  cv.imshow("Manual", frame)
  
  key = cv.waitKey(30) & 0xFF

  #if key == ord('c') and contornos:
  #  nombre = input("Ingrese etiqueta de esta forma: ")
  #  for index, cnt in enumerate(contornos):
  #    C.append(cnt.copy())
  #    X.append(cv.HuMoments(cv.moments(cnt)).flatten().tolist())
  #    Y.append(nombre)
  #    print(f"Guardado contorno de referencia: {nombre}")
  if key == ord('q'):
    break


#guardar_dataset()
webcam.release()
cv.destroyAllWindows()
