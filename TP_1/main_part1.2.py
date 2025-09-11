
import cv2 as cv
import numpy as np
import os
import csv

ancho=800
alto=800

area_minima=100*100
area_maxima=700*700

def filtrarContornos(contornos,area_minima,area_maxima):
  contornos_filtrados=[]
  for contorno in contornos:
      area=cv.contourArea(contorno)
      if area>=area_minima and area<=area_maxima:
            contornos_filtrados.append(contorno)
  return contornos_filtrados


#Circulo de Referencia
CirculoOrginal=cv.imread('CirculoReferencia.jpg', cv.IMREAD_COLOR)
CirculoOrginal=cv.resize(CirculoOrginal,(ancho,alto))

Circulo=cv.imread('CirculoReferencia.jpg', cv.IMREAD_GRAYSCALE)
Circulo=cv.resize(Circulo,(ancho,alto))
_, CirculoBinary = cv.threshold(Circulo,150, 255, cv.THRESH_BINARY)
contornos, jerarquia=cv.findContours(CirculoBinary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
CirculoContornos=filtrarContornos(contornos,area_minima,area_maxima)
cv.drawContours (CirculoOrginal, CirculoContornos, -1, (0,255,0), 3)
cv.imshow("CirculoReferencia",CirculoOrginal)


#Cuadrado de Referencia
CuadradoOrginal=cv.imread('CuadradoReferencia.jpg', cv.IMREAD_COLOR)
CuadradoOrginal=cv.resize(CuadradoOrginal,(ancho,alto))

Cuadrado=cv.imread('CuadradoReferencia.jpg', cv.IMREAD_GRAYSCALE)
Cuadrado=cv.resize(Cuadrado,(ancho,alto))
_, CuadradoBinary = cv.threshold(Cuadrado,150, 255, cv.THRESH_BINARY)
contornos, jerarquia=cv.findContours(CuadradoBinary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
CuadradoContornos=filtrarContornos(contornos,area_minima,area_maxima)
cv.drawContours (CuadradoOrginal, CuadradoContornos, -1, (0,255,0), 3)
cv.imshow("CuadradoReferencia",CuadradoOrginal)


#Triangulo de Referencia
TrianguloOrginal=cv.imread('TrianguloReferencia.jpg', cv.IMREAD_COLOR)
TrianguloOrginal=cv.resize(TrianguloOrginal,(ancho,alto))

Triangulo=cv.imread('TrianguloReferencia.jpg', cv.IMREAD_GRAYSCALE)
Triangulo=cv.resize(Triangulo,(ancho,alto))
_, TrianguloBinary = cv.threshold(Triangulo,150, 255, cv.THRESH_BINARY)
contornos,jerarquia=cv.findContours(TrianguloBinary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
TrianguloContornos=filtrarContornos(contornos,area_minima,area_maxima)
cv.drawContours (TrianguloOrginal, TrianguloContornos, -1, (0,255,0), 3)
cv.imshow("TrianguloReferencia",TrianguloOrginal)


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

def validarCategorias(contorno):
    mejor_coincidencia = None
    menor_distancia = float('inf')
    for contorno_ref in CirculoContornos:
      distancia = cv.matchShapes(contorno, contorno_ref, cv.CONTOURS_MATCH_I1, 0.0)
      if distancia < menor_distancia:
          menor_distancia = distancia
          mejor_coincidencia = 'Circulo' 
    for contorno_ref in CuadradoContornos:
      distancia = cv.matchShapes(contorno, contorno_ref, cv.CONTOURS_MATCH_I1, 0.0)
      if distancia < menor_distancia:
          menor_distancia = distancia
          mejor_coincidencia = 'Cuadrado' 
    for contorno_ref in TrianguloContornos:
      distancia = cv.matchShapes(contorno, contorno_ref, cv.CONTOURS_MATCH_I1, 0.0)
      if distancia < menor_distancia:
          menor_distancia = distancia
          mejor_coincidencia = 'Triangulo' 
    return menor_distancia,mejor_coincidencia

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
  X=[]
  Y=[]
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

    menor_distancia,mejor_coincidencia=validarCategorias(contorno)
    if menor_distancia < umbral_match:
      cv.putText(frame, f"{mejor_coincidencia}", (x, y-10),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
      cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
      momentos = cv.moments(contorno)
      hu_momentos = cv.HuMoments(momentos).flatten().tolist()
      X.append(hu_momentos)
      Y.append(mejor_coincidencia)
    else:
      cv.putText(frame, "Desconocido", (x, y-10),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
      cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
      momentos = cv.moments(contorno)
      hu_momentos = cv.HuMoments(momentos).flatten().tolist()
      X.append(hu_momentos)
      Y.append("Desconocido")


    

    #else:
    #  cv.putText(frame, "Sin informacion", (x, y-10),
    #              cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

  cv.imshow("Debug", frame_morfo)
  cv.imshow("Manual", frame)
  
  key = cv.waitKey(30) & 0xFF
  if key == ord('g'):
     guardar_dataset()
  if key == ord('q'):
    break


#guardar_dataset()
webcam.release()
cv.destroyAllWindows()
