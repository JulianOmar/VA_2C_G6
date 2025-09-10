from sklearn import tree
from joblib import dump, load
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 as cv
import os

if os.name == "nt":
  webcam = cv.VideoCapture(0, cv.CAP_DSHOW) # WINDOWS
elif os.name == "posix":
  webcam = cv.VideoCapture(0) # LINUX / OS

try:
  df = pd.read_csv('data.csv')
except:
  print("No existe el archivo data.csv")
  exit(1)

X = df.drop('label', axis=1).astype(float).values
Y = df['label'].values

clasificador = tree.DecisionTreeClassifier(random_state=42).fit(X, Y)

plt.figure(figsize=(12,8))
tree.plot_tree(
    clasificador, 
    filled=True, 
    feature_names=df.columns[:-1], 
    class_names=df['label'].unique(),
    rounded=True
)
plt.savefig("tree_plot.png", dpi=300)
plt.close()

cv.namedWindow("Manual")
cv.createTrackbar("Umbral", "Manual", 40, 255, lambda x: None)
cv.createTrackbar("Morfologico", "Manual", 3, 20, lambda x: None)
cv.createTrackbar("Area", "Manual", 300, 5000, lambda x: None)

def umbral_manual(frame):
  t = cv.getTrackbarPos("Umbral", "Manual")
  _, th_manual = cv.threshold(frame, t, 255, cv.THRESH_BINARY_INV)
  return th_manual


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



while True:
  ret, frame = webcam.read()

  if not ret: break

  gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

  frame_thresold = umbral_manual(gray)

  frame_morfo = operaciones_morfologicas(frame_thresold)

  contornos = buscar_contornos(frame_morfo)

  for contorno in contornos:
    #cv.drawContours(frame, [contorno], -1, (0,255,0), 2)
  
    x, y, w, h = cv.boundingRect(contorno)

    momentos = cv.moments(contorno)
    hu_momentos = cv.HuMoments(momentos).flatten().tolist()
    descriptor = np.array(hu_momentos).reshape(1, -1)

    probs = clasificador.predict_proba(descriptor)[0]
    best_idx = np.argmax(probs)
    best_proba = probs[best_idx]
    prediccion = clasificador.classes_[best_idx]
    print(probs)
   
    if best_proba > 0.9:
      cv.putText(frame, f"{prediccion}", (x, y-10),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
      cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
      cv.putText(frame, "Desconocido", (x, y-10),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
      cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

  cv.imshow("Debug", frame_morfo)
  cv.imshow("Manual", frame)
  
  key = cv.waitKey(30) & 0xFF

  if key == ord('q'):
    break

webcam.release()
cv.destroyAllWindows()