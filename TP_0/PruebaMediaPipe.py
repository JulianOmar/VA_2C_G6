import cv2 as cv
import numpy as np
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib
import time


cap=cv.VideoCapture(0, cv.CAP_DSHOW)

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode


mp_hands=mp.solutions.hands

base_options_stylizer = None

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result.gestures))
    global base_options_stylizer
    if len(result.gestures) != 0:
      if result.gestures[0][0].category_name == 'Victory':
        base_options_stylizer= python.BaseOptions(model_asset_path='TP_0/face_stylizer_oil_painting.task')
        print('Victoria')
      elif result.gestures[0][0].category_name == 'Open_Palm':
        base_options_stylizer= python.BaseOptions(model_asset_path='TP_0/face_stylizer_color_sketch.task')
        print('Palma')
      elif result.gestures[0][0].category_name == 'Closed_Fist':
        base_options_stylizer= python.BaseOptions(model_asset_path='TP_0/face_stylizer_color_ink.task')
        print('puño')
      elif result.gestures[0][0].category_name == 'Thumb_Up':        
        print('finalizar_programa')
        finalizar_programa(cap)
      else:
        base_options_stylizer=None
    else:
      print('Ningún gesto')
      base_options_stylizer=None



options_gesture  = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='TP_0/gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
GestureRecognizer = mp.tasks.vision.GestureRecognizer


with GestureRecognizer.create_from_options(options_gesture) as recognizer:
    while True:
      ret, frame=cap.read()
      if ret == False:
         break
      height, width, _=frame.shape
      frame=cv.flip(frame,1)
      frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
      recognition_result = recognizer.recognize_async(mp_image,time.time_ns()//1_000_000)

      if base_options_stylizer is not None:
        options = vision.FaceStylizerOptions(base_options=base_options_stylizer)
        with vision.FaceStylizer.create_from_options(options) as stylizer:
          stylized_image = stylizer.stylize(mp_image)
          rgb_stylized_image = cv.cvtColor(stylized_image.numpy_view(), cv.COLOR_BGR2RGB)

          cv.imshow('Modelo',rgb_stylized_image)
          cv.imshow('Cara',frame)
      else:
          cv.imshow('Cara',frame)
      if cv.waitKey(1) & 0xFF == 27:
            break


    # Show the stylized image
def finalizar_programa(cap):
    """
    Libera los recursos de la cámara y cierra todas las ventanas de OpenCV.
    """
    if cap is not None:
        cap.release()
    cv.destroyAllWindows()

