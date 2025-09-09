import cv2
import matplotlib.pyplot as plt
 
umbral=0
actualizacion=True
actualizacion2=True

def getUmbral(valor):
    global umbral  
    global actualizacion
    umbral=valor
    actualizacion=True
def getUmbral2(valor):
    global umbral  
    global actualizacion2
    umbral=valor
    actualizacion2=True
ancho=800
alto=800

# Read the image
img = cv2.imread('cuadrado.jpg', cv2.IMREAD_GRAYSCALE)
img=cv2.resize(img,(ancho,alto))
img_orig = cv2.imread('cuadrado.jpg', cv2.IMREAD_COLOR)
img_orig=cv2.resize(img_orig,(ancho,alto))

img2 = cv2.imread('cuadrado2.jpg', cv2.IMREAD_GRAYSCALE)
img2=cv2.resize(img2,(ancho,alto))
img_orig2 = cv2.imread('caudrado.jpg', cv2.IMREAD_COLOR)
img_orig2=cv2.resize(img_orig2,(ancho,alto))

cv2.namedWindow('binary')
cv2.createTrackbar('Barra deslizante','binary',0,255,getUmbral)

kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
#kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

kernel_open= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#kernel_open=cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
#kernel_open=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
# Display the results

cv2.namedWindow('binary_img2')
cv2.createTrackbar('Barra deslizante2','binary_img2',0,255,getUmbral2)


area_minima=100*100
area_maxima=600*600

while True:
    

    if actualizacion:
        contornos_filtrados = []
        # Obtener copia imagen original y aplicar umbral a la imagen en tonos de grises
        img_orig_cop=img_orig.copy()
        _, binary = cv2.threshold(img,umbral, 255, cv2.THRESH_BINARY)
        """_, binary_inv = cv2.threshold(img, umbral, 255, cv2.THRESH_BINARY_INV)
        _, trunc = cv2.threshold(img, umbral, 255, cv2.THRESH_TRUNC)
        _, tozero = cv2.threshold(img, umbral, 255, cv2.THRESH_TOZERO)
        _, tozero_inv = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)"""
        #dilation = cv2.dilate(binary,kernel,iterations = 1)

        open= cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_open)
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("binary",binary)
        cv2.imshow("Open",open)
        #cv2.imshow("binary_inv",binary_inv)
        #    cv2.imshow("trunc",trunc)
        #    cv2.imshow("tozero",tozero)
        #    cv2.imshow("tozero_inv",tozero_inv)
        cv2.imshow("binary_closing",closing)
        contornos, jerarquia = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for contorno in contornos:
            area=cv2.contourArea(contorno)
            if area>=area_minima and area<=area_maxima:
                contornos_filtrados.append(contorno)
        
        cv2.drawContours (img_orig_cop, contornos_filtrados, -1, (0,255,0), 3)
        cv2.imshow("Original",img_orig_cop)
        actualizacion=False

    if actualizacion2:
        
        contornos_filtrados2 = []
        # Obtener copia imagen original y aplicar umbral a la imagen en tonos de grises
        img_orig2_cop=img_orig2.copy()
        _, binary2 = cv2.threshold(img2,umbral, 255, cv2.THRESH_BINARY)



        open_2= cv2.morphologyEx(binary2, cv2.MORPH_CLOSE, kernel_open)
        closing_2 = cv2.morphologyEx(binary2, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("binary_img2",binary2)
        cv2.imshow("Open_img2",open_2)
        #cv2.imshow("binary_inv",binary_inv)
        #    cv2.imshow("trunc",trunc)
        #    cv2.imshow("tozero",tozero)
        #    cv2.imshow("tozero_inv",tozero_inv)
        cv2.imshow("binary_closing_img2",closing_2)
        contornos2, jerarquia2 = cv2.findContours(binary2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        for contorno in contornos2:
            area=cv2.contourArea(contorno)
            if area>=area_minima and area<=area_maxima:
                contornos_filtrados2.append(contorno)

        cv2.drawContours (img_orig2_cop, contornos_filtrados2, -1, (0,255,0), 3)
        cv2.imshow("Original2",img_orig2_cop)
        actualizacion2=False

    for contorno1 in contornos_filtrados:
        for contorno2 in contornos_filtrados2:
            ret=cv2.matchShapes(contorno1,contorno2,1,0.0)
            if ret <= 0.20:
                print(ret)
                print("Son iguales")
    if cv2.waitKey(30) == ord('0'):
        break

    
cv2.destroyAllWindows()
