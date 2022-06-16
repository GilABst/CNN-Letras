#!/usr/bin/env python
# coding: utf-8

# In[1]:

#IMPORTACIONES NECESARIAS
import cv2 as cv
import math
import numpy as np


# In[2]:

# FUNCIÓN IMAGENES, NOS PERMITE GIRAR, TRANSLADAR Y GUARDAR LAS IMAGENES GENERADAS
def Imagenes(Img, General, ejex, ejey):
    
    #ESTABLECEMOS RUTA DE IMAGEN Y SE LEVANTA
    img1 = cv.imread('C:\\Users\\gilba\\Desktop\\IA\\DataSet\\'+str(Img))
    
    #ELIMINAMOS RUIDO DE LA IMAGEN CON FUNCION GAUSSIANBLUR 
    #PRIMER PARAMETRO IMAGEN A LIMPIAR
    #SEGUNDO PARAMETRO TAMANIO DE KERNEL, EN ESTE CASO 5 X 5 PX SOLO QUEREMOS ELIMIANR RUIDO NO QUEREMOS VERLA BORROSA
    #TERCER PARAMETEO ESPECIFICA LOS LIMITES DE LA IMAGEN
    gaussiana = cv.GaussianBlur(img1, (5,5), 0)

    #CONVERTIMOS DE RGB A GRISES
    #PRIMER PARAMETRO IMAGEN A CONVERTIR
    #SEGUNDO PARAMETRO DE QUE ESCALA A QUE ESCALA
    imageOut = cv.cvtColor(gaussiana, cv.COLOR_BGR2GRAY)
    
    ancho = img1.shape[0]
    alto = img1.shape[1]
    
    ##APLICAMOS FILTRO BLANCO Y NEGRO PARA BINARIZAR LA LETRA Y ELIMINAR RUIDO
    #MIDE FILAS Y COLUMNAS DE LA IMAGEN
    for x in range(imageOut.shape [0]):       
        for y in range (imageOut.shape[1]):
            if imageOut[x,y] < 150:
                imageOut[x,y] =  255 #PONE EL PIXEL EN BLANCO
            else:
                imageOut[x,y] = 0 #SINO LO PONE NEGRO

    #GENERAMOS LA MATRIZ DE TRANSFORMACION DE LA IMAGEN, PARA TRANSLADARLA
    M= np.float32([[1,0,ejex], [0,1,ejey]])
    #WARPAFFINE TRANSLADO POR MEDIO DE OPERACIONES MATRICIALES
    #PRIMER PARAMETRO LA IMAGEN A MODIFICAR
    #SEGUNDO PARAMATRO MATRIZ TRANSFORMADA
    #TERCER PARAMETRO ANCHO Y ALTO DE LA IMGEN
    img = cv.warpAffine(imageOut, M, (ancho, alto))
    
    #DATOS DE IMAGEN
    h,w = img.shape[:2] #SACAMOS ALTURA Y ANCHURA DE LA IMAGEN HEIGTH, WEIGHT
    imgz = np.zeros((h*2, w*2), dtype = 'uint8') #FUNCION NUMPY ZEROS SIRVE PARA CREAR UNA MATRIZ DE CEROS  
    
    
    ##CONTADOR DE GUARDADO Y GRADOS DE INCLINACION DE IMAGEN
    num_img = -90

    
    ## GIRAMOS LA IMAGEN
    ## GIRA LA LETRA - GIRO A MANECILLAS DEL RELOJ
    ##              + GIRO EN CONTRA DE MANECILLAS
    while(num_img != 91):
        #GIRAMOS LA IMAGEN CON LA FUNCION GETROTATION
        #PRIMER PARAMETRO NOS SIRVE PARA CENTRAR LA IMAGEN
        #SEGUNDO PARAMETRO GRADO DE GIRO
        #TERCER PARAMETRO FACTOR ESCALA
        mw = cv.getRotationMatrix2D( (h//2, w//2), num_img, 1 )
        #DEFINIMOS IMAGEN DE SALIDA, AFINAMOS CON WARPAFFINE
        #PRIMER PARAMETRO IMAGEN A MODIFICAR
        #SEGUNDO PARAMETRO LA MATRIZ TRANSFORMADA
        #TERCER PARAMETRO TAMANIO DE LA IMAGEN DE SALIDA
        imgz = cv.warpAffine(img,mw,(h,w))
        numero = num_img +650
        ruta = 'C:\\Users\\gilba\\Desktop\\IA\\DataSet\\'+str(General)+str(ejex)+'_'+str(ejey)+'_'+ str(numero)+').jpg'
        cv.imwrite(ruta, imgz)
        num_img += 1
    


# In[3]:


#ESTABLECEMOS LA RUTA GENERAL DE LA CARPETA
General = [
    "0\\0 (","1\\1 (","2\\2 (","3\\3 (","4\\4 (","5\\5 (","6\\6 (","7\\7 (","8\\8 (","9\\9 (","AMayus\\A (","aMinus\\a (",
    "BMayus\\B (","bMinus\\b (","CMayus\\C (","cMinus\\c (","DMayus\\D (","dMinus\\d (","EMayus\\E (","eMinus\\e (","FMayus\\F (","fMinus\\f (","GMayus\\G (","gMinus\\g (",
    "HMayus\\H (","hMinus\\h (","IMayus\\I (","iMinus\\i (","JMayus\\J (","jMinus\\j (","KMayus\\K (","kMinus\\k (","LMayus\\L (","lMinus\\l (","MMayus\\M (","mMinus\\m (",
    "NMayus\\N (","nMinus\\n (","NNMayus\\NN (","nnMinus\\nn (","OMayus\\O (","oMinus\\o (","PMayus\\P (","pMinus\\p (","QMayus\\Q (","qMinus\\q (","RMayus\\R (","rMinus\\r (",
    "SMayus\\S (","sMinus\\s (","TMayus\\T (","tMinus\\t (","UMayus\\U (","uMinus\\u (","VMayus\\V (","vMinus\\v (","WMayus\\W (","wMinus\\w (","XMayus\\X (","xMinus\\x (",
    "YMayus\\Y (","yMinus\\y (","ZMayus\\Z (","zMinus\\z ("
]

#ESTABLECEMOS LA RUTA GENERAL DE LA IMAGEN 
Img = [
    "0\\03.jpg","1\\13.jpg","2\\23.jpg","3\\33.jpg","4\\43.jpg","5\\53.jpg","6\\63.jpg","7\\73.jpg","8\\83.jpg","9\\93.jpg","AMayus\\A3.jpg","aMinus\\a3.jpg",
    "BMayus\\B3.jpg","bMinus\\b3.jpg","CMayus\\C3.jpg","cMinus\\c3.jpg","DMayus\\D3.jpg","dMinus\\d3.jpg","EMayus\\E3.jpg","eMinus\\e3.jpg","FMayus\\F3.jpg","fMinus\\f3.jpg","GMayus\\G3.jpg","gMinus\\g3.jpg",
    "HMayus\\H3.jpg","hMinus\\h3.jpg","IMayus\\I3.jpg","iMinus\\i3.jpg","JMayus\\J3.jpg","jMinus\\j3.jpg","KMayus\\K3.jpg","kMinus\\k3.jpg","LMayus\\L3.jpg","lMinus\\l3.jpg","MMayus\\M3.jpg","mMinus\\m3.jpg",
    "NMayus\\N3.jpg","nMinus\\n3.jpg","NNMayus\\NN3.jpg","nnMinus\\nn3.jpg","OMayus\\O3.jpg","oMinus\\o3.jpg","PMayus\\P3.jpg","pMinus\\p3.jpg","QMayus\\Q3.jpg","qMinus\\q3.jpg","RMayus\\R3.jpg","rMinus\\r3.jpg",
    "SMayus\\S3.jpg","sMinus\\s3.jpg","TMayus\\T3.jpg","tMinus\\t3.jpg","UMayus\\U3.jpg","uMinus\\u3.jpg","VMayus\\V3.jpg","vMinus\\v3.jpg","WMayus\\W3.jpg","wMinus\\w3.jpg","XMayus\\X3.jpg","xMinus\\x3.jpg",
    "YMayus\\Y3.jpg","yMinus\\y3.jpg","ZMayus\\Z3.jpg","zMinus\\z3.jpg"
]

#MATRIZ DE TRANSLACION DE IMAGEN DESDE (-1,-1) HASTA (1,1) 
lados = [[1,-1,1,-1],[1,1,-1,-1]]

#INICIALIZACION UNICAMENTE PARA PODER CONTINUAR
x = 1
y = 1

#CICLO, RECORREMOS EL ARREGLO DE RUTA GENERAL Y RUTA DE LA IMAGEN PARA LOGRAR LA AUTOMATIZACION DEL PROCESO
size = len(General)
for h in range(size):
    for i in range(4):
        for j in range(10):
            #CALCULAMOS LA TRANSLACION QUE LE DARA A LA IMAGEN POR CICLO
            #SE MULTIPLICA POR J PARA DARLE UNA TRANSLACION DE HASTA 10 PIXELES DE IZQUIERDA A DERECHA, ARRIBA A ABAJO
            #PRIMERO EN LA TRANSLACION EJE X
            x = lados [0][i] * (j)
            #SEGUNDO TRANSLACION EJE Y
            #y = lados [1][i] * (j) SOLO USE TRANSLACION EJE X, YA QUE SI USABA LA TRANSLACION EJE Y SE PERDIAN CARACTERISTICAS ESPECIFICAS
            #DE ALGUNAS IMAGENES COMO LA 'NN'
            Imagenes(Img[h], General[h],x,y)
    
##FINALIZAR
print ("PROCESO FINALIZADO")


# In[ ]:




