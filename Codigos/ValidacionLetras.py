#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORTAMOS LAS LIBRERIAS NECESARIAS
import numpy as np
from keras import models
import cv2 as cv


# In[2]:


#CARGAMOS EL MODELO
model = models.load_model("C:\\Users\\gilba\\Desktop\\IA\\ProyectoFinal\\Letras\\letras94.h5py")


# In[3]:


#ASEGURAMOS QUE EL MODELO SE CARGO CORRECTAMENTE, OPCIONAL NO TIENE FUNCIONALIDAD EN EL PROGRAMA
model.summary()


# In[6]:


#ARREGLO DE CLASES PARA CUANDO REALICEMOS LA PREDICCION COINCIDAN

clases_letras = ['0', '1', '2','3','4' ,'5','6','7','8','9','AMayus','aMinus','BMayus','bMinus','CMayus','cMinus','DMayus',
              'dMinus','EMayus','eMinus','FMayus','fMinus','GMayus','gMinus', 'HMayus', 'hMinus', 'IMayus', 'iMinus', 
              'JMayus', 'jMinus', 'KMayus','kMinus', 'LMayus', 'lMinus', 'MMayus', 'mMinus', 'NMayus', 'nMinus', 'NNMayus', 
              'nnMinus', 'OMayus', 'oMinus','PMayus', 'pMinus', 'QMayus', 'qMinus', 'RMayus', 'rMinus', 'SMayus', 'sMinus', 
              'TMayus', 'tMinus', 'UMayus', 'uMinus', 'VMayus', 'vMinus', 'WMayus', 'wMinus','XMayus','xMinus', 'YMayus',
              'yMinus','ZMayus','zMinus']

#DECALRAMOS UN CONTADOR PARA NO DESBORDAR EL CICLO DE BUSQUEDA DE LA PREDICCION, 
#SOBRE TODO PARA QUE SE VEA MAS CLARO, VISUAL
contador = len(clases_letras)


# In[7]:


#TRATAMOS LA IMAGEN Y LA CONVERTIMOS A NUMPY ARRAY 

def tratarImg(im):
    #SE PODRIAN AGREGAR FILTRO DE COLOR Y REDIMENSION PARA TRATAR LA IMAGEN ANTES DE 
    #CONVERTIRLA A UN ARRAY Y REALIZAR LA PREDICCION, ESTO NOS DARIA MAYOR PRESICION
    
    #SE CORRIJE LA GAMA DE LA IMAGEN DE BGR A RGB :)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)

    #SE CREA EL NUMPY ARRAY
    img_array = np.array(im)
    
    #EXPANDE LA MATRIZ ARRAY PARA EVITAR EL ERROR NONE,50,50,3
    img_array = np.expand_dims(img_array, axis=0)
    
    #RETORNAMOS EL NUMPY ARRAY DE LA IMAGEN
    return img_array


# In[8]:


#LEEMOS LA IMAGEN
#IM LA USAREMOS PARA REALIZAR LA PREDICCION
im = cv.imread('C:\\Users\\gilba\\Desktop\\IA\\DataSetLetras\\8\\8 (0_1_42).jpg', 0)

#IM2 LA USAREMOS PARA MOSTRARLA UNICAMENTE
imagen2 = cv.imread('C:\\Users\\gilba\\Desktop\\IA\\DataSetLetras\\8\\8 (0_1_42).jpg')
#RESIZE PARA VER EL RESULTADO MAS AMPLIO
imagen2 = cv.resize(imagen2,(600,600))

mostrar = ""

#SE ENVIA IM AL METODO PARA TRATARLA Y CONVERTIRLA EN ARRAY
img_array = tratarImg(im)

#SE REALIZA LA PREDICCION CON EL MODELO Y LA IMAGEN ARRAY
prediction = model.predict(img_array)#[0][0]

#SE IMPRIME LA PREDICCION
print (prediction)

#RECORREMOS EL ARREGLO Y GUARDAMOS LA PREDICCION PARA AGREGARLA A LA IMAGEN FINAL
for i in range(contador):
    #SI EL VALOR DE LA CASILLA [0][i] ES IGUAL A 1 SACAMOS EL NOMBRE DE ESA MISMA CELDA EN EL ARRAY CLASES 
    if prediction[0][i].all() == 1 :
        #GUARDA LA PREDICCION PARA MOSTRAR DICHO TEXTO EN LA IMAGEN
        mostrar="Predice: "+clases_letras[i]
        #ROMPE EL CICLO SI ENCONTRO UN 1
        break
        
#ELIMINAMOS RUIDO DE LA IMAGEN CON FUNCION GAUSSIANBLUR 
#PRIMER PARAMETRO IMAGEN A LIMPIAR
#SEGUNDO PARAMETRO TAMANIO DE KERNEL, EN ESTE CASO 5 X 5 PX SOLO QUEREMOS ELIMIANR RUIDO NO QUEREMOS VERLA BORROSA
#TERCER PARAMETEO ESPECIFICA LOS LIMITES DE LA IMAGEN
gaussiana = cv.GaussianBlur(imagen2, (5,5), 0)

#CONVERTIMOS DE RGB A GRISES
#PRIMER PARAMETRO IMAGEN A CONVERTIR
#SEGUNDO PARAMETRO DE QUE ESCALA A QUE ESCALA
imageOut = cv.cvtColor(gaussiana, cv.COLOR_BGR2GRAY)

#BINARISAMOS LA IMAGEN PARA VERLA MAS CLARA
for x in range(imageOut.shape[0]):
    for y in range(imageOut.shape[1]):
        if imageOut[x,y] < 150:
            imagen2[x,y]=0 #PONE EL PIXEL EN NEGRO
        else:
            imagen2[x,y]=255 #PONE EL PIXEL EN BLANCO

#PUT TEXT AGREGA EL TEXTO EN LA IMAGEN
#PRIMER PARAMETRO IMAGEN A MOSTRAR
#SEGUNDO PARAMETRO TEXTO A AGREGAR
#TERCER PARAMETRO POSICION
#CUARTO PARAMETRO TIPO DE LETRA
#QUINTO PARAMETRO TAMANIO DE LA LETRA
#SEXTO PARAMETRO COLOR
#SEPTIMO PARAMETRO TRAZO DE LA LETRA, NUMERO DE PIXELES
cv.putText(imagen2, mostrar,(0,50), cv.FONT_HERSHEY_SIMPLEX, 1,(50,255,0),2)
cv.imshow("imagen", imagen2)
cv.waitKey(0)
cv.destroyAllWindows()

