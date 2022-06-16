#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Networks

# 

# # Importar Librerías

# In[1]:


import cv2 as cv
import numpy as np
import os
import re
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[2]:


import keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential,Input,Model
#from keras.layers import Dense, Dropout, Flatten
#rom keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, Conv2D
)
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU


# # Cargar set de Imágenes

# In[3]:


dirname = os.path.join(os.getcwd(),'\\Users\\gilba\\Desktop\\IA\\DataSet')
imgpath = dirname + os.sep 

images = []
directories = []
dircount = []
prevRoot=''
cant=0

print("leyendo imagenes de ",imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            cant=cant+1
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            #VALIDACIÓN DE 3 CANALES, AGUNAS LAS LEVANTA EN 2, (NULL, 50, 50)
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
            images.append(image)
            b = "Leyendo..." + str(cant)
            print (b, end="\r")
            if prevRoot !=root:
                print(root, cant)
                prevRoot=root
                directories.append(root)
                dircount.append(cant)
                cant=0
dircount.append(cant)

dircount = dircount[1:]
dircount[0]=dircount[0]+1
print('Directorios leidos:',len(directories))
print("Imagenes en cada directorio", dircount)
print('suma Total de imagenes en subdirs:',sum(dircount))


# # Creamos las etiquetas

# In[4]:


labels=[]
indice=0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice=indice+1
print("Cantidad etiquetas creadas: ",len(labels))


# In[5]:


letras=[]
indice=0
for directorio in directories:
    name = directorio.split(os.sep)
    print(indice , name[len(name)-1])
    letras.append(name[len(name)-1])
    indice=indice+1


# In[6]:


y = np.array(labels)
X = np.array(images, dtype=np.uint8) #convierto de lista a numpy


# Find the unique numbers from the train labels
classes = np.unique(y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)


# # Creamos Sets de Entrenamiento y Test

# In[7]:


train_X,test_X,train_Y,test_Y = train_test_split(X,y,test_size=0.2)
print('Training data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)


# In[8]:


plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(train_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(train_Y[0]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_Y[0]))


# # Preprocesamos las imagenes

# In[9]:


train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X/255.
test_X = test_X/255.
plt.imshow(test_X[0,:,:])


# ## Hacemos el One-hot Encoding para la red

# In[10]:


# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Display the change for category label using one-hot encoding
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])


# # Creamos el Set de Entrenamiento y Validación

# In[11]:


#Mezclar todo y crear los grupos de entrenamiento y testing
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)


# In[12]:


print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)


# # Creamos el modelo de CNN

# In[13]:


#declaramos variables con los parámetros de configuración de la red
INIT_LR = 1e-1 # Valor inicial de learning rate. El valor 1e-3 co0rresponde con 0.001\ 0.1
epochs = 10 # Cantidad de iteraciones completas al conjunto de imagenes de entrenamiento
batch_size = 64 # cantidad de imágenes que se toman a la vez en memoria


# In[14]:


letras_model = Sequential()
letras_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(50,50,3)))#CAMBIAR TAMAÑO DE IMAGEN
letras_model.add(LeakyReLU(alpha=0.1))
letras_model.add(MaxPooling2D((2, 2),padding='same'))
letras_model.add(Dropout(0.5))

letras_model.add(Flatten())
letras_model.add(Dense(32, activation='linear'))
letras_model.add(LeakyReLU(alpha=0.1))
letras_model.add(Dropout(0.5))
letras_model.add(Dense(nClasses, activation='softmax'))


# In[15]:


letras_model.summary()


# In[16]:


letras_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adagrad(learning_rate=INIT_LR, decay=INIT_LR / 100),metrics=['accuracy'])


# # Entrenamos el modelo: Aprende a clasificar imágenes

# In[17]:


# este paso puede tomar varios minutos, dependiendo de tu ordenador, cpu y memoria ram libre
letras_train = letras_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))


# In[18]:


# guardamos la red, para reutilizarla en el futuro, sin tener que volver a entrenar
letras_model.save("C:\\Users\\gilba\\Desktop\\IA\\letras94.h5py")


# # Evaluamos la red

# In[19]:


test_eval = letras_model.evaluate(test_X, test_Y_one_hot, verbose=1)


# In[20]:


print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])


# In[21]:


letras_train.history


# In[22]:


accuracy = letras_train.history['accuracy']
val_accuracy = letras_train.history['val_accuracy']
loss = letras_train.history['loss']
val_loss = letras_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[23]:


predicted_classes2 = letras_model.predict(test_X)


# In[24]:


predicted_classes=[]
for predicted_letras in predicted_classes2:
    predicted_classes.append(predicted_letras.tolist().index(max(predicted_letras)))
predicted_classes=np.array(predicted_classes)


# In[25]:


predicted_classes.shape, test_Y.shape


# # Aprendamos de los errores: Qué mejorar

# In[26]:


correct = np.where(predicted_classes==test_Y)[0]
print("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[0:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[correct].reshape(50,50,3), cmap='gray', interpolation='none')#CAMBIAR TAMAÑO DE IMAGEN
    plt.title("{}, {}".format(letras[predicted_classes[correct]],
                                                    letras[test_Y[correct]]))

    plt.tight_layout()


# In[27]:


incorrect = np.where(predicted_classes!=test_Y)[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[0:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[incorrect].reshape(50,50,3), cmap='gray', interpolation='none')
    plt.title("{}, {}".format(letras[predicted_classes[incorrect]],
                                                    letras[test_Y[incorrect]]))
    plt.tight_layout()


# In[28]:


target_names = ["Class {}".format(i) for i in range(nClasses)]
print(classification_report(test_Y, predicted_classes, target_names=target_names))


# In[ ]:




