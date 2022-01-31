#!/usr/bin/env python
# coding: utf-8

# # Reconhecer assinatura ou rasura

# In[1]:


#imports and versions
import platform
print(platform.platform())
import sys
print(sys.version)
import numpy as np
print(np.__version__)
import matplotlib
print(matplotlib.__version__)
from matplotlib import pyplot as plt
import cv2
print(cv2.__version__)
import pandas as pd
print(pd.__version__)
import sklearn
print(sklearn.__version__)
import seaborn as sns
print(sns.__version__)
from sklearn import metrics, svm, neighbors, ensemble
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
import cf_matrix


# In[2]:


import pyzbar
print(pyzbar.__version__)
import pdf2image
print(pdf2image.__version__)


# In[15]:


#ler dados treino e teste
#pd.read_csv tem como separador default a virgula
train_data = pd.read_csv("./input/assinaturas_treino.csv")
test_data = pd.read_csv("./input/assinaturas_teste.csv")
#test_data = pd.read_csv("./input/novoteste.csv")


# In[16]:


#verificar se os dados treino foram lidos correctamente
train_data.head()


# In[17]:


#verificar se os dados teste foram lidos correctamente
test_data.tail()


# ## Verificar os imports e os dados

# In[23]:


labels = ["True Neg","False Pos","False Neg","True Pos"]
categories = ["Unsigned", "Signed"]


# In[24]:


X_train = np.array(train_data.iloc[:,1:])
y_train = np.array(train_data.iloc[:,0])
X_test = np.array(test_data.iloc[:,1:])
y_test = np.array(test_data.iloc[:,0])
train_data.shape


# In[25]:


test_data.shape


# In[26]:


y_train


# ## Standardização

# In[27]:


from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

#verifica se for standardizado
print (X_train)


# ## MLP Classifier

# In[28]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=[100])
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)


# In[29]:


print("MLP")
print(metrics.classification_report(y_test, y_pred))
accuracy = metrics.accuracy_score(y_test, y_pred)
average_accuracy = np.mean(y_test == y_pred) * 100
print("The average accuracy is {0:.3f}%".format(average_accuracy))


# In[30]:


print("Confusion Matrix:")
cf_mat = confusion_matrix(y_test, y_pred)
print(cf_mat)
print(f"\n")

#sns.heatmap(cf_mat/np.sum(cf_mat), annot=True, fmt='.2%', cmap="Blues")
cf_matrix.make_confusion_matrix(cf_mat, group_names=labels, categories=categories, cmap="Blues")


# In[ ]:





# In[26]:


model = keras.Sequential([
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


# In[27]:


model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


# In[ ]:





# In[28]:


model.fit(X_train, y_train, epochs=18)


# In[29]:


test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)


# In[30]:


print(f"\nTest accuracy: {test_acc:.2%}\n")


# ## SVM Classifier

# In[55]:


svc = svm.SVC(C=15, kernel='rbf', gamma=0.01)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)


# In[56]:


print('Support Vector Machines - SVM')
print(metrics.classification_report(y_test, y_pred))
accuracy = metrics.accuracy_score(y_test, y_pred)
average_accuracy = np.mean(y_test == y_pred) * 100
print('The average accuracy is {0:.1f}%'.format(average_accuracy))


# In[57]:


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\n")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




