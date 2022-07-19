#imports and versions
import platform
#print(platform.platform())
import sys
#print(sys.version)
import numpy as np
#print(np.__version__)
import cv2
#print(cv2.__version__)
import pandas as pd
#print(pd.__version__)
import sklearn
#print(sklearn.__version__)
import seaborn as sns
#print(sns.__version__)
from sklearn import metrics, svm, neighbors, ensemble
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
#print(tf.__version__)
from tensorflow import keras
import cf_matrix
import pyzbar
#print(pyzbar.__version__)
import pdf2image
import os


# In[2]:
def find_model(filename):
    result = []

    for root, dir, files in os.walk("."):
        if filename in files:
            result.append(os.path.join(root,filename))
    return result


# In[3]:


def load_training_data():
    #ler dados treino e teste
    #pd.read_csv tem como separador default a virgula
    train_data = pd.read_csv("./input/assinaturas_treino.csv")

    X_train = np.array(train_data.iloc[:,1:])
    y_train = np.array(train_data.iloc[:,0])


    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    return X_train, y_train, scaler


def classifier_training():
    X_train, y_train, scaler = load_training_data()

    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(activation="relu", alpha=0.05, hidden_layer_sizes=[100], learning_rate="invscaling")
    mlp.fit(X_train, y_train)
    return mlp


# In[15]:

def testa_classificador():
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


    print(X_test.shape)



    # In[26]:



    ## Standardização

    # In[27]:


    from sklearn.preprocessing import StandardScaler
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)

    #verifica se for standardizado
    print (X_test)


    # # ## MLP Classifier

    # # In[28]:


    # from sklearn.neural_network import MLPClassifier
    # mlp = MLPClassifier(hidden_layer_sizes=[100])
    # mlp.fit(X_train, y_train)
    # y_pred = mlp.predict(X_test)


    # # In[29]:


    # print("MLP")
    # print(metrics.classification_report(y_test, y_pred))
    # accuracy = metrics.accuracy_score(y_test, y_pred)
    # average_accuracy = np.mean(y_test == y_pred) * 100
    # print("The average accuracy is {0:.3f}%".format(average_accuracy))


    # # In[30]:


    # print("Confusion Matrix:")
    # cf_mat = confusion_matrix(y_test, y_pred)
    # print(cf_mat)
    # print(f"\n")

    # #sns.heatmap(cf_mat/np.sum(cf_mat), annot=True, fmt='.2%', cmap="Blues")
    # cf_matrix.make_confusion_matrix(cf_mat, group_names=labels, categories=categories, cmap="Blues")



    # # In[26]:


    # model = keras.Sequential([
    #     keras.layers.Dense(100, activation='relu'),
    #     keras.layers.Dense(10, activation='softmax')
    # ])


    # # In[27]:


    # model.compile(optimizer='adam',
    #              loss='sparse_categorical_crossentropy',
    #              metrics=['accuracy'])




    # # In[28]:


    # model.fit(X_train, y_train, epochs=18)


    # # In[29]:


    # test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)


    # # In[30]:


    # print(f"\nTest accuracy: {test_acc:.2%}\n")


    # # ## SVM Classifier

    # # In[55]:


    # svc = svm.SVC(C=15, kernel='rbf', gamma=0.01)
    # svc.fit(X_train, y_train)
    # y_pred = svc.predict(X_test)


    # # In[56]:


    # print('Support Vector Machines - SVM')
    # print(metrics.classification_report(y_test, y_pred))
    # accuracy = metrics.accuracy_score(y_test, y_pred)
    # average_accuracy = np.mean(y_test == y_pred) * 100
    # print('The average accuracy is {0:.1f}%'.format(average_accuracy))


    # # In[57]:


    # print("Confusion Matrix:")
    # cf_mat_svm = confusion_matrix(y_test, y_pred)
    # print(cf_mat_svm)
    # print(f"\n")
    # cf_matrix.make_confusion_matrix(cf_mat_svm, group_names=labels, categories=categories, cmap="Blues")



    # # In[58]:
    # from joblib import dump
    # dump(mlp, "modelsignature.joblib")
