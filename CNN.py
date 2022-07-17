import numpy as np
import os
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
#os.add_dll_directory("C:/Users/japl0/Desktop/cuda/bin")

import tensorflow as tf
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import seaborn as sb
import pandas as pd
import PIL
import random
#import keras_tuner as kt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation
from keras.preprocessing import image
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from random import uniform
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.regularizers import l2
import keras.backend as K



def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


fix_gpu()

#215 : 90
IMG_WIDTH = 215
IMG_HEIGHT = 90
USERS = 50
BATCH_SIZE = 2

tensorboard_callbacks = TensorBoard(log_dir='logs/')


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


# Cria o dataset atraves de uma diretoria
# com os parametros indicados
def cria_dataset(DATADIR):

    ds_train = tf.keras.preprocessing.image_dataset_from_directory (
        DATADIR,
        labels='inferred',
        label_mode="int",
        #class_names=[]
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        image_size=(IMG_WIDTH, IMG_HEIGHT),
        shuffle=True,
        seed=123,
        validation_split=0.01,
        subset="training",
    )

    ds_validation = tf.keras.preprocessing.image_dataset_from_directory (
        DATADIR,
        labels='inferred',
        label_mode="int",
        #class_names=[]
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        image_size=(IMG_WIDTH, IMG_HEIGHT),
        shuffle=True,
        seed=123,
        validation_split=0.01,
        subset="validation",
    )

    return ds_train, ds_validation

#Treina e guarda o modelo e exporta com o respetivo nome
def treinar_modelo(ds_train, ds_validation, batch):

    s = 1

    weight_decay = 0.005

    #Arquitetura da rede CNN

    model = models.Sequential([
        layers.Input((IMG_WIDTH, IMG_HEIGHT, 1)),
        keras.layers.Conv2D(filters=96*s, kernel_size=(11,11),padding="same", strides=(4,4), activation='relu', kernel_regularizer=l2(weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=256*s, kernel_size=(5,5), strides=(1,1), padding="same", activation='relu', kernel_regularizer=l2(weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=384*s, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', kernel_regularizer=l2(weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384*s, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', kernel_regularizer=l2(weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256*s, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', kernel_regularizer=l2(weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, kernel_regularizer=l2(weight_decay)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, kernel_regularizer=l2(weight_decay)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(USERS, activation='softmax')

    ])

    tensorflow_callback = keras.callbacks.TensorBoard(
        log_dir="tb_callback_dir", histogram_freq=1,
    )


    # A schedule e responsavel por diminuir o learning rate durante o treino

    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=2500,
        decay_rate=0.9,
        staircase=True)

    model.compile(
                    loss='sparse_categorical_crossentropy',
                    optimizer=keras.optimizers.SGD(lr_schedule),
                    metrics=['accuracy'])
    #model.summary()


    # Treinar o modelo com os dados passados para a funcao

    history = model.fit(
        ds_train,
        validation_data = ds_validation,
        epochs=25,
        callbacks=[tensorboard_callbacks],
    )

    #print(history.history['val_loss'])

    #plt.plot(history.history['val_loss'], label='validation loss')
    #plt.plot(history.history['loss'], label='training loss')
    #plt.show()

    #results = model.evaluate(ds_validation)
    #print("validation : test loss, test acc:", results)

    #results = model.evaluate(ds_train)
    #print("training : test loss, test acc:", results)

    return model


# Exportar o modelo para o formato .h5

def exportar_modelo(model, nome):
    model.save(nome + '.h5')

# Importar o modelo no formato .h5

def importar_modelo(nome):
    return keras.models.load_model(nome)

# Funcao que retorna a previsão do modelo que é um array com as probabilidades de cada classe

def previsao_modelo(model, DATADIR):
    img = image.load_img(DATADIR, target_size = (IMG_WIDTH, IMG_HEIGHT), color_mode='grayscale')
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis = 0)
    return model.predict(img)

# De momento nao esta a ser usado mas pode ser util para limpar ruido das imagens

def threshold_dataset(DATADIR, clean):
    count = 0
    for alunos in os.listdir(DATADIR):
        os.mkdir(clean + "/" + alunos)
        for assinaturas in os.listdir(DATADIR + "/" + alunos):
            img = cv2.imread(DATADIR + "/" + alunos + "/" + assinaturas, 0)
            th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,10)
            th = Image.fromarray(th).save(clean + "/" + alunos + "/" + str(count) + '.jpg')
            count+=1


# Passo fundamental para que cada pixel tenha um valor entre [0, 1]

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


# Atualmente nao e utilizada esta funcao, no entanto pode ser util para remover faltas de um dataset
# para isso e necessario ter um modelo que reconheca as faltas

def remove_faltas_dataset(ds_dir, temp_dir):
    count = 0
    DATADIR = ds_dir
    modelo = importar_modelo('alex-net-faltas.h5')
    print(modelo.summary())
    for alunos in os.listdir(DATADIR):
        for assinaturas in os.listdir(DATADIR + "/" + alunos):
            img = cv2.imread(DATADIR + "/" + alunos + "/" + assinaturas, 0)
            previsao = np.argmax(previsao_modelo(modelo, DATADIR + "/" + alunos + "/" + assinaturas))
            if (previsao == 0):
                im1 = Image.open(DATADIR + "/" + alunos + "/" + assinaturas)
                im1 = im1.save(temp_dir + "/" + str(count) + ".jpg")
                print(previsao_modelo(modelo, DATADIR + "/" + alunos + "/" + assinaturas))
                os.remove(DATADIR + "/" + alunos + "/" + assinaturas)
                count+=1
            #else:
                #im1 = Image.open(DATADIR + "/" + alunos + "/" + assinaturas)
                #im1 = im1.save(temp_dir + "/" + str(count) + ".jpg")
                #count+=1


# Funcao utilizada para criar as pastas e colocar as imagens

def constroi_dataset(ds_dir, save_dir, num_alunos, num_assinaturas):
    nAlunos = 0
    if(os.path.isdir(save_dir) == False):
        os.mkdir(save_dir)
    for alunos in os.listdir(ds_dir):
        count = 0
        nAlunos+=1
        if(nAlunos>num_alunos):
            break
        if(os.path.isdir(save_dir + "/" + alunos) == False):
            os.mkdir(save_dir + "/" + alunos)
        for assinaturas in os.listdir(ds_dir + "/" + alunos):
            im1 = Image.open(ds_dir + "/" + alunos + "/" + assinaturas)
            im1 = im1.save(save_dir + "/" + alunos + "/" +  str(count) + ".jpg")
            count+=1
            if(count==num_assinaturas):
                break

# Funcao de data augmentation utilizada para melhorar os resultados do modelo

def expande_dataset(datadir, num):
    for alunos in os.listdir(datadir):
        #numFiles = len([entry for entry in os.listdir(datadir + "/" + alunos)])
        numFiles = 0
        for x in range(num):
            file = random.choice(os.listdir(datadir + "/" + alunos + "/"))
            im1 = Image.open(datadir + "/" + alunos + "/" + file).convert("L")
            im1 = im1.resize((IMG_WIDTH,IMG_HEIGHT))
            vetor_translacao = (uniform(-15, 15), uniform(-3, 3))
            angulo = uniform(-1, 1) / 10

            #print("Angulo : " + str(angulo))
            #print("Translacao : " + str(vetor_translacao))

            #print()

            im1 = image.img_to_array(im1)
            im1 = np.expand_dims(im1, axis = 0)
            #image2 = tf.image.random_brightness(im1, max_delta=0.1)
            #image2 = tf.image.random_contrast(image2, lower=0.1, upper=0.2)
            image2 = tfa.image.transform_ops.rotate(im1, angulo)
            image2 = tfa.image.translate(image2, vetor_translacao)
            image2 = np.reshape(image2, (IMG_HEIGHT,IMG_WIDTH))
            #print(datadir + "/" + alunos + "/"+ str(numFiles) + '.jpg')
            cv2.imwrite(datadir + "_augmented/" + alunos + "/"+ str(numFiles) + '.jpg',image2)
            #print(datadir + "/" + alunos + "/" +  str(numFiles) + '.jpg')
            numFiles+=1
            #imgplot = plt.imshow(image2, interpolation="nearest")
            #plt.show()

# Funcao utilizada para estudar os dados do modelo como por exemplo as matrizes de confusao

def compute_metrics(ds_dir, model):
    current_class = 0
    tp = 0
    fp = 0
    fn = 0
    dim = 50
    confusion_matrix = [[0 for x in range(dim)] for y in range(dim)]
    for alunos in os.listdir(ds_dir):
        print('A calcular matriz de confusao : ' + str(current_class*2) + '%')
        for assinatura in os.listdir(ds_dir + '/' + alunos):
            previsao = np.argmax(previsao_modelo(modelo, ds_dir + '/' + alunos + '/' + assinatura))
            confusion_matrix[previsao][current_class]+=1

        current_class+=1

    df_cm = pd.DataFrame(confusion_matrix, range(dim), range(dim))
    sb.set(font_scale=1.4) # for label size
    sb.heatmap(df_cm, cmap='viridis') # font size
    plt.show()
    # i = prediction, j = actual
    m1 = 0
    m2 = 0
    for i in range(dim):
        tp = 0
        fp = 0
        fn = 0
        tp += confusion_matrix[i][i]
        for j in range(dim):
            if(i != j):
                fp += confusion_matrix[i][j]
        for j in range(dim):
            if(i != j):
                fn += confusion_matrix[j][i]
        m1 += float(tp / (tp + fn))
        m2 += float(tp / (tp + fp))
    print("precison mean : " + str(m2 / 50))
    print("recall mean : " + str(m1 / 50))
    return confusion_matrix, tp, fp, fn


# Funcao usada para calcular o batch size dependendo do training set

def compute_batch_size(DATADIR):
    count = 0
    for path in os.listdir(DATADIR):
        for file in os.listdir(DATADIR + '/' + path):
            if os.path.isfile(DATADIR + '/' + path + '/' + file):
                count += 1

    print("Total de : " + str(count))
    BATCH_SIZE = int((count * 0.7) / 450) + 1
    if(BATCH_SIZE < 2):
        BATCH_SIZE = 2
    print("Batch size : " + str(BATCH_SIZE))
    return BATCH_SIZE


# Funcao usada para detetar as possiveis falsificacoes em assinaturas

def matrizIncerteza(model, turma):
    matriz_incerteza = []
    confianca = 0.6
    for alunos in os.listdir("./turmas/" + str(turma)):
        assinaturas_incertas = 0
        for assinatura in os.listdir("./turmas/" + turma + "/" + alunos):
            r = previsao_modelo(model, "./turmas/" + turma + "/" + alunos + "/" + assinatura)
            previsao = np.argmax(r)
            if(r[0][previsao] < confianca):
                assinaturas_incertas += 1
        matriz_incerteza.append((alunos, assinaturas_incertas))
    return matriz_incerteza




'''
DATADIR = 'C:/Users/japl0/Desktop/CNN/datasets_caso_uso/10Genuinas/DATASET-50AL-170AS'

BATCH_SIZE = compute_batch_size(DATADIR)

ds_train, ds_validation = cria_dataset(DATADIR)

ds_train = ds_train.map(normalize_img)

ds_validation = ds_validation.map(normalize_img)

#ds_train = ds_train.map(augment)


#expande_dataset('C:/Users/japl0/Desktop/CNN/datasets_caso_uso/10Genuinas/DATASET-50AL-170AS', 160)
#constroi_dataset('C:/Users/japl0/Desktop/CNN/assinaturas', 'C:/Users/japl0/Desktop/CNN/datasets_caso_uso/10Genuinas/DATASET-50AL-10AS', 50, 30)


modelo = treinar_modelo(ds_train, ds_validation)

confusion_matrix, tp, fp, fn = compute_metrics('C:/Users/japl0/Desktop/CNN/datasets_caso_uso/DATASET-50AL-VALIDACAO2', modelo)


#threshold_dataset('C:/Users/japl0/Desktop/CNN/datasets_caso_uso/DATASET-50AL-VALIDACAO', 'C:/Users/japl0/Desktop/CNN/datasets_caso_uso/DATASET-50AL-VALIDACAO-TH')


print()
print('----- Set de validacao independente -----')

DATADIR = 'C:/Users/japl0/Desktop/CNN/datasets_caso_uso/DATASET-50AL-VALIDACAO2'

ds_train, ds_validation = cria_dataset(DATADIR)

ds_train = ds_train.map(normalize_img)

ds_validation = ds_validation.map(normalize_img)

results = modelo.evaluate(ds_validation)
print("validation : test loss, test acc:", results)

results = modelo.evaluate(ds_train)
print("validation : test loss, test acc:", results)

#print("recall : " + str(float(tp / (tp + fn))))
#print("precision : " + str(float(tp / (tp + fp))))


'''

#exportar_modelo(modelo, 'alex-net-faltas')

'''
while True:
    print('Diretoria :')
    dir = input()
    print(np.argmax(previsao_modelo(modelo, dir)))
'''

#exportar_modelo(modelo, 'CNN-ACC0-X')
