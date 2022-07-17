import pdfparajpeg
#imports and versions
import platform
print(platform.platform())
import sys
print(sys.version)
import numpy as np
print(np.__version__)
import cv2
print(cv2.__version__)
import pandas as pd
print(pd.__version__)


# ### Verificar tamanho das imagens

# ## Função para criar o header da tabela

# In[3]:


# A tabela vai ter 246 elementos
# 0      --> label
# 1-15   --> soma das linhas
# 30-245 --> soma das colunas

# Funcao que cria um header para o dataset 
# formato: [label, sumrow[1...30], sumcol[1...215]]
# argumento = label
def criaHeader():
    z=["label"]
    for r in range(1,31):
        z+=["sumrow"+str(r)]
    for c in range(1,216):
        z+=["sumcol"+str(c)]
    return(z)


# In[ ]:





# ## Transformar as imagens - resize, blur, thresh

# In[10]:


#vai ao directorio buscar o filepath das imagens 0 e 1
imgarray0 = pdfparajpeg.find_jpg("./assinaturas/0_org")
imgarray1 = pdfparajpeg.find_jpg("./assinaturas/1_org")
#abre uma lista para reservar as imagens carregadas
sigarray0 = []
sigarray1 = []

# vai percorrer o array dos endereços das imagens NAO ASSINADAS
# e transformar para uma pasta onde se encontram as mesmas imagens processadas
i=1
for imgs in imgarray0:
    img = cv2.imread(imgs) # ler a imagem
    #dar tamanho fixo
    img_sized = cv2.resize(img, (215,30))
    #converter para um canal BW
    gray = cv2.cvtColor(img_sized, cv2.COLOR_BGR2GRAY)
    #blur - reduzir ruído e contornos
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    #truncar pixels abaixo de valor - redução de ruido
    ret, thresh = cv2.threshold(blur, 240, 255, cv2.THRESH_TRUNC)
    # adiciona ao array de imagens
    sigarray0.append(thresh)
    
    #escreve imagem para pasta TESTE TESTE
    cv2.imwrite("./assinaturas/0_pro/teste"+str(i)+".jpg", thresh)
    i+=1
    
# vai percorrer o array dos endereços das imagens ASSINADAS
# e transformar para uma pasta onde se encontram as mesmas imagens processadas
i=1
for imgs in imgarray1:
    img = cv2.imread(imgs) # ler a imagem
    #dar tamanho fixo
    img_sized = cv2.resize(img, (215,30))
    #converter para um canal BW
    gray = cv2.cvtColor(img_sized, cv2.COLOR_BGR2GRAY)
    #blur - reduzir ruído e contornos
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    #truncar pixels abaixo de valor - redução de ruido
    ret, thresh = cv2.threshold(blur, 240, 255, cv2.THRESH_TRUNC)
    # adiciona ao array de imagens
    sigarray1.append(thresh)
    
    #escreve imagem para pasta TESTE TESTE
    cv2.imwrite("./assinaturas/1_pro/teste"+str(i)+".jpg", thresh)
    i+=1


# In[11]:


# verifica arrays
print("ARRAY DE 0:")
print(len(sigarray0))
print("ARRAY DE 1:")
print(len(sigarray1))


# ## Criar csv treino e teste

# In[12]:


label = 0

#soma as linhas da imagem
sum_of_rows = np.sum(sigarray0[9], axis = 1)
#soma as colunas da imagem
sum_of_columns = np.sum(sigarray0[9], axis = 0)

#DEBUG - verifica
print(sum_of_rows)
print(sum_of_columns)

# concatena os vectores soma_linha + soma_coluna
vector_img = np.concatenate((sum_of_rows, sum_of_columns))

# adiciona a label no inicio do vector
vector_img = np.insert(vector_img, 0, label)

vector_img = np.transpose(vector_img)

#DEBUG - verifica
print(vector_img)


# ### Array de imagens com label 0 - não assinadas

# In[13]:


#define a label a atribuir as imagens
label = 0
dataset_array0 = np.zeros(246)
# vai percorrer o array das imagens já transformadas    
for iter in range(len(sigarray0)):
    #soma as linhas da imagem
    sum_of_rows = np.sum(sigarray0[iter], axis = 1)
    #soma as colunas da imagem
    sum_of_columns = np.sum(sigarray0[iter], axis = 0)


    # concatena os vectores soma_linha + soma_coluna
    vector_img = np.concatenate((sum_of_rows, sum_of_columns))

    # adiciona a label no inicio do vector
    vector_img = np.insert(vector_img, 0, label)

    vector_img = np.transpose(vector_img)
    
    dataset_array0 = np.vstack((dataset_array0, vector_img))

dataset_array0 = np.delete(dataset_array0, 0, 0)


# In[14]:


# verificar se a forma esta correcta - senao tranpose()
print(dataset_array0[:10])


# In[15]:


print(len(dataset_array0))


# ### Array de imagens com label 1 - assinadas

# In[16]:


#define a label a atribuir as imagens
label = 1
dataset_array1 = np.zeros(246)
# vai percorrer o array das imagens já transformadas    
for iter in range(len(sigarray1)):
    #soma as linhas da imagem
    sum_of_rows = np.sum(sigarray1[iter], axis = 1)
    #soma as colunas da imagem
    sum_of_columns = np.sum(sigarray1[iter], axis = 0)


    # concatena os vectores soma_linha + soma_coluna
    vector_img = np.concatenate((sum_of_rows, sum_of_columns))

    # adiciona a label no inicio do vector
    vector_img = np.insert(vector_img, 0, label)

    vector_img = np.transpose(vector_img)
    
    dataset_array1 = np.vstack((dataset_array1, vector_img))
    
dataset_array1 = np.delete(dataset_array1, 0, 0)


# In[17]:


# verificar se a forma esta correcta
print(dataset_array1[:10])


# In[18]:


print(len(dataset_array1))


# ## Escolher aleatoriamente para treino e teste

# #### Teste (30%) = 1400 + 1400 = 2800
# #### Treino (70%) = 3269 + 3269 = 6538

# In[40]:


iteracao = np.arange(6538)
np.random.shuffle(iteracao)


# In[41]:


print(iteracao)


# ## Converter para csv
# 

# In[19]:


# passar para pandas DataFrame
df0=pd.DataFrame(dataset_array0)
df1=pd.DataFrame(dataset_array1)


# In[20]:


# ordem aleatoria
df0 = df0.sample(frac = 1)
df1 = df1.sample(frac = 1)


# In[21]:


df0_test = df0.iloc[0:2000]
df1_test = df1.iloc[0:2000]
#df0_train = df0.iloc[1400:]
#df1_train = df1.iloc[1400:4669]


# In[23]:


len(df1_test)


# In[24]:


dataset_array_teste = pd.concat([df0_test, df1_test])
#dataset_array_treino = pd.concat([df0_train, df1_train])


# In[26]:


len(dataset_array_teste)


# In[27]:


# colocar index pela ordem correcta
dataset_array_teste.head()


# In[28]:


# Header da tabela
dataset_array_teste = dataset_array_teste.set_axis(criaHeader(), axis = 1, inplace = False)
#dataset_array_treino = dataset_array_treino.set_axis(criaHeader(), axis = 1, inplace = False)


# In[29]:


dataset_array_teste = dataset_array_teste.astype(int)
#dataset_array_treino = dataset_array_treino.astype(int)
dataset_array_teste.head()


# In[31]:


#escrever a matrix para um ficheiro csv
#pd.DataFrame(dataset_array_treino).to_csv("./input/assinaturas_treino.csv", index=False)
pd.DataFrame(dataset_array_teste).to_csv("./input/novoteste.csv", index=False)


