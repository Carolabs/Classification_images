# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Trabajo Aprendizaje automático I
# ### <font color=green>*Máster en Informática Industrial y Robótica*</font>
# 
# #### Nombre y apellidos:
# #### Domingo Capelo Luces
# #### Iraisy Carolina Figueroa Silva

# %%


# %% [markdown]
# #### En este trabajo vamos a estudiar...
# %% [markdown]
# ### PASO 1 :
# ### TRATAMIENTO DE DATOS: 
# #### Partiendo de nuestro grupo de imagenes .jpg tenemos que generar el grupo de datos de entrada y salida de nuestro modelo.
# #### Para poder tratar las images usaremos las siguientes librerias:
# ### - LIBRERÍA CV2
# #### Nos permite convertir jpg ---> Array Bytes
# #### Syntax: cv2.imread(path, flag)
# #### path: ruta con formato "string" en la que se encuentra la imagen a leer.
# #### flag: especifica el formato de color en el que se leerá la imagen.
# #### flag=0 blanco y negro
# #### flag=1 color
# ### - LIBRERÍA fnmatch y os
# #### Nos permiten buscar archivos dentro de una carpeta mediante su ruta (path).
# #### Teniendo en cuenta esto, para encontrar un archivo lo único que necesitaremos es su ruta.
# %% [markdown]
# ## Importamos librerías
# #### Si falta alguna por instalar:
# #### cv2 ----> python -m pip install opencv-python

# %%
import cv2
import fnmatch
import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score, plot_confusion_matrix, classification_report
import sklearn.neural_network
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import datasets, layers, models
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from pandas import ExcelWriter
import xlsxwriter
print('Done')

# %% [markdown]
# ## Funcion: List_name_files(path, l)
# #### Nos devuelve una lista con los nombres de los archivos que se encuentran en la carpeta.

# %%
def List_name_files(path, l):
    try:
        l = []
        for item in os.listdir(path):
            l.append(item)
    except:
        print("Error in funtion : list_name_files(path, l)")
        l = []
    return l
print('Done')

# %% [markdown]
# #### Ejemplo de uso:
# #### Guardamos los nombres de todos los archivos en nuetra carpeta de Data Images

# %%
# Ruta
PathData="dataset/Data Images"
# Lista vacias
L_nameDataImag = []
L_nameDataImag = List_name_files(PathData, L_nameDataImag)
for i in range(5): #Ejemplo imprimiendo 5 nombres Train
    print("Imagen "+str(i)+": "+L_nameDataImag[i])
print('Done')

# %% [markdown]
# ## Funcion: List_Array_RGB(path, l)
# #### Devuelve una lista que representa el array de píxeles de cada imagen.

# %%
def List_Array_RGB(path, l):
    try:
        l = []
        for item in os.listdir(path):
            p = path + "/" + str(item)
            l.append(cv2.imread(p, flags=1))
    except:
        print("Error in funtion : list_name_files(path, l)")
        l = []
    return l
print('Done')

# %% [markdown]
# #### Ejemplo de uso:
# #### Guardamos en la lista L_RGBData, las matrices de pixeles de cada imagen. 

# %%
#Lista vacia
L_RGBData = []
L_RGBData = List_Array_RGB(PathData, L_RGBData)
for i in range(5):  #Ejemplo imprimiendo 5 Arrays
    print(str(L_nameDataImag[i])+"--> pixels:"+str(L_RGBData[i].shape))

# %% [markdown]
# ## Redimensión 64x64
# #### Como vemos anteriormente las imagenes no tienen el mismo tamaño, por lo que para trabajar con ellas las redimensionaremos todas con el mismo tamaño, en este caso elegimos redimensionarlas como : 64x64
# %% [markdown]
# ## Funcion: Redim(pathImage,pathSave,listnames,D1,D2):
# #### Redimensiona las imagenes que se encuentren en una carpeta determina y las guarda en otra ruta con el mismo nombre.
# #### pathImage: Ruta en la que se encuentran las imagenes sin redimensionar.
# #### pathSave: Ruta en la que queremos guardar las imagenes redimensionadas.
# #### D1: Dimensión 1 en pixeles
# #### D2: Dimensión 2 en pixeles

# %%
def Redim(pathImage,pathSave,D1,D2):
    try:
        for item in os.listdir(pathImage):
            path = pathImage + "/" + str(item)
            img = Image.open(path)
            new_imag = img.resize((D1, D2))
            pathsave_=pathSave + "/" + str(item)
            new_imag.save(pathsave_)
    except:
        print("Error in funtion : Redim(pathImage,pathSave,listnames,D1,D2):")
print('Done')

# %% [markdown]
# #### Ejemplo de uso:
# #### Redimensionar las imágenes y las guardamos en la carpeta Data_Images64x64

# %%
PathData_64x64="dataset/Data_Images64x64"
Redim(PathData,PathData_64x64,64,64)
print('Done')

# %% [markdown]
# #### Ahora que tenemos las imagenes redimensionadas, con el uso de la función explicada anteriormente "List_Array_RGB", guardaremos la lista de imágenes redimesionadas.

# %%
L_RGBData_64x64 = []
L_RGBData_64x64 = List_Array_RGB(PathData_64x64, L_RGBData_64x64)
for i in range(5):  #Ejemplo imprimiendo 5 Arrays
    print(str(L_nameDataImag[i])+"--> pixels redim: "+str(L_RGBData_64x64[i].shape))

# %% [markdown]
# ## Función: Plot_Imagenes(StrTitulo,ListaTitulo,ListaImg,NumImg):
# #### Grafica n imágenes con sus respectivos títulos
# #### StrTitulo: String que se quiera poner poner como título genérico
# #### ListaTitulo: Si tenemos una Lista con nombres.
# #### ListaImg: Lista de imágenes a imprimir
# #### NumImg : Número de gráficas a mostrar

# %%
def Plot_Imagenes(StrTitulo,ListaTitulo,ListaImg,NumImg):
    _, axes = plt.subplots(nrows=1, ncols=NumImg, figsize=(12, 8))  # Crea una figura con 1x5 subplots
    i=0
    for ax in axes: # Imprimos las 5 primeras imagen
        ax.set_axis_off()
        #Titulo L_nameTrainImag
        titulo=str(StrTitulo)+str(ListaTitulo[i])
        ax.set_title(titulo)
        #Array imagen L_RGBTrain
        ax.imshow(ListaImg[i],cmap=plt.cm.gray_r)
        i+=1
print('Done')

# %% [markdown]
# #### Ejemplo de uso:
# #### Comprobamos las diferencias entre la imagen original y la redimensionada.

# %%
Plot_Imagenes("Calidad 100% \n",L_nameDataImag,L_RGBData,5)
Plot_Imagenes("Redim 64x64 \n",L_nameDataImag,L_RGBData_64x64,5)
print('Done')

# %% [markdown]
# #### Cargamos los datos de entrenamiento .csv, mediante el uso de la librería pandas.
# #### Representamos los datos en forma de tabla "Nombre imagen" y "clase a la que pertenece"
# %% [markdown]
# ## Función: CSV_to_RandomTable(path)
# #### Devuelve una tabla con los datos ordenados de forma aleatoria
# #### path: ruta del archivo .csv

# %%
def CSV_to_RandomTable(path):
    try:
        Datos= pd.read_csv(path)
        #Vamos a mezclar los datos para que no tengan ningun orden de prioridad
        n_filas, m_colum=Datos.shape
        Datos=Datos.sample(n=n_filas, random_state=1) #La opción sample permite crear una tabla aleatoria
        return Datos
    except:
        print("Error in funtion : CSV_to_RadomTable(path)")
print('Done')

# %% [markdown]
# #### Ejemplo de uso:
# #### Cargamos los datos .csv del conjunto de datos, se muestra "nombre imagen" "clase"

# %%
# Importar datos
Path_CsvData='dataset/Data.csv'
Datos = CSV_to_RandomTable(Path_CsvData)

# Barajar (opcional)
#Datos = sk.utils.shuffle(Datos)

print(Datos)

# %% [markdown]
# ## Asociación Entradas Salidas
# #### Como vemos en la tabla anterior se asocia el nombre de una imagen con una clase.
# #### A nosotros nos interesa que asocie la matriz de píxeles de cada imagen con su clase.
# #### Para ello lo primero que haremos sera Guardar los Nombres y las Clases de la tabla anterior en dos listas independientes.

# %%
#Nombres Tabla
Data_NameImg=Datos.iloc[:,:-1].values
print("ORDEN DE NOMBRES\n")
print("Tabla.csv")
for i in range(5): #Ejemplo imprimiendo 5 nombres
    print("Indice "+ str(i)+": "+Data_NameImg[i])
print("\nLista Vectores de pixeles")
for i in range(5): #Ejemplo imprimiendo 5 nombres
    print("Indice "+ str(i)+": "+L_nameDataImag[i])

#Clases Tabla
Data_ClaseImg=Datos.iloc[:,1:].values

# %% [markdown]
# #### Como se puede observar en el print anterior, el orden de la tabla .csv no concuerda con el orden de nuestra lista de imagenes leídas.
# #### Por lo que no podemos hacer una asociación directa entre una matriz de píxeles (64,64,3)---->"clase".
# #### Tendremos que ordenar la lista primero, para ello la ordenaremos con el orden de la tabla .csv de forma que podamos asociar (64,64,3)---->"clase", esta nueva lista será la lista L_DataOrdenada
# %% [markdown]
# ## Función: Ordenar_Datos(nombreTabla,ListaNombres,ListaRGB):
# #### Devuelve una lista con el orden de nombres especificados en una tabla.csv
# #### nombreTabla= nombre de la tabla que condicionará el orden de la lista.
# #### ListaNombres= nombres de la lista desordenada.
# #### ListaRGB= Lista que se desea ordenar.

# %%
def Ordenar_Datos(nombreTabla,ListaNombres,ListaRGB):
    try:
        l=[]
        for i in range(len(nombreTabla)):#Recorro lista de nombres tabla
            if nombreTabla[i] in ListaNombres :#Si existe el nombre de la tabla dentro de nuestra lista de nombres
                indice=ListaNombres.index(nombreTabla[i])#Guardo el indice de ese nombre de nuestra Lista L_nameTrainImag
                l.append(ListaRGB[indice])#Con este indice sabemos el valor del vector respecto a la tabla
    except:
        print("Error in funtion : Ordenar_Datos(nombreTabla,ListaNombres,ListaRGB)")
    return l


# %%
L_DataOrdenada=[]
L_DataOrdenada=Ordenar_Datos(Data_NameImg,L_nameDataImag,L_RGBData_64x64)

# %% [markdown]
# #### Comprobamos que están bien asociadas nuestras entradas salidas
# #### Para ello imprimos las imágenes 5 primeras imagenes que muestra nuestra tabla .csv "hardcodeando ruta" y las comparamos con las 5 primeras imágenes con el nuevo orden.

# %%
print("Image"+"           "+"Class")
for i in range (10):
    print(str(L_DataOrdenada[i].shape)+"     " +str(Data_ClaseImg[i]))
img0 = cv2.imread("dataset/Data Images/image4707.jpg", flags=1)
img1 = cv2.imread("dataset/Data Images/image9827.jpg", flags=1)
img2 = cv2.imread("dataset/Data Images/image8322.jpg", flags=1)
img3 = cv2.imread("dataset/Data Images/image8426.jpg", flags=1)
img4 = cv2.imread("dataset/Data Images/image2346.jpg", flags=1)
L_Hardcodeado=[img0,img1,img2,img3,img4]
Plot_Imagenes("Hardcodeado \n Clase: ",Data_ClaseImg,L_Hardcodeado,5)
Plot_Imagenes("Automático \n Clase: ",Data_ClaseImg,L_DataOrdenada,5)

# %% [markdown]
# ## ENTRADAS SALIDAS
# #### Como observamos nuestras imágenes ya se encuentran ordenadas según su clase.
# #### Procedemos separar las entradas y salidas.

# %%
X=[]#Entradas
t=[]#Salidas
X=L_DataOrdenada
t=Data_ClaseImg
t # Es un array de strings


# %%
# Las salidas en t, deben transformarse de string a un valor numérico para facilitarle la tarea a la regresión
Clases=['Food', 'Attire', 'Decorationandsignage', 'misc']
Clases_num = {'Food': 0, 'Attire': 1, 'Decorationandsignage':2, 'misc':3}

# Los datos en texto se transforman en numéricos
t_num = Datos['Class'].map(Clases_num)
for i in range (11):
    print("t"+str(i)+": "+str(t_num[i]))

# %% [markdown]
# ## Clasificación
# 
# #### Para aplicar un clasificador a estos datos, necesitamos aplanar las imágenes, convirtiendo cada matriz 3-D de tamaño ``(64, 64,3)`` en un vector de tamaño ``(12288,)``. Por tanto, todo el conjunto de datos tendrá un tamaño ``(n_samples, n_features)``, donde ``n_samples`` es el número de imágenes y ``n_features`` es el número total de píxeles en cada imagen. Para aplanar los datos se empleará la función [reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html) de numpy que permite dar una nueva forma a una matriz sin cambiar sus datos:
# %% [markdown]
# ## Función: From_3D_to_1D(Entrada):

# %%
def From_3D_to_1D(Entrada):
    try:
        n_samples = len(Entrada)  #Obtenemos el número de imágenes totales en el conjunto de datos
        X_Train_Matriz=np.array(Entrada) #Convertimos la lista en una matriz para poder trabjar con ella
        data=X_Train_Matriz.reshape(n_samples,-1)
    except:
         print("Error in funtion : From_3D_to_1D(Entrada,)")
         return 0
    return data


# %%
X_1D=From_3D_to_1D(X) #Imagenes aplanadas
print("Numero de datos Entrenamiento: "+ str(len(X_1D)))
print("Dimension Imagenes Entrenamiento: "+ str(X[0].shape))
print("Imagenes aplanadas Entrenamiento: "+ str(X_1D.shape))


# %%
# Normalizar vector de imágenes a [0..1]
X_1D_norm = X_1D/255.0
for i in range(15):
    print("X"+str(i)+": "+str(X_1D_norm[i]))

# %% [markdown]
# ##  ANÁLISIS NÚMERO DE CLASES

# %%
F=0
A=0
D=0
M=0
for i in range(len(t)):
    if(t[i]=='Food'):
        F+=1
    if(t[i]=='Attire'):
        A+=1
    if(t[i]=='Decorationandsignage'):
        D+=1
    if(t[i]=='misc'):
        M+=1
print("Clase '",Clases[0],"' hay",F,"imagenes")
print("Clase '",Clases[1],"' hay",A,"imagenes")
print("Clase '",Clases[2],"' hay",D,"imagenes")
print("Clase '",Clases[3],"' hay",M,"imagenes")

# %% [markdown]
# ESTRATEGIA DE ENTRENAMIENTO / PARTICIÓN DE DATOS Train/Test
# #### Debido a:
# #### - No disponemos de una parte importante como conjunto de pruebas para nuestro modelo.
# #### - Se corre el riesgo que al dividir los datos para Train/Test estén desbalanceados por lo que entrenamiento del modelo y su posterior uso sobre los datos de Test no sean fiables.
# #### Se opta por adoptar la estrategia K-fold Cruzada en la que:
# #### Todas las muestras del conjunto de datos se usan alguna vez para entrenar o como parte del conjunto de prueba.
# 
# %% [markdown]
# ## Funciones auxiliares

# %%
# Diagrama de cajas
def boxPlot(L_modelos,L_titulos,title):
    data = L_modelos
    fig7, ax = plt.subplots()
    ax.set_title(title)
    ax.boxplot(data,labels=L_titulos)
    plt.xticks(rotation=90)

# Función de evaluación de f1 ya que no viene por defecto en Keras
def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
#to_excel
def saveToExcel(df, file, sheet, headers):
    df.to_excel(excel_writer = file, sheet_name = sheet, header = headers)

def saveDictToExcel(dic, file, sheet, headers):
    # Se transforma el diccionario en dataframe
    df = pd.DataFrame()
    for element in headers:
        df[element] = dic[element]

    # Finalmente gravamos a disco
    saveToExcel(df = df, file = file, sheet = sheet, headers = headers)

# %% [markdown]
# ## MODELOS APRENDIZAJE AUTOMÁTICO
# ### LogisticRegression
# #### multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
# ### multi_class{‘auto’, ‘ovr’, ‘multinomial’}, default=’auto’
# If the option chosen is ‘ovr’, then a binary problem is fit for each label. For ‘multinomial’ the loss minimised is the multinomial loss fit across the entire probability distribution, even when the data is binary. ‘multinomial’ is unavailable when solver=’liblinear’. ‘auto’ selects ‘ovr’ if the data is binary, or if solver=’liblinear’, and otherwise selects ‘multinomial’.
# ### roc_auc_ovo--> one vs one devuelve el area bajo la curva roc
# #### Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
# 
# Note: this implementation can be used with binary, multiclass and multilabel classification, but some restrictions apply (see Parameters).

# %%
# Valores iniciales
CV = 5 # Por defecto 5-fold
neighbours = 10
maxIte = 100000 # Número máximo de iteraciones
scoring = ['accuracy','roc_auc_ovo','f1_macro', 'precision_macro', 'recall_macro']

# %% [markdown]
# ## PRIMERA TÉCNICA: REGRESIÓN LOGÍSTICA CON SCILEARN

# %%
#LR
def lr(X, Y, max_Ite, cv, metrics):
    modelLR = LogisticRegression(penalty='none', solver = 'lbfgs', max_iter=max_Ite, multi_class='auto')
    scoresLR = cross_validate(modelLR, X, Y, cv=cv, scoring=scoring)
    return modelLR, scoresLR


# %%
# Cross validation
modelLR, scoresLR = lr(X_1D_norm, t_num, maxIte, CV, scoring)

# %% [markdown]
# ## SEGUNDA TÉCNICA: DISCRIMINANTE LINEAL CON SCILEARN

# %%
#LDA
def lda(X, Y, cv, metrics):
    modelLDA = LinearDiscriminantAnalysis()
    scoresLDA = cross_validate(modelLDA, X, Y, cv=cv, scoring=metrics)
    return modelLDA, scoresLDA


# %%
# Cross validation
modelLDA, scoresLDA = lda(X_1D_norm, t_num, CV, scoring)

# %% [markdown]
# ## TERCERA TÉCNICA: K-NEAREST NEIGHBOURS CON SCILEARN

# %%
#KNN
def knn(X, Y, n_neighbours, cv, metrics):
    modelKNN = KNeighborsClassifier(n_neighbors=n_neighbours)
    scoresKNN = cross_validate(modelKNN, X, Y, cv=cv, scoring=metrics)
    return modelKNN, scoresKNN


# %%
# Cross validation
modelKNN, scoresKNN = knn(X_1D_norm, t_num, neighbours, CV, scoring)

# %% [markdown]
# ## CUARTA TÉCNICA: RED NEURONAL CONVOLUCIONAL CON KERAS
# %% [markdown]
# ## Funciones auxiliares

# %%
def From_3D_to_3Dnorm(Entrada):
    try:
        n_samples = len(Entrada)  #Obtenemos el número de imágenes totales en el conjunto de datos
        X_Train_Matriz=np.array(Entrada)/255.0 #Convertimos la lista 
    except:
         print("Error in funtion : From_3D_to_3Dnorm(Entrada,)")
         return 0
    return X_Train_Matriz

# %% [markdown]
# ## Entrenamiento del modelo de red de neuronas con validación cruzada:

# %%
def conv(X, Y, conNeurons, denseNeurons, lr, k_folds, k_fold_reps, batch, epochs, metrics):
    
    models = [] # Array con modelos
    n_classes =  len(np.unique(Y))
    results = pd.DataFrame(columns=metrics)  

    def conv_model(conNeurons, denseNeurons, lr, metrics):
        modelDL = keras.models.Sequential()

        # Crear convoluciones
        modelDL.add(layers.Conv2D(conNeurons[0], (3, 3), activation='relu', input_shape=(64, 64, 3))) # La primera necesita el tamaño de entrada
        modelDL.add(layers.MaxPooling2D((2, 2)))

        for i in range(1, len(conNeurons) - 1):
            modelDL.add(layers.Conv2D(conNeurons[i], (3, 3), activation='relu'))
            modelDL.add(layers.MaxPooling2D((2, 2)))

        modelDL.add(layers.Conv2D(conNeurons[len(conNeurons) - 1], (3, 3), activation='relu')) # La última es así

        # Capa intermedia
        modelDL.add(layers.Flatten())

        # Crea densas
        for i in range(0, len(denseNeurons) - 1):
            modelDL.add(layers.Dense(denseNeurons[i], activation='relu'))

        modelDL.add(layers.Dense(denseNeurons[len(denseNeurons) - 1], activation='softmax')) # La última capa tiene que ser así

        # Compilar
        modelDL.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=metrics)  
        modelDL.summary()
        return modelDL

    # Se genera una K-fold estratificada
    rkf = RepeatedStratifiedKFold(n_splits=k_folds, n_repeats=k_fold_reps, random_state=42)

    # Se realizan tantos entrenamientos como valor de se indica en la validación cruzada
    for i, (train_index, test_index) in enumerate(rkf.split(X, Y)):
        # Muestra el paso de la k-fold en la que nos encontramos
        print('k_fold', i+1, 'de', k_folds * k_fold_reps)
    
        # Se categorizan las clases de salida
        Y_cat = keras.utils.to_categorical(Y, num_classes=n_classes)
    
        # Se obtienen los paquetes de datos de entrenamiento y test en base a los índices aleatorios generados en la k-fold
        X_train, Y_train = X[train_index], Y_cat[train_index]
        X_test, Y_test = X[test_index], Y_cat[test_index]
    
        # Se carga el modelo en cada paso de la kfold para resetear el entrenamiento (pesos)
        models.append(conv_model(conNeurons, denseNeurons, lr, metrics))
    
        # Se realiza el entrenamiento de la red de neuronas
        history = models[i].fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch, verbose=True)
    
        # Se añade una línea en la tabla de resultados (dataframe de pandas) con los resultados de las métricas seleccionadas
        results.loc[i] = models[i].evaluate(X_test, Y_test, batch_size=None)[1:]  # Se descarta la métrica 0 porque es el valor de la función de error
    
    return results
        


# %%
#Se normaliza el vector de imágenes
X_3Dnorm = From_3D_to_3Dnorm(X)


# %%
k_folds = 2
k_fold_reps = 2 
epochs = 2
batch_size = 20
lr = 0.001
metrics = ['accuracy','AUC','Recall','Precision', f1_score]
conNeurons = [32, 64, 64]
denseNeurons = [64, 4]

scoresDL = conv(X_3Dnorm, t_num, conNeurons, denseNeurons, lr, k_folds, k_fold_reps, batch_size, epochs, metrics)

# Rename columns of the dataframe
scoresDL.columns = ['Accuracy','AUC','Recall','Precision', 'F1']

# %% [markdown]
# ## COMPARATIVA DE LAS TÉCNICAS

# %%
# MÉTRICAS
l_accuracy=[scoresLR['test_accuracy'],scoresLDA['test_accuracy'],scoresKNN['test_accuracy'], scoresDL['Accuracy']]
l_auc=[scoresLR['test_roc_auc_ovo'],scoresLDA['test_roc_auc_ovo'],scoresKNN['test_roc_auc_ovo'], scoresDL['AUC']]
l_f1=[scoresLR['test_f1_macro'],scoresLDA['test_f1_macro'],scoresKNN['test_f1_macro'], scoresDL['F1']]
l_rec=[scoresLR['test_recall_macro'],scoresLDA['test_recall_macro'],scoresKNN['test_recall_macro'], scoresDL['Recall']]
l_pre=[scoresLR['test_precision_macro'],scoresLDA['test_precision_macro'],scoresKNN['test_precision_macro'], scoresDL['Precision']]
lN=['LR','LA','KNN','DL']

# Gráficas
boxPlot(l_accuracy,lN,'Modelos Exactitud')
boxPlot(l_auc,lN,'Modelos AUC')
boxPlot(l_f1,lN,'Modelos F1')
boxPlot(l_rec,lN,'Modelos Recall')
boxPlot(l_pre,lN,'Modelos Precision')

# %% [markdown]
# ## Estudio de los Hiperparámetros
# %% [markdown]
# ## Funciones auxiliares

# %%
# Imprime los diagramas de cajas para todos los escenarios
def imprimirEscenarios(scores, newScores, metrics):

    # Vector con las métricas de cada escenario
    lN_new=['Original']
    l_new = dict()

    for metric in metrics:
        l_new[metric] = [scores[metric]]
    

    # Cargas las métricas
    for key, value in newScores.items():
        lN_new.append(key)
        for metric in metrics:
            l_new[metric].append(newScores[key][metric])


    # Diagrama de cajas
    for metric in metrics:
        boxPlot(l_new[metric],lN_new,'Modelos ' + str(metric))

# Guarda datos en hoja de Excel evitando que se borre lo que ya existe
def saveHyperparametersScenariosDF(scores, newScores, file, headers):

    # Se transforma el diccionario de entrada en dataframe
    scores_df = pd.DataFrame()
    for element in headers:
        scores_df[element] = scores[element]

    writer = ExcelWriter(file)

    scores_df.to_excel(writer, 'Original', engine='xlsxwriter', columns=headers)

    for key, value in newScores.items():
        value.to_excel(writer, key, engine='xlsxwriter', columns=headers)

    writer.save()


def saveHyperparametersScenarios(scores, newScores, file, headers):

    # Se transforma el diccionario de entrada en dataframe
    scores_df = pd.DataFrame()
    for element in headers:
        scores_df[element] = scores[element]

    writer = ExcelWriter(file)

    scores_df.to_excel(writer, 'Original', engine='xlsxwriter', columns=headers)

    for key, value in newScores.items():

        # Se transforma el diccionario de entrada en dataframe
        newScores_df = pd.DataFrame()
        for element in headers:
            newScores_df[element] = value[element]
            print(value[element])
            

        newScores_df.to_excel(writer, key, engine='xlsxwriter', columns=headers)

    writer.save()

# %% [markdown]
# ### Variación para LR

# %%
### Modificar hiperparámetros para LR
array_CV = np.arange(2, 3, 1)
scoring = ['accuracy','roc_auc_ovo','f1_macro', 'precision_macro', 'recall_macro']
newModelsLR = dict()
newScoresLR = dict()

# Cross validation
for cv in array_CV:
    scenario = str('CV'+ str(cv))
    newModelsLR[scenario], newScoresLR[scenario] = lr(X_1D_norm, t_num, maxIte, cv, scoring)

# Imprime figuras
imprimirEscenarios(scoresLR, newScoresLR, ['test_accuracy','test_roc_auc_ovo','test_f1_macro', 'test_precision_macro', 'test_recall_macro'])

# Guarda los resultados en el Excel
saveHyperparametersScenarios(scoresLR, newScoresLR, 'modelLR.xlsx', ['test_accuracy','test_roc_auc_ovo','test_recall_macro','test_precision_macro', 'test_f1_macro'])

# %% [markdown]
# ### Variación para LDA

# %%
### Modificar hiperparámetros para LDA
array_CV = np.arange(2, 4, 1)
newModelsLDA = dict()
newScoresLDA = dict()

# Cross validation
for cv in array_CV:
    scenario = str('CV'+ str(cv))
    newModelsLDA[scenario], newScoresLDA[scenario] = lda(X_1D_norm, t_num, cv, scoring)

# Imprime figuras
imprimirEscenarios(scoresLDA, newScoresLDA, ['test_accuracy','test_roc_auc_ovo','test_f1_macro', 'test_precision_macro', 'test_recall_macro'])

# Guarda los resultados en el Excel
saveHyperparametersScenarios(scoresLDA, newScoresLDA, 'modelLDA.xlsx', ['test_accuracy','test_roc_auc_ovo','test_recall_macro','test_precision_macro', 'test_f1_macro'])

# %% [markdown]
# ### Variación para KNN

# %%
### Modificar hiperparámetros para KNN
array_CV = np.arange(2, 6, 1)
array_neighbours = np.arange(10, 101, 10)
newModelsKNN = dict()
newScoresKNN = dict()

# Cross validation para cada escenario
for cv in array_CV:
    for neighbour in array_neighbours:
        scenario = str('CV'+ str(cv) + '_' + 'NEI' + str(neighbour))
        newModelsKNN[scenario], newScoresKNN[scenario] = knn(X_1D_norm, t_num, neighbour, cv, scoring)

# Imprime figuras
imprimirEscenarios(scoresKNN, newScoresKNN, ['test_accuracy','test_roc_auc_ovo','test_f1_macro', 'test_precision_macro', 'test_recall_macro'])

# Guarda los resultados en el Excel
saveHyperparametersScenarios(scoresKNN, newScoresKNN, 'modelKNN.xlsx', ['test_accuracy','test_roc_auc_ovo','test_recall_macro','test_precision_macro', 'test_f1_macro'])

# %% [markdown]
# ### Variación para DL

# %%
# Modificar parámetros para DL
k_folds = 2
k_fold_reps = 1 
epochs = 10
batch_size = 20
lr = 0.001
metrics = ['accuracy','AUC','Recall','Precision', f1_score]

array_conNeurons = np.array([[32, 64, 64], [64, 128, 128], [128, 256, 256]])
array_denseNeurons = np.array([[64, 4], [128, 4], [256, 128, 4]])
newModelsDL = dict()
newScoresDL = dict()


# Cross validation para cada escenario
for i in range(0, len(array_conNeurons)):
    for j in range(0, len(array_denseNeurons)):

        # Nombre
        scenario = str('Conv_' + str(i * len(array_denseNeurons) + j))
        
        newScoresDL[scenario] = conv(X_3Dnorm, t_num, con, den, lr, k_folds, k_fold_reps, batch_size, epochs, metrics)
        newScoresDL[scenario].columns = ['Accuracy','AUC','Recall','Precision', 'F1']

# Imprime figuras
imprimirEscenarios(scoresDL, newScoresDL, ['Accuracy','AUC','Recall','Precision', 'F1'])

# Guarda los resultados en el Excel
saveHyperparametersScenariosDF(scoresDL, newScoresDL, 'modelDL.xlsx', ['Accuracy','AUC','Recall','Precision', 'F1'])

# %% [markdown]
# # Contraste de hipótesis
# %% [markdown]
# Contraste de hipótesis: Primero se muestran los resultados obtenidos aplicando el test de Kruskall-Wallis y el test de Anova. Si se rechaza la hipótesis, entonces se puede afirmar que los modelos son diferentes y, posteriormente, se aplican dos test de comparación múltiple mediante el uso de la clase MultiComparison. Estos métodos comprueban si hay diferencias significativas con un p<0.05, corrigiendo el hecho de que se están haciendo múltiples comparaciones que normalmente aumentarían la probabilidad de que se identifique una diferencia significativa. Un resultado de "reject = true" significa que se ha observado una diferencia significativa:
# 
# Método de Tukey. Se emplea para ello la función tukeyhsd. Método de Holm-Bonferroni. Se emplea para ello la función allpairtest

# %%
def contraste(accuracyLR, accuracyLDA, accuracyKNN, accuracyDL, alpha):
    # Los vectores para comparar tienen que tener el mismo tamaño, así que cogemos el mínimo de los dos valores
    vals = min(len(accuracyLR),len(accuracyLDA),len(accuracyKNN),len(accuracyDL))
    print(vals)
    F_statistic, pVal = stats.kruskal(accuracyLR[0:vals], accuracyLDA[0:vals], accuracyKNN[0:vals], accuracyDL[1:vals + 1])
    F_statistic2, pVal2 = stats.f_oneway(accuracyLR[0:vals], accuracyLDA[0:vals], accuracyKNN[0:vals], accuracyDL[1:vals + 1])
    print ('p-valor KrusW:', pVal)
    print ('p-valor ANOVA:', pVal2)

    if pVal <= alpha:
        print('Rechazamos la hipótesis: los modelos son diferentes\n')
        stacked_data = np.vstack((accuracyLR[0:vals], accuracyLDA[0:vals], accuracyKNN[0:vals], accuracyDL[1:vals + 1])).ravel()
        stacked_model = np.vstack((np.repeat('modelLR',vals),np.repeat('modelLDA',vals),np.repeat('modelKNN',vals), np.repeat('modelDL', vals))).ravel()    
        MultiComp = MultiComparison(stacked_data, stacked_model)
        comp = MultiComp.allpairtest(stats.ttest_rel, method='Holm')
        print (comp[0])    
        print(MultiComp.tukeyhsd(alpha=alpha))
    else:
        print('Aceptamos la hipótesis: los modelos son iguales')


# %%
# contraste
contraste(scoresLR['test_accuracy'], scoresLDA['test_accuracy'], scoresKNN['test_accuracy'], results['Accuracy'], 0.001)


