from entrada_salida import normalizar_entradas
from cargar_imagen import cargar_imagenes, create_folder 
from datos_csv import cargar_csv
from tecnicas import lr, knn, rf, conv
from sklearn import metrics
import numpy as np

# Crear directorio
create_folder('dataset/Data_Images64x64')

# Carga las imagenes y las redimensiona a 64x64
print('Comenzando preprocesado de imagenes.....')
resolucion = 64
cargar_imagenes('dataset/Data Images','dataset/Data_Images64x64', resolucion)
print('Finalizando preprocesado de imagenes.....')

print("Comienzo del ordenamiento de datos")
dato_entrada, t_num = cargar_csv('dataset/Data.csv', 'dataset/Data Images','dataset/Data_Images64x64', {'Food': 0, 'Attire': 1, 'Decorationandsignage':2, 'misc':3})
print("Fin del ordenamiento de datos")

print("Normalizar entradas....")
X_1D_norm, X_3Dnorm = normalizar_entradas(dato_entrada)


# Entrenamiento LR
print("Iniciando entrenamiento LR")
CV = np.arange(start=2, stop=3, step=1)
maxIte = 100000
scoresLR = []

for cv in CV:
    print('Con CV =', cv)
    #scores = lr(X = X_1D_norm, Y = t_num, max_Ite = maxIte, cv = cv)
    #scoresLR.append(scores)

# Entrenamiento RF
print("Iniciando entrenamiento RF")
CV = np.arange(start=2, stop=6, step=1)
#trees = np.arange(start=25, stop=201, step=25)
trees = 50
modelRF = []
scoresRF = []

for cv in CV:
    print('Con CV =', cv, 'y trees =', trees)
    #scores = rf(X = X_1D_norm, Y = t_num, trees = trees, cv = cv)
    #scoresRF.append(scores)


# Entrenamiento KNN
print("Iniciando entrenamiento KNN")
CV = np.arange(start=2, stop=6, step=1)
#neighbours = np.arange(start=25, stop=201, step=25)
neighbours = 50
scoresKNN = []

for cv in CV:
    print('Con CV =', cv, 'y neighbours =', neighbours)
    #scores = knn(X = X_1D_norm, Y = t_num, n_neighbours = neighbours, cv = cv)
    #scoresKNN.append(scores)


# Entrenamiento Keras
print("Iniciando entrenamiento red convolucional con keras")
CV = np.arange(start=2, stop=3, step=1)
ler = 0.001
batch = 20 
epochs = 5
conNeurons = np.array([8, 16, 16])
denseNeurons = np.array([64, 4])
scoresDL = []

for cv in CV:
    print('Con CV =', cv, ', conNeurons =', conNeurons, 'y denseNeurons =', denseNeurons)
    scores = conv(X = X_3Dnorm, Y = t_num, cv = cv, conNeurons = conNeurons, denseNeurons = denseNeurons, ler = ler,  batch = batch, epochs = epochs, resolucion = resolucion)
    scoresDL.append(scores)

print(scoresDL)

# Se realiza las comparativas de los principales parametros de las tecnicas evaluadas
#comparativa_tecnicas(scoresLR, scoresLDA, scoresKNN, scoresDL)

# Se realiza el contraste de hipotesis de los valores originales
#alpha = 0.001
#contraste(scoresLR['test_accuracy'], scoresLDA['test_accuracy'], scoresKNN['test_accuracy'], scoresDL['Accuracy'], alpha)

#Se realiza el estudio de hiperparametros y se exportan a excel
#estudio_hiperparametros(X_1D_norm, X_3Dnorm, t_num, scoresLR, scoresLDA, scoresKNN, scoresDL, maxIte,scoring, metrics, 64)