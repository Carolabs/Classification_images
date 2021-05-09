from entrada_salida import normalizar_entradas
from cargar_imagen import cargar_imagenes, create_folder 
from datos_csv import cargar_csv
from tecnicas import lr, knn, rf, conv, pca
from funcion_aux import cajas
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

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


# Reduccion de la dimensionalidad para entrenar las regresiones del sklearn
# Se pasan de 12288 elementos a 50
n_components = 50
PCA, X_1D_reduced = pca(X_1D_norm, n_components)
plt.bar(np.arange(1, len(PCA.explained_variance_ratio_) + 1), PCA.explained_variance_ratio_ * 100)
plt.show(block=True)

# Clusterizacion

# Entrenamiento LR
print("Iniciando entrenamiento LR")
CV = np.arange(start=2, stop=6, step=1)
maxIte = 100000
scoresLR = dict()

for cv in CV:
    print('Con CV =', cv)
    scoresLR['LR_CV' + str(cv)] = lr(X = X_1D_reduced, Y = t_num, max_Ite = maxIte, cv = cv)

# Imprimir graficas
for metric in ['accuracy','roc_auc_ovo','f1_macro', 'precision_macro', 'recall_macro']:
    cajas(scoresLR, metric, 'LR-' + str(metric))

# Entrenamiento RF
print("Iniciando entrenamiento RF")
CV = np.arange(start=2, stop=6, step=1)
trees = np.arange(start=25, stop=201, step=25)
scoresRF = dict()

for cv in CV:
    for tree in trees:
        print('Con CV =', cv, 'y trees =', trees)
        scoresRF['RF_CV' + str(cv) + '_trees' + str(tree)] = rf(X = X_1D_reduced, Y = t_num, trees = tree, cv = cv)

# Imprimir graficas
for metric in ['accuracy','roc_auc_ovo','f1_macro', 'precision_macro', 'recall_macro']:
    cajas(scoresLR, metric, 'RF-' + str(metric))

# Entrenamiento KNN
print("Iniciando entrenamiento KNN")
CV = np.arange(start=2, stop=6, step=1)
neighbours = np.arange(start=25, stop=201, step=25)
scoresKNN = dict()

for cv in CV:
    for neighbour in neighbours:
        print('Con CV =', cv, 'y neighbours =', neighbours)
        scoresKNN['KNN_CV' + str(cv) + '_neighbours' + str(neighbour)] = knn(X = X_1D_reduced, Y = t_num, n_neighbours = neighbour, cv = cv)

# Imprimir graficas
for metric in ['accuracy','roc_auc_ovo','f1_macro', 'precision_macro', 'recall_macro']:
    cajas(scoresLR, metric, 'KNN-' + str(metric))

# Entrenamiento Keras
print("Iniciando entrenamiento red convolucional con keras")
CV = np.arange(start=2, stop=3, step=1)
ler = 0.001
batch = 20 
epochs = 5
conNeurons = np.array([8, 16, 16])
denseNeurons = np.array([64, 4])
scoresDL = dict()

for cv in CV:
    print('Con CV =', cv, ', conNeurons =', conNeurons, 'y denseNeurons =', denseNeurons)
    scoresDL['DL_CV' + str(cv)] = conv(X = X_3Dnorm, Y = t_num, cv = cv, conNeurons = conNeurons, denseNeurons = denseNeurons, ler = ler,  batch = batch, epochs = epochs, resolucion = resolucion)


# Se realiza las comparativas de los principales parametros de las tecnicas evaluadas
#comparativa_tecnicas(scoresLR, scoresLDA, scoresKNN, scoresDL)

# Se realiza el contraste de hipotesis de los valores originales
#alpha = 0.001
#contraste(scoresLR['test_accuracy'], scoresLDA['test_accuracy'], scoresKNN['test_accuracy'], scoresDL['Accuracy'], alpha)

#Se realiza el estudio de hiperparametros y se exportan a excel
#estudio_hiperparametros(X_1D_norm, X_3Dnorm, t_num, scoresLR, scoresLDA, scoresKNN, scoresDL, maxIte,scoring, metrics, 64)