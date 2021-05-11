from entrada_salida import normalizar_entradas
from cargar_imagen import cargar_imagenes, create_folder 
from datos_csv import cargar_csv
from tecnicas import lr, knn, rf, conv, pca, km
from funcion_aux import cajas, saveToExcel
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
print("Iniciando reduccion de la dimensionalidad con PCA")
n_components = 50
PCA, X_1D_reduced = pca(X = X_1D_norm, n_components = n_components)
plt.bar(np.arange(1, len(PCA.explained_variance_ratio_) + 1), PCA.explained_variance_ratio_ * 100)
plt.show(block=True)

# Clusterizacion mediante el uso de k-medias
print("Iniciando clusterizacion con k-medias")
n_clusters = 10
inercias = []

for n_cluster in range(1, n_clusters + 1):
    print('Con', n_cluster,'clusters')
    kmeans = km(X = X_1D_reduced, clusters = n_cluster)
    inercias.append(kmeans.inertia_)

_ = plt.plot(np.arange(1, len(inercias) + 1), inercias)
plt.ylabel("Inercia")
plt.xlabel("Clusters")
plt.show(block=True)

# Entrenamiento LR
print("Iniciando entrenamiento LR")
CV = np.arange(start = 2, stop = 6, step = 1)
maxIte = 100000
scoresLR = dict()

for cv in CV:
    print('Con CV =', cv)
    scoresLR['LR_CV' + str(cv)] = lr(X = X_1D_reduced, Y = t_num, max_Ite = maxIte, cv = cv)

# Imprimir graficas
for metric in ['test_accuracy','test_roc_auc_ovo','test_f1_macro', 'test_precision_macro', 'test_recall_macro']:
    cajas(scoresLR, metric, 'LR-' + str(metric))

# Entrenamiento RF
print("Iniciando entrenamiento RF")
CV = np.arange(start = 2, stop = 6, step = 1)
trees = np.arange(start = 5, stop = 101, step = 5)
scoresRF = dict()

for cv in CV:
    for tree in trees:
        print('Con CV =', cv, 'y trees =', tree)
        scoresRF['RF_CV' + str(cv) + '_trees' + str(tree)] = rf(X = X_1D_reduced, Y = t_num, trees = tree, cv = cv)

# Imprimir graficas
for metric in ['test_accuracy','test_roc_auc_ovo','test_f1_macro', 'test_precision_macro', 'test_recall_macro']:
    cajas(scoresRF, metric, 'RF-' + str(metric))

# Entrenamiento KNN
print("Iniciando entrenamiento KNN")
CV = np.arange(start = 2, stop = 6, step = 1)
neighbours = np.arange(start = 5, stop = 101, step = 5)
scoresKNN = dict()

for cv in CV:
    for neighbour in neighbours:
        print('Con CV =', cv, 'y neighbours =', neighbour)
        scoresKNN['KNN_CV' + str(cv) + '_neighbours' + str(neighbour)] = knn(X = X_1D_reduced, Y = t_num, neighbours = neighbour, cv = cv)

# Imprimir graficas
for metric in ['test_accuracy','test_roc_auc_ovo','test_f1_macro', 'test_precision_macro', 'test_recall_macro']:
    cajas(scoresKNN, metric, 'KNN-' + str(metric))

# Entrenamiento Keras
print("Iniciando entrenamiento red convolucional con keras")
CV = np.arange(start = 2, stop = 3, step = 1)
ler = 0.001
batch = 20 
epochs = 1
conNeurons = np.array([[8, 16, 16], [16, 16, 16], [16, 32, 32]])
denseNeurons = np.array([[32, 4], [64, 4], [128, 4]])
scoresDL = dict()
red = 0

for cv in CV:
    for convolucion in conNeurons:
        for densa in denseNeurons:
            print('Con CV =', cv, ', conNeurons =', conNeurons, 'y denseNeurons =', denseNeurons)
            scoresDL['DL_CV' + str(cv) + '_Conv' + str(red)] = conv(X = X_3Dnorm, Y = t_num, cv = cv, conNeurons = convolucion, denseNeurons = densa, ler = ler,  batch = batch, epochs = epochs, resolucion = resolucion)
            red += 1

# Imprimir graficas
for metric in ['accuracy','roc_auc_ovo','f1_macro', 'precision_macro', 'recall_macro']:
    cajas(scoresDL, metric, 'DL-' + str(metric))

# Grabar a Excel los datos obtenidos para cada tecnica y con la variacion de hiperparametros estudiada
saveToExcel(scoresLR, 'scoresLR.xlsx')
saveToExcel(scoresRF, 'scoresRF.xlsx')
saveToExcel(scoresKNN, 'scoresKNN.xlsx')
saveToExcel(scoresDL, 'scoresDL.xlsx')