from entrada_salida import normalizar_entradas
from cargar_imagen import cargar_imagenes, create_folder 
from datos_csv import cargar_csv
from entrada_salida import normalizar_entradas
from regresiones import entrenar_regresiones
from red_convolucional import entrena_redconv, f1_score
from comparativa import comparativa_tecnicas
from hiperparametros import estudio_hiperparametros 
from contraste_hipotesis import contraste

# Crear directorio
create_folder('dataset/Data_Images64x64')

# Carga las imagenes y las redimensiona a 64x64
print('Comenzando preprocesado de imagenes.....')
cargar_imagenes('dataset/Data Images','dataset/Data_Images64x64', 64)
print('Finalizando preprocesado de imagenes.....')

print("Comienzo del ordenamiento de datos")
dato_entrada, t_num = cargar_csv('dataset/Data.csv', 'dataset/Data Images','dataset/Data_Images64x64', {'Food': 0, 'Attire': 1, 'Decorationandsignage':2, 'misc':3})
print("Fin del ordenamiento de datos")

print("Normalizar entradas....")
X_1D_norm, X_3Dnorm = normalizar_entradas(dato_entrada)

# Entrenamiento inicial de las tres regresiones
CV = 5 # Por defecto 5-fold
neighbours = 10
maxIte = 100000 # Número máximo de iteraciones
scoring = ['accuracy','roc_auc_ovo','f1_macro', 'precision_macro', 'recall_macro']
scoresLR, scoresLDA, scoresKNN = entrenar_regresiones(X_1D_norm, t_num, CV, neighbours, maxIte, scoring)

# Entrenamiento de la red convolucional
print("Iniciando entrenamiento red convolucional con keras")
k_folds = 5
k_fold_reps = 1 
epochs = 5
batch_size = 20
ler = 0.001
metrics = ['accuracy','AUC','Recall','Precision', f1_score]
conNeurons = [32, 64, 64]
denseNeurons = [64, 4]
scoresDL = entrena_redconv(X_3Dnorm, t_num, conNeurons, denseNeurons, ler, k_folds, k_fold_reps, batch_size, epochs, metrics, 64)

# Se realiza las comparativas de los principales parametros de las tecnicas evaluadas
comparativa_tecnicas(scoresLR, scoresLDA, scoresKNN, scoresDL)

# Se realiza el contraste de hipotesis de los valores originales
alpha = 0.001
contraste(scoresLR['test_accuracy'], scoresLDA['test_accuracy'], scoresKNN['test_accuracy'], scoresDL['Accuracy'], alpha)

#Se realiza el estudio de hiperparametros y se exportan a excel
estudio_hiperparametros(X_1D_norm, X_3Dnorm, t_num, scoresLR, scoresLDA, scoresKNN, scoresDL, maxIte,scoring, metrics, 64)