import numpy as np

# Clasificación
# Para aplicar un clasificador a estos datos, necesitamos aplanar las imágenes, convirtiendo cada matriz 3-D de tamaño ``(64, 64,3)`` en un vector de tamaño ``(12288,)``. Por tanto, todo el conjunto de datos tendrá un tamaño ``(n_samples, n_features)``, donde ``n_samples`` es el número de imágenes y ``n_features`` es el número total de píxeles en cada imagen. Para aplanar los datos se empleará la función [reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html) de numpy que permite dar una nueva forma a una matriz sin cambiar sus datos:
# Función: From_3D_to_1D(Entrada):
def From_3D_to_1Dnorm(Entrada):
    try:
        n_samples = len(Entrada)  #Obtenemos el número de imágenes totales en el conjunto de datos
        X_Train_Matriz=np.array(Entrada)/255.0 #Convertimos la lista en una matriz para poder trabjar con ella
        data=X_Train_Matriz.reshape(n_samples,-1)
    except:
         print("Error in funtion : From_3D_to_1D(Entrada,)")
         return 0
    return data

def From_3D_to_3Dnorm(Entrada):
    try:
        n_samples = len(Entrada)  #Obtenemos el número de imágenes totales en el conjunto de datos
        X_Train_Matriz=np.array(Entrada)/255.0 #Convertimos la lista 
    except:
         print("Error in funtion : From_3D_to_3Dnorm(Entrada,)")
         return 0
    return X_Train_Matriz

def normalizar_entradas (x):

    return From_3D_to_1Dnorm(x), From_3D_to_3Dnorm(x)
