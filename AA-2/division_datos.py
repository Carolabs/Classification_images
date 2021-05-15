from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from tensorflow import keras

def dividir_datos(X, Y, ratio):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = ratio)
    print('El conjunto de entrenamiento contiene',Y_train.shape[0], 'datos')
    print('El conjunto de test contiene',Y_test.shape[0], 'datos')
    return X_train, X_test, Y_train, Y_test

def categorizar_datos(Y):
    # Se categorizan las clases de salida
    n_classes = len(np.unique(Y))
    return keras.utils.to_categorical(Y, num_classes = n_classes)

def K_fold_estratificada(X, Y, k_folds):
    
    # Se genera una K-fold estratificada
    rkf = StratifiedKFold(n_splits = k_folds)

    train = []
    test = []

    # Se realizan tantos entrenamientos como valor de se indica en la validación cruzada
    for _, (train_index, test_index) in enumerate(rkf.split(X, Y)):
        # Paso de la k-fold al vector de salida
        train.append(train_index)
        test.append(test_index)
    
    return train, test