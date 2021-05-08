import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

# CUARTA TÉCNICA: RED NEURONAL CONVOLUCIONAL CON KERAS

def conv(X, Y, conNeurons, denseNeurons, lr, k_folds, k_fold_reps, batch, epochs, metrics, resolucion):
    
    models = [] # Guarda los modelos que se van generar
    n_classes =  len(np.unique(Y))
    results = pd.DataFrame(columns=metrics)  

    def conv_model(conNeurons, denseNeurons, lr, metrics):
        modelDL = keras.models.Sequential()

        # Crear convoluciones
        modelDL.add(layers.Conv2D(conNeurons[0], (3, 3), activation='relu', input_shape=(resolucion, resolucion, 3))) # La primera necesita el tamaño de entrada
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

def entrena_redconv(X, Y, conNeurons, denseNeurons, lr, k_folds, k_fold_reps, batch_size, epochs, metrics, resolucion):

    scoresDL = conv(X, Y, conNeurons, denseNeurons, lr, k_folds, k_fold_reps, batch_size, epochs, metrics, resolucion)

    # Rename columns of the dataframe
    scoresDL.columns = ['Accuracy','AUC','Recall','Precision', 'F1']

    return scoresDL