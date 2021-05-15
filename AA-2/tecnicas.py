from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics
from keras import backend as K
import pandas as pd
from funcion_aux import historial
from division_datos import K_fold_estratificada, categorizar_datos

# MODELOS APRENDIZAJE AUTOMÁTICO
def dtc(X, Y, samples_split, samples_leaf, cv):

    # Se crea el score
    scoresDTC = pd.DataFrame(columns = ['accuracy','roc_auc_ovo','f1_macro', 'precision_macro', 'recall_macro'])
    
    # Para cada set se entrena el modelo y se calculan sus scores
    # Se usa una k-fold estratificada metiendo en el cross_validate un cv, o metiendo los sets de datos
    train, test = K_fold_estratificada(X = X, Y = Y, k_folds = cv)

    for split in range(0, cv):
        # Introduce como vector Y, un vector que va de 0 a 3
        modelDTC = DecisionTreeClassifier(min_samples_split = samples_split, min_samples_leaf = samples_leaf).fit(X = X[train[split]], y = Y[train[split]])
        Y_pred = modelDTC.predict(X[test[split]])
        Y_pred_proba = modelDTC.predict_proba(X[test[split]])

        # Guarda las metricas
        scoresDTC.loc[split + 1] = [metrics.accuracy_score(Y[test[split]], Y_pred),
                                    metrics.roc_auc_score(Y[test[split]], Y_pred_proba, average='macro', multi_class = 'ovo'),
                                    metrics.f1_score(Y[test[split]], Y_pred, average='macro'),
                                    metrics.precision_score(Y[test[split]], Y_pred, average='macro', zero_division = 0),
                                    metrics.recall_score(Y[test[split]], Y_pred, average='macro')]
    
    return scoresDTC

def knn(X, Y, neighbours, cv):

    # Se crea el score
    scoresKNN = pd.DataFrame(columns = ['accuracy','roc_auc_ovo','f1_macro', 'precision_macro', 'recall_macro'])
    
    # Para cada set se entrena el modelo y se calculan sus scores
    # Se usa una k-fold estratificada metiendo en el cross_validate un cv, o metiendo los sets de datos
    train, test = K_fold_estratificada(X = X, Y = Y, k_folds = cv)

    for split in range(0, cv):
        # Introduce como vector Y, un vector que va de 0 a 3
        modelKNN = KNeighborsClassifier(n_neighbors = neighbours).fit(X = X[train[split]], y = Y[train[split]])
        Y_pred = modelKNN.predict(X[test[split]])
        Y_pred_proba = modelKNN.predict_proba(X[test[split]])

        # Guarda las metricas
        scoresKNN.loc[split + 1] = [metrics.accuracy_score(Y[test[split]], Y_pred),
                                    metrics.roc_auc_score(Y[test[split]], Y_pred_proba, average='macro', multi_class = 'ovo'),
                                    metrics.f1_score(Y[test[split]], Y_pred, average='macro'),
                                    metrics.precision_score(Y[test[split]], Y_pred, average='macro', zero_division = 0),
                                    metrics.recall_score(Y[test[split]], Y_pred, average='macro')]
    
    return scoresKNN

def rf(X, Y, trees, cv):

    # Se crea el score
    scoresRF = pd.DataFrame(columns = ['accuracy','roc_auc_ovo','f1_macro', 'precision_macro', 'recall_macro'])
    
    # Para cada set se entrena el modelo y se calculan sus scores
    # Se usa una k-fold estratificada metiendo en el cross_validate un cv, o metiendo los sets de datos
    train, test = K_fold_estratificada(X = X, Y = Y, k_folds = cv)

    for split in range(0, cv):
        # Introduce como vector Y, un vector que va de 0 a 3
        modelRF = RandomForestClassifier(n_estimators = trees).fit(X = X[train[split]], y = Y[train[split]])
        Y_pred = modelRF.predict(X[test[split]])
        Y_pred_proba = modelRF.predict_proba(X[test[split]])

        # Guarda las metricas
        scoresRF.loc[split + 1] = [metrics.accuracy_score(Y[test[split]], Y_pred),
                                    metrics.roc_auc_score(Y[test[split]], Y_pred_proba, average='macro', multi_class = 'ovo'),
                                    metrics.f1_score(Y[test[split]], Y_pred, average='macro'),
                                    metrics.precision_score(Y[test[split]], Y_pred, average='macro', zero_division = 0),
                                    metrics.recall_score(Y[test[split]], Y_pred, average='macro')]
    
    return scoresRF

def km(X, clusters):
    modelKM = KMeans(n_clusters = clusters, random_state = 0).fit(X)
    return modelKM

def nn(X, n_neighbours):
    modelNN = NearestNeighbors(n_neighbors = n_neighbours).fit(X)
    return modelNN

def dbscan(X, epsilon, samples):
    modelDBSCAN = DBSCAN(eps = epsilon, min_samples = samples).fit(X)
    return modelDBSCAN

def pca(X, n_components):
    modelPCA = PCA(n_components = n_components)
    reduced_X = modelPCA.fit_transform(X)
    return modelPCA, reduced_X

def ica(X, n_components):
    modelICA = FastICA(n_components = n_components, random_state = 0)
    reduced_X = modelICA.fit_transform(X)
    return modelICA, reduced_X

def conv(X, Y, cv, conNeurons, denseNeurons, ler, batch, epochs, resolucion):
    # Se divide el conjunto con k_fold estratificada
    train_index, test_index = K_fold_estratificada(X = X, Y = Y, k_folds = cv)

    # Se crea el score
    scoresDL = pd.DataFrame(columns=['accuracy','roc_auc_ovo','f1_macro', 'precision_macro', 'recall_macro'])

    # Para cada set se entrena el modelo y se calculan sus scores
    for k_fold in range(0, cv):
        # Se crea el modelo
        modelDL = crea_modelo(conNeurons = conNeurons, denseNeurons = denseNeurons, ler = ler, resolucion = resolucion)
        # Se categorizan las clases de salida
        Y_cat = categorizar_datos(Y = Y)
        # Se obtienen los paquetes de datos de entrenamiento y test en base a los índices aleatorios generados en la k-fold
        X_train, Y_train = X[train_index[k_fold]], Y_cat[train_index[k_fold]]
        X_test, Y_test = X[test_index[k_fold]], Y_cat[test_index[k_fold]]
        # Se realiza el entrenamiento de la red de neuronas
        historiaDL = modelDL.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch, verbose=True)
        historial(historiaDL, 'DL_CV' + str(k_fold + 1) + '_Conv' + str(conNeurons) + '_Dense' + str(denseNeurons))
        scoresDL.loc[k_fold + 1] = modelDL.evaluate(X_test, Y_test, batch_size=None)[1:]  # Se descarta la métrica 0 porque es el valor de la función de error

    return scoresDL

def crea_modelo(conNeurons, denseNeurons, ler, resolucion):
    # Se crea el modelo
    model = keras.models.Sequential()

    # Crear convoluciones
    model.add(layers.Conv2D(conNeurons[0], (3, 3), activation = 'relu', input_shape = (resolucion, resolucion, 3))) # La primera necesita el tamaño de entrada
    model.add(layers.MaxPooling2D((2, 2)))

    for i in range(1, len(conNeurons) - 1):
        model.add(layers.Conv2D(conNeurons[i], (3, 3), activation = 'relu'))
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(conNeurons[len(conNeurons) - 1], (3, 3), activation = 'relu')) # La última es así

    # Capa intermedia
    model.add(layers.Flatten())

    # Crea densas
    for i in range(0, len(denseNeurons) - 1):
        model.add(layers.Dense(denseNeurons[i], activation = 'relu'))

    model.add(layers.Dense(denseNeurons[len(denseNeurons) - 1], activation = 'softmax')) # La última capa tiene que ser así

    # Compilar
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=ler), metrics = ['accuracy','AUC','Recall','Precision', f1_score])  
    model.summary()

    return model

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