import numpy as np
from funcion_aux import imprimirEscenarios, saveHyperparametersScenarios, saveHyperparametersScenariosDF
from regresiones import lr, lda, knn
from red_convolucional import f1_score, conv

# Estudio de los Hiperparámetros
def estudio_hiperparametros(X1D, X3D, Y, scoresLR, scoresLDA, scoresKNN, scoresDL, maxIte,scoring_regresiones, metricas_conv):
    # Modificar hiperparámetros para LR
    print("Inicio de estudio hiperparametros LR")
    array_CV = np.arange(2, 5, 1)

    newModelsLR = dict()
    newScoresLR = dict()

    # Cross validation
    for cv in array_CV:
        scenario = str('CV'+ str(cv))
        newModelsLR[scenario], newScoresLR[scenario] = lr(X1D, Y, maxIte, cv, scoring_regresiones)

    # Imprime figuras
    imprimirEscenarios(scoresLR, newScoresLR, ['test_accuracy','test_roc_auc_ovo','test_f1_macro', 'test_precision_macro', 'test_recall_macro'])

    # Guarda los resultados en el Excel
    saveHyperparametersScenarios(scoresLR, newScoresLR, 'modelLR.xlsx', ['test_accuracy','test_roc_auc_ovo','test_recall_macro','test_precision_macro', 'test_f1_macro'])

    # Modificar hiperparámetros para LDA
    print("Inicio de estudio hiperparametros LDA")
    array_CV = np.arange(2, 5, 1)
    newModelsLDA = dict()
    newScoresLDA = dict()

    # Cross validation
    for cv in array_CV:
        scenario = str('CV'+ str(cv))
        newModelsLDA[scenario], newScoresLDA[scenario] = lda(X1D, Y, cv, scoring_regresiones)

    # Imprime figuras
    imprimirEscenarios(scoresLDA, newScoresLDA, ['test_accuracy','test_roc_auc_ovo','test_f1_macro', 'test_precision_macro', 'test_recall_macro'])

    # Guarda los resultados en el Excel
    saveHyperparametersScenarios(scoresLDA, newScoresLDA, 'modelLDA.xlsx', ['test_accuracy','test_roc_auc_ovo','test_recall_macro','test_precision_macro', 'test_f1_macro'])

    # Modificar hiperparámetros para KNN
    print("Inicio de estudio hiperparametros KNN")
    array_CV = np.arange(2, 6, 1)
    array_neighbours = np.arange(25, 201, 25)
    newModelsKNN = dict()
    newScoresKNN = dict()

    # Cross validation para cada escenario
    for cv in array_CV:
        for neighbour in array_neighbours:
            scenario = str('CV'+ str(cv) + '_' + 'NEI' + str(neighbour))
            newModelsKNN[scenario], newScoresKNN[scenario] = knn(X1D, Y, neighbour, cv, scoring_regresiones)

    # Imprime figuras
    imprimirEscenarios(scoresKNN, newScoresKNN, ['test_accuracy','test_roc_auc_ovo','test_f1_macro', 'test_precision_macro', 'test_recall_macro'])

    # Guarda los resultados en el Excel
    saveHyperparametersScenarios(scoresKNN, newScoresKNN, 'modelKNN.xlsx', ['test_accuracy','test_roc_auc_ovo','test_recall_macro','test_precision_macro', 'test_f1_macro'])
   
    # Modificar parámetros para DL
    print("Inicio de estudio hiperparametros DL")
    k_folds = 5
    k_fold_reps = 1 
    epochs = 25
    batch_size = 20
    lr = 0.001

    array_conNeurons = np.array([[64, 128, 128], [64, 64, 128, 128]])
    array_denseNeurons = np.array([[128, 4], [256, 128, 4]])
    newModelsDL = dict()
    newScoresDL = dict()


    # Cross validation para cada escenario
    for i in range(0, len(array_conNeurons)):
        for j in range(0, len(array_denseNeurons)):

            # Nombre
            scenario = str('Conv_' + str(i * len(array_denseNeurons) + j))
            
            newScoresDL[scenario] = conv(X3D, Y, array_conNeurons[i], array_denseNeurons, lr, k_folds, k_fold_reps, batch_size, epochs, metricas_conv)
            newScoresDL[scenario].columns = ['Accuracy','AUC','Recall','Precision', 'F1']

    # Imprime figuras
    imprimirEscenarios(scoresDL, newScoresDL, ['Accuracy','AUC','Recall','Precision', 'F1'])

    # Guarda los resultados en el Excel
    saveHyperparametersScenariosDF(scoresDL, newScoresDL, 'modelDL.xlsx', ['Accuracy','AUC','Recall','Precision', 'F1'])

    return newScoresLR, newScoresLDA, newScoresKNN, newScoresDL
