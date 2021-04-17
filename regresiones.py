from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score, plot_confusion_matrix, classification_report
import sklearn.neural_network

# MODELOS APRENDIZAJE AUTOMÁTICO
# LogisticRegression
# multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
# multi_class{‘auto’, ‘ovr’, ‘multinomial’}, default=’auto’
# If the option chosen is ‘ovr’, then a binary problem is fit for each label. For ‘multinomial’ the loss minimised is the multinomial loss fit across the entire probability distribution, even when the data is binary. ‘multinomial’ is unavailable when solver=’liblinear’. ‘auto’ selects ‘ovr’ if the data is binary, or if solver=’liblinear’, and otherwise selects ‘multinomial’.
# roc_auc_ovo--> one vs one devuelve el area bajo la curva roc
# Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
# Note: this implementation can be used with binary, multiclass and multilabel classification, but some restrictions apply (see Parameters).

def lr(X, Y, max_Ite, cv, metrics):
    modelLR = LogisticRegression(penalty='none', solver = 'lbfgs', max_iter=max_Ite, multi_class='auto')
    scoresLR = cross_validate(modelLR, X, Y, cv=cv, scoring=metrics)
    return modelLR, scoresLR

def lda(X, Y, cv, metrics):
    modelLDA = LinearDiscriminantAnalysis()
    scoresLDA = cross_validate(modelLDA, X, Y, cv=cv, scoring=metrics)
    return modelLDA, scoresLDA

def knn(X, Y, n_neighbours, cv, metrics):
    modelKNN = KNeighborsClassifier(n_neighbors=n_neighbours)
    scoresKNN = cross_validate(modelKNN, X, Y, cv=cv, scoring=metrics)
    return modelKNN, scoresKNN

def entrenar_regresiones(X, Y, CV, neighbours, maxIte, scoring): 
    print('Iniciando entrenamiento LR')
    modelLR, scoresLR = lr(X, Y, maxIte, CV, scoring)

    print('Iniciando entrenamiento LDA')
    modelLDA, scoresLDA = lda(X, Y, CV, scoring)

    print('Iniciando entrenamiento KNN')
    modelKNN, scoresKNN = knn(X, Y, neighbours, CV, scoring)

    return scoresLR, scoresLDA, scoresKNN