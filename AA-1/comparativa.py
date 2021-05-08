from funcion_aux import boxPlot

# COMPARATIVA DE LAS TÉCNICAS

def comparativa_tecnicas(scoresLR, scoresLDA, scoresKNN, scoresDL):
    # MÉTRICAS
    l_accuracy=[scoresLR['test_accuracy'],scoresLDA['test_accuracy'],scoresKNN['test_accuracy'], scoresDL['Accuracy']]
    l_auc=[scoresLR['test_roc_auc_ovo'],scoresLDA['test_roc_auc_ovo'],scoresKNN['test_roc_auc_ovo'], scoresDL['AUC']]
    l_f1=[scoresLR['test_f1_macro'],scoresLDA['test_f1_macro'],scoresKNN['test_f1_macro'], scoresDL['F1']]
    l_rec=[scoresLR['test_recall_macro'],scoresLDA['test_recall_macro'],scoresKNN['test_recall_macro'], scoresDL['Recall']]
    l_pre=[scoresLR['test_precision_macro'],scoresLDA['test_precision_macro'],scoresKNN['test_precision_macro'], scoresDL['Precision']]
    lN=['LR','LA','KNN','DL']

    # Gráficas
    boxPlot(l_accuracy,lN,'Modelos Exactitud')
    boxPlot(l_auc,lN,'Modelos AUC')
    boxPlot(l_f1,lN,'Modelos F1')
    boxPlot(l_rec,lN,'Modelos Recall')
    boxPlot(l_pre,lN,'Modelos Precision')
