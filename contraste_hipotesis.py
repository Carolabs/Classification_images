import numpy as np
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison

# # Contraste de hipótesis
# Contraste de hipótesis: Primero se muestran los resultados obtenidos aplicando el test de Kruskall-Wallis y el test de Anova. Si se rechaza la hipótesis, entonces se puede afirmar que los modelos son diferentes y, posteriormente, se aplican dos test de comparación múltiple mediante el uso de la clase MultiComparison. Estos métodos comprueban si hay diferencias significativas con un p<0.05, corrigiendo el hecho de que se están haciendo múltiples comparaciones que normalmente aumentarían la probabilidad de que se identifique una diferencia significativa. Un resultado de "reject = true" significa que se ha observado una diferencia significativa:
# 
# Método de Tukey. Se emplea para ello la función tukeyhsd. Método de Holm-Bonferroni. Se emplea para ello la función allpairtest

def contraste(accuracyLR, accuracyLDA, accuracyKNN, accuracyDL, alpha):
    # Los vectores para comparar tienen que tener el mismo tamaño, así que cogemos el mínimo de los dos valores
    vals = min(len(accuracyLR),len(accuracyLDA),len(accuracyKNN),len(accuracyDL))
    print(vals)
    F_statistic, pVal = stats.kruskal(accuracyLR[0:vals], accuracyLDA[0:vals], accuracyKNN[0:vals], accuracyDL[1:vals + 1])
    F_statistic2, pVal2 = stats.f_oneway(accuracyLR[0:vals], accuracyLDA[0:vals], accuracyKNN[0:vals], accuracyDL[1:vals + 1])
    print ('p-valor KrusW:', pVal)
    print ('p-valor ANOVA:', pVal2)

    if pVal <= alpha:
        print('Rechazamos la hipótesis: los modelos son diferentes\n')
        stacked_data = np.vstack((accuracyLR[0:vals], accuracyLDA[0:vals], accuracyKNN[0:vals], accuracyDL[1:vals + 1])).ravel()
        stacked_model = np.vstack((np.repeat('modelLR',vals),np.repeat('modelLDA',vals),np.repeat('modelKNN',vals), np.repeat('modelDL', vals))).ravel()    
        MultiComp = MultiComparison(stacked_data, stacked_model)
        comp = MultiComp.allpairtest(stats.ttest_rel, method='Holm')
        print (comp[0])    
        print(MultiComp.tukeyhsd(alpha=alpha))
    else:
        print('Aceptamos la hipótesis: los modelos son iguales')
