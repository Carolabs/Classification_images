import matplotlib.pyplot as plt
import pandas as pd
from pandas import ExcelWriter


# Diagrama de cajas
def boxPlot(L_modelos,L_titulos,title):
    data = L_modelos
    fig7, ax = plt.subplots()
    ax.set_title(title)
    ax.boxplot(data,labels=L_titulos)
    plt.xticks(rotation=90)
    
#to_excel
def saveToExcel(df, file, sheet, headers):
    df.to_excel(excel_writer = file, sheet_name = sheet, header = headers)

def saveDictToExcel(dic, file, sheet, headers):
    # Se transforma el diccionario en dataframe
    df = pd.DataFrame()
    for element in headers:
        df[element] = dic[element]

    # Finalmente gravamos a disco
    saveToExcel(df = df, file = file, sheet = sheet, headers = headers)

# Imprime los diagramas de cajas para todos los escenarios
def imprimirEscenarios(scores, newScores, metrics):

    # Vector con las métricas de cada escenario
    lN_new=['Original']
    l_new = dict()

    for metric in metrics:
        l_new[metric] = [scores[metric]]
    

    # Cargas las métricas
    for key, value in newScores.items():
        lN_new.append(key)
        for metric in metrics:
            l_new[metric].append(newScores[key][metric])


    # Diagrama de cajas
    for metric in metrics:
        boxPlot(l_new[metric],lN_new,'Modelos ' + str(metric))


# Guarda datos en hoja de Excel evitando que se borre lo que ya existe
def saveHyperparametersScenariosDF(scores, newScores, file, headers):

    # Se transforma el diccionario de entrada en dataframe
    scores_df = pd.DataFrame()
    for element in headers:
        scores_df[element] = scores[element]

    writer = ExcelWriter(file)

    scores_df.to_excel(writer, 'Original', engine='xlsxwriter', columns=headers)

    for key, value in newScores.items():
        value.to_excel(writer, key, engine='xlsxwriter', columns=headers)

    writer.save()


def saveHyperparametersScenarios(scores, newScores, file, headers):

    # Se transforma el diccionario de entrada en dataframe
    scores_df = pd.DataFrame()
    for element in headers:
        scores_df[element] = scores[element]

    writer = ExcelWriter(file)

    scores_df.to_excel(writer, 'Original', engine='xlsxwriter', columns=headers)

    for key, value in newScores.items():

        # Se transforma el diccionario de entrada en dataframe
        newScores_df = pd.DataFrame()
        for element in headers:
            newScores_df[element] = value[element]
            print(value[element])
            

        newScores_df.to_excel(writer, key, engine='xlsxwriter', columns=headers)

    writer.save()
