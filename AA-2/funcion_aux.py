import matplotlib.pyplot as plt
import pandas as pd
from pandas import ExcelWriter

# Diagrama de cajas
def cajas(scores, score, title):
    keys = []
    data = []
    # Recuperar los datos
    for key, value in scores.items():
        keys.append(key)
        data.append(value[score])

    _, ax = plt.subplots()
    ax.set_title(title)
    ax.boxplot(data,labels = keys)
    plt.xticks(rotation = 90)
    plt.show(block = True)
    
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
            

        newScores_df.to_excel(writer, key, engine='xlsxwriter', columns=headers)

    writer.save()
