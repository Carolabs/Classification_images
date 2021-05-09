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


# to_excel
def saveToExcel(scores, file):
    writer = ExcelWriter(file)

    for key, value in scores.items():
        value.to_excel(excel_writer = writer, sheet_name = key, engine = 'xlsxwriter', columns = ['accuracy','roc_auc_ovo','f1_macro', 'precision_macro', 'recall_macro'])
    
    writer.save()
