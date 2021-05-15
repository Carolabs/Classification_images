import matplotlib.pyplot as plt
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

# Plot epochs
def historial(history, name):
    plt.figure(figsize=(12,5))
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label = 'Test')
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.grid()
    plt.title('Modelo ' + str(name))
    plt.legend(loc='lower right')
    plt.show(block = True)

# to_excel
def saveToExcel(scores, file):
    writer = ExcelWriter(file)

    for key, value in scores.items():
        value.to_excel(excel_writer = writer, sheet_name = key, engine = 'xlsxwriter', columns = ['accuracy','roc_auc_ovo','f1_macro', 'precision_macro', 'recall_macro'])
    
    writer.save()

