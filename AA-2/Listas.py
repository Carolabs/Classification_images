import cv2
import os

# Funcion: List_name_files(path, l)
#  Nos devuelve una lista con los nombres de los archivos que se encuentran en la carpeta.
def List_name_files(path, l):
    try:
        l = []
        for item in os.listdir(path):
            l.append(item)
    except:
        print("Error in funtion : list_name_files(path, l)")
        l = []
    return l



# Funcion: List_Array_RGB(path, l)
# Devuelve una lista que representa el array de pixeles de cada imagen.

def List_Array_RGB(path, l):
    try:
        l = []
        for item in os.listdir(path):
            p = path + "/" + str(item)
            l.append(cv2.imread(p, flags=1))
    except:
        print("Error in funtion : list_name_files(path, l)")
        l = []
    return l