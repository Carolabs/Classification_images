import pandas as pd
from Listas import  List_name_files, List_Array_RGB

#  Cargamos los datos de entrenamiento .csv, mediante el uso de la librería pandas.
#  Representamos los datos en forma de tabla "Nombre imagen" y "clase a la que pertenece"
#  Función: CSV_to_RandomTable(path)
#  Devuelve una tabla con los datos ordenados de forma aleatoria
#  path: ruta del archivo .csv

def CSV_to_RandomTable(path):
    try:
        Datos= pd.read_csv(path)
        #Vamos a mezclar los datos para que no tengan ningun orden de prioridad
        n_filas, m_colum=Datos.shape
        Datos=Datos.sample(n=n_filas, random_state=1) #La opción sample permite crear una tabla aleatoria
        return Datos
    except:
        print("Error in funtion : CSV_to_RadomTable(path)")




#  Como se puede observar en el print anterior, el orden de la tabla .csv no concuerda con el orden de nuestra lista de imagenes leídas.
#  Por lo que no podemos hacer una asociación directa entre una matriz de píxeles (64,64,3)---->"clase".
#  Tendremos que ordenar la lista primero, para ello la ordenaremos con el orden de la tabla .csv de forma que podamos asociar (64,64,3)---->"clase", esta nueva lista será la lista L_DataOrdenada

#  Función: Ordenar_Datos(nombreTabla,ListaNombres,ListaRGB):
#  Devuelve una lista con el orden de nombres especificados en una tabla.csv
#  nombreTabla= nombre de la tabla que condicionará el orden de la lista.
#  ListaNombres= nombres de la lista desordenada.
#  ListaRGB= Lista que se desea ordenar.


def Ordenar_Datos(nombreTabla,ListaNombres,ListaRGB):
    try:
        l=[]
        for i in range(len(nombreTabla)):#Recorro lista de nombres tabla
            if nombreTabla[i] in ListaNombres :#Si existe el nombre de la tabla dentro de nuestra lista de nombres
                indice=ListaNombres.index(nombreTabla[i])#Guardo el indice de ese nombre de nuestra Lista L_nameTrainImag
                l.append(ListaRGB[indice])#Con este indice sabemos el valor del vector respecto a la tabla
    except:
        print("Error in funtion : Ordenar_Datos(nombreTabla,ListaNombres,ListaRGB)")
    return l

def cargar_csv(Path_CsvData, origen, destino, clases):
    #  Cargamos los datos .csv del conjunto de datos, se muestra "nombre imagen" "clase"
    # Importar datos
    Datos = CSV_to_RandomTable(Path_CsvData)

    #Nombres Tabla
    Data_NameImg=Datos.iloc[:,:-1].values
    L_RGBData = []
    L_RGBData = List_Array_RGB(destino, L_RGBData)

    L_nameDataImag = []
    L_nameDataImag = List_name_files(origen, L_nameDataImag)

    return Ordenar_Datos(Data_NameImg,L_nameDataImag,L_RGBData), Datos['Class'].map(clases)