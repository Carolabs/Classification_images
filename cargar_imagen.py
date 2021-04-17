import os
from PIL import Image
import matplotlib.pyplot as plt
from Listas import  List_name_files, List_Array_RGB


# Las imagenes no tienen el mismo tamano, por lo que para trabajar con ellas las redimensionaremos todas con el mismo tamano, en este caso elegimos redimensionarlas como : 64x64
#  Funcion: Redim(pathImage,pathSave,listnames,D1,D2):
#  Redimensiona las imagenes que se encuentren en una carpeta determina y las guarda en otra ruta con el mismo nombre.
#  pathImage: Ruta en la que se encuentran las imagenes sin redimensionar.
#  pathSave: Ruta en la que queremos guardar las imagenes redimensionadas.
#  D1: Dimension 1 en pixeles
#  D2: Dimension 2 en pixeles

def Redim(pathImage,pathSave,D1,D2):
    try:
        for item in os.listdir(pathImage):
            path = pathImage + "/" + str(item)
            img = Image.open(path)
            new_imag = img.resize((D1, D2))
            pathsave_=pathSave + "/" + str(item)
            new_imag.save(pathsave_)
    except:
        print("Error in funtion : Redim(pathImage,pathSave,listnames,D1,D2):")


# Funcion: Plot_Imagenes(StrTitulo,ListaTitulo,ListaImg,NumImg):
#  Grafica n imagenes con sus respectivos titulos
#  StrTitulo: String que se quiera poner poner como titulo generico
#  ListaTitulo: Si tenemos una Lista con nombres.
#  ListaImg: Lista de imagenes a imprimir
#  NumImg : Numero de graficas a mostrar

def Plot_Imagenes(StrTitulo,ListaTitulo,ListaImg,NumImg):
    _, axes = plt.subplots(nrows=1, ncols=NumImg, figsize=(12, 8))  # Crea una figura con 1x5 subplots
    i=0
    for ax in axes: # Imprimos las 5 primeras imagen
        ax.set_axis_off()
        #Titulo L_nameTrainImag
        titulo=str(StrTitulo)+str(ListaTitulo[i])
        ax.set_title(titulo)
        #Array imagen L_RGBTrain
        ax.imshow(ListaImg[i],cmap=plt.cm.gray_r)
        i+=1

def cargar_imagenes(origen, destino, resolucion):
    # Guardamos los nombres de todos los archivos en nuetra carpeta de Data Images
    # Ruta
    PathData=origen
    
    # Lista vacias
    L_nameDataImag = []
    L_nameDataImag = List_name_files(PathData, L_nameDataImag)
    for i in range(5): #Ejemplo imprimiendo 5 nombres Train
        print("Imagen "+str(i)+": "+L_nameDataImag[i])

    # Guardamos en la lista L_RGBData, las matrices de pixeles de cada imagen. 
    # Lista vacia
    L_RGBData = []
    L_RGBData = List_Array_RGB(PathData, L_RGBData)

    for i in range(5):  #Ejemplo imprimiendo 5 Arrays
        print(str(L_nameDataImag[i])+"--> pixels:"+str(L_RGBData[i].shape))

    # Redimensionar las imagenes y las guardamos en la carpeta Data_Images64x64
    PathData_final=destino
    print('Comenzando a redimensionar las imagenes')
    Redim(PathData,PathData_final,resolucion,resolucion)
    print('Imagenes redimensionadas')

    # Ahora que tenemos las imagenes redimensionadas, con el uso de la funcion explicada anteriormente "List_Array_RGB", guardaremos la lista de imagenes redimesionadas.
    L_RGBData_final = []
    print('Guardando lista de las imagenes')
    L_RGBData_final = List_Array_RGB(PathData_final, L_RGBData_final)
    print('Lista de las imagenes guardada')

    for i in range(5):  #Ejemplo imprimiendo 5 Arrays
        print(str(L_nameDataImag[i])+"--> pixels redim: "+str(L_RGBData_final[i].shape))

    # Comprobamos las diferencias entre la imagen original y la redimensionada.
    Plot_Imagenes("Calidad 100% \n",L_nameDataImag,L_RGBData,5)
    Plot_Imagenes("Redimensionadas \n",L_nameDataImag,L_RGBData_final,5)