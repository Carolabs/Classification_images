# from PIL import Image #Permite resize
# img2 = Image.open("dataset\Train Images\image4.jpg")
# new_imag2 = img2.resize((13, 16))
# new_imag2.save("nombre.png", "png")
import cv2

# Syntax: cv2.imread(path, flag)
# path: A string representing the path of the image to be read.
# flag: It specifies the way in which image should be read. It’s default value is cv2.IMREAD_COLOR
# cv2.IMREAD_COLOR: It specifies to load a color image.
# flag=0 black and white
# flag=1 color

# img = cv2.imread("dataset\Train Images\image10891.jpg", flags=1)

# cv2.imshow("img", img)
# cv2.waitKey()
# p = img.shape
# print(p)


import fnmatch
import os

# File match
def num_files(path, ext):  # Return num of files."ext" in a folder
    try:
        n = len(fnmatch.filter(os.listdir(path), "*" + ext))
    except:
        print("Error in funtion : num_files(path, ext)")
        n = 0
    return n


N_TrainImg = num_files("dataset\Train Images", ".jpg")
N_TestImg = num_files("dataset\Test Images", ".jpg")
X_Test = []
X_Train = []
# for i in range(nTestImg):#Creamos array de imagenes
# X_Test.append(cv2.imread("dataset\Train Images\image10891.jpg", flags=1))

# File match
L_nameTrainImag = []


def list_name_files(path, l):  # Return a list with the names of files
    try:
        l = []
        for item in os.listdir(path):
            l.append(item)
    except:
        print("Error in funtion : list_name_files(path, l)")
        l = []
    return l


L_RGBTrain = []


def List_Array_RGB(path, l):  # Return a list with the names of files
    try:
        l = []
        for item in os.listdir(path):
            p = path + "/" + str(item)
            # print(p)
            l.append(cv2.imread(p, flags=1))
    except:
        print("Error in funtion : list_name_files(path, l)")
        l = []
    return l


L_RGBTrain = List_Array_RGB("dataset\Train Images", L_RGBTrain)
L_nameTrainImag = list_name_files("dataset\Train Images", L_nameTrainImag)
print(L_RGBTrain[1].shape)
print(len(L_RGBTrain))
print(L_nameTrainImag[1])
cv2.imshow("imag", L_RGBTrain[1])
cv2.waitKey()