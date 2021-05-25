import os
import pandas as pd

file_path = "D:/data/2011_09_28/2011_09_28_drive_0075_sync/image_02/data/"
path_list = os.listdir(file_path)  # os.listdir(file)会历遍文件夹内的文件并返回一个列表
print(path_list)
path_name = []  # 把文件列表写入save.txt中

def saveList(pathName):
    for file_name in pathName:
        with open("Image.txt", "a") as f:
            f.write(file_name.split(".")[0] + "\n")

def dirList(path_list):
    for i in range(0, len(path_list)):
        path = os.path.join(file_path, path_list[i])
    if os.path.isdir(path):
        saveList(os.listdir(path))

dirList(path_list)
saveList(path_list)

data = pd.read_csv("C:/Users/1/Desktop/批量处理/Image.txt", header=None);  # 读取需要修改的文件
data = data.astype('str')
for i in range(len(data)):
    data.iloc[i] = '2011_10_03/2011_10_03_drive_0034_sync ' + data.iloc[i] +' l' # 加上 data/obj/ 前缀
    print(data.iloc[i])

data.to_csv('./split.txt', index=None)
