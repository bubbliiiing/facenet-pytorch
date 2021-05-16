#------------------------------------------------#
#   进行训练前需要利用这个文件生成cls_train.txt
#------------------------------------------------#
import os
from os import getcwd

wd = getcwd()

datasets_path = "datasets/"
types_name = os.listdir(datasets_path)
types_name = sorted(types_name)

list_file = open('cls_train.txt', 'w')
for cls_id, type_name in enumerate(types_name):
    photos_path = os.path.join(datasets_path, type_name)
    if not os.path.isdir(photos_path):
        continue
    photos_name = os.listdir(photos_path)
    for photo_name in photos_name:
        list_file.write(str(cls_id) + ";" + '%s/%s'%(wd, os.path.join(photos_path, photo_name)))
        list_file.write('\n')
list_file.close()

