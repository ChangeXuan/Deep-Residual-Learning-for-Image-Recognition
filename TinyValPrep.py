import os
import shutil

from PIL import Image

val_path =  '/home/xuan/桌面/DataSet/tiny-imagenet-200/train/n02113799/images'
annotations_name = 'val_annotations.txt'
annotions_path = os.path.join(val_path, annotations_name)

k = [d.name for d in os.scandir(val_path) if d.is_dir()]
print(k[:10])
for root, dirs, files in os.walk(val_path):
    print(len(files))
    break

for image_name in next(os.walk(val_path))[2]:
    print(image_name)
    break
     
# annotions_file = open(annotions_path)
# while 1:
#     line = annotions_file.readline()[:-1]
#     content_ary = line.split('\t')
#     os.mkdir(content_ary[1])
#     # 从a复制到b
#     shutil.copy(local_img_name+'/'+new_obj_name,path+'/'+new_obj_name)
#     break
#     if not line:
#         break
#     pass # do something