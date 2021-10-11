'''
自定义数据加载类
# 备注文件
- wnids.txt文件为文件名
- world.txt文件为文件名和所对应的类别别称
# 训练集
- 总共200个类别，每个类别有500张图片，每个类别对应着一个文件夹
- 
''' 
from torch.utils.data import Dataset, DataLoader
from torchvision import _is_tracing, models,utils,datasets,transforms
import numpy as np
import sys
import os
from PIL import Image

class TinyDataLoader(Dataset):
    def __init__(self, root, train = True, transform = None):
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")
        self.dataset_len = 0
        self.name_find_class = {}
        self.class_find_name = {}
        self.name_find_word = {}
        self.class_name_ary = []
        self.image_taget = []

        # 把文件名的类别转到语言类别
        self.create_map_for_word()

        # 如果在训练，则按照训练集的数据进行生成
        if train:
            self.create_data_for_train()
        else:
            self.create_data_for_val()


    # 默认魔法函数，用来获取数据的大小
    def __len__(self):
        return self.dataset_len


    # 默认魔法函数，用来获取每一项数据
    def __getitem__(self, idx):
        image_path, target = self.image_taget[idx]
        image = None
        with open(image_path, 'rb') as f:
            image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, target
            

    # 按照索引文件，将文件名映射成单词
    def create_map_for_word(self):
        words_file = os.path.join(self.root_dir, 'words.txt')
        temp_name_find_word = {}
        with open(words_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                content_ary = line.split('\t')
                temp_name_find_word[content_ary[0]] = content_ary[1]

        # 对训练集的文件夹进行扫描，并得到类别名的数组
        self.class_name_ary = [item .name for item in os.scandir(self.train_dir) if item.is_dir()]
        # 对文件名数组进行排序
        self.class_name_ary = sorted(self.class_name_ary)
        # 构建两个字典
        for idx, name in enumerate(self.class_name_ary):
            # 文件名->类别
            self.name_find_class[name] = idx
             # 类别->文件名
            self.class_find_name[idx] = name
            # 文件名->单词
            self.name_find_word[name] = temp_name_find_word[name]


    # 构建专属训练集的数据
    def create_data_for_train(self):
        image_nums = 0
        for idx, class_name in  enumerate(self.class_name_ary):
           one_class_path = os.path.join(self.train_dir, class_name, 'images')
           for image_name in next(os.walk(one_class_path))[2]:
                if image_name.endswith(".JPEG"):
                    # 构建图片路径-类别的元组
                    path_target = (os.path.join(one_class_path, image_name), idx)
                    self.image_taget.append(path_target)
                    image_nums += 1
        
        self.dataset_len = image_nums


    # 构建专属验证集的数据
    def create_data_for_val(self):
        # 读取txt文档获取name
        val_annotations_file = os.path.join(self.val_dir, 'val_annotations.txt')
        val_image_path = os.path.join(self.val_dir, "images")
        image_nums = 0

        with open(val_annotations_file, 'r') as fo:
            lines = fo.readlines()
            # 使用set的原因是，可能多个图片对应着一个类
            for line in lines:
                content_ary = line.split('\t')
                # 数组的第一项为图片名称
                # 数组的第二项为文件名
                # 余项为box
                path_target = (os.path.join(val_image_path, content_ary[0]), self.name_find_class[content_ary[1]])
                self.image_taget.append(path_target)
                image_nums += 1

        self.dataset_len = image_nums

