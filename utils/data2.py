'''
在data基础上增加了对海德堡图片的裁剪
'''



import torch
import torch.utils.data as data_utils
import numpy as np
import os
from PIL import Image, ImageDraw
import cv2
from datetime import datetime
from xpinyin import Pinyin
from PIL import Image
import copy

def data_filter(l, img_root):
    path = os.path.join(img_root, l[0])
    n = os.listdir(path)

    if l[9] != '' and l[10] == l[15] and l[14] != '' and len(n)>=4:

        return 1
    else:
        return 0

class MyDataset(data_utils.Dataset):

    def __init__(self,img_root,data_list,dataset,transform1=None,transform2=None):

        self.img_root = img_root
        self.transform1 = transform1
        self.transform2 = transform2
        self.items = []
        with open(data_list, 'r') as handle:
            data_list = [line.split(',') for i, line in enumerate(handle.readlines()) if i > 0 ]


        for l in data_list:
            if l[0] == '':
                l[:11] = last[:11]
            last = l

        self.data_list = [t for t in data_list if data_filter(t, img_root)]

        self.idx_a = idx_a = [2, 7, 5]
        self.idx_b = idx_b = [2, 12, 5]



        for item, c in enumerate(self.data_list):

            ids = self.data_list[item][0]
            name = self.data_list[item][1]
            mach = self.data_list[item][10]
            d = {'': '.tif', '海德堡': '.tif', '蔡司': '.tiff', '科林': '.jpg'}

            file_type = d[mach]
            img_a_name = '_'.join([self.data_list[item][i] for i in idx_a])
            img_b_name = '_'.join([self.data_list[item][i] for i in idx_b])

            img_a_r_path = os.path.join(self.img_root, ids,
                                        name + '_' + img_a_name + '_横扫_' + mach + file_type)
            img_a_c_path = os.path.join(self.img_root, ids,
                                        name + '_' + img_a_name + '_纵扫_' + mach + file_type)

            img_b_r_path = os.path.join(self.img_root, ids,
                                        name + '_' + img_b_name + '_横扫_' + mach + file_type)

            img_b_c_path = os.path.join(self.img_root, ids,
                                        name + '_' + img_b_name + '_纵扫_' + mach + file_type)

            if not (os.path.exists(img_a_r_path) and os.path.exists(img_a_c_path)
            and os.path.exists(img_b_r_path) and os.path.exists(img_b_c_path)):
                continue

            self.items.append(item)

        if dataset == 'train':
            self.items = self.items[:len(self.items)//5*4]
        else:
            self.items = self.items[len(self.items)//5*4:]


    def __getitem__(self, idx):

        item = self.items[idx]

        ids = self.data_list[item][0]
        name = self.data_list[item][1]
        mach = self.data_list[item][10]
        d = {'': '.tif', '海德堡': '.tif', '蔡司': '.tiff', '科林': '.jpg'}

        file_type = d[mach]
        img_a_name = '_'.join([self.data_list[item][i] for i in self.idx_a])
        img_b_name = '_'.join([self.data_list[item][i] for i in self.idx_b])

        img_a_r_path = os.path.join(self.img_root, ids,
                                    name + '_' + img_a_name + '_横扫_' + mach + file_type)
        img_a_c_path = os.path.join(self.img_root, ids,
                                    name + '_' + img_a_name + '_纵扫_' + mach + file_type)

        img_b_r_path = os.path.join(self.img_root, ids,
                                    name + '_' + img_b_name + '_横扫_' + mach + file_type)

        img_b_c_path = os.path.join(self.img_root, ids,
                                    name + '_' + img_b_name + '_纵扫_' + mach + file_type)

        img_a_r = Image.open(img_a_r_path).convert('RGB')
        img_a_c = Image.open(img_a_c_path).convert('RGB')

        img_b_r = Image.open(img_b_r_path).convert('RGB')
        img_b_c = Image.open(img_b_c_path).convert('RGB')

        label_a = self.data_list[item][9]
        label_b = self.data_list[item][14]

        ### 处理label

        pl = ['指数', '手动', 'FC', 'CF', '手动', '眼前', '光感']
        al = [i for i in pl if i in label_a]
        bl = [i for i in pl if i in label_b]
        if len(al):
            label_a = 0.01
        label_a = torch.tensor(float(label_a), dtype=torch.float32)

        if len(bl):
            label_b = 0.01
        label_b = torch.tensor(float(label_b), dtype=torch.float32)

        item = [ids, img_a_r, img_a_c, img_b_r, img_b_c, label_a, label_b]
        if mach == '海德堡':
            for idx in range(1, 5):
                item[idx] = item[idx].crop((496, 0, 1264, 496))

        if self.transform1:
            item[1] = self.transform1(item[1])
            item[2] = self.transform1(item[2])
        if self.transform2:
            item[3] = self.transform1(item[3])
            item[4] = self.transform1(item[4])
        return  item

    def __len__(self):
        return len(self.items)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def arg2str(args):
    args_dict = vars(args)
    option_str = datetime.now().strftime('%b%d_%H-%M-%S') + '\n'

    for k, v in sorted(args_dict.items()):
        option_str += ('{}: {}\n'.format(str(k), str(v)))

    return option_str