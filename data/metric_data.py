# encoding: utf-8

# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import os
import re
from collections import defaultdict

from torch.utils.data import Dataset
from ret_benchmark.utils.img_reader import read_image
import numpy as np
import torch

def generate_P(img_source,num_class,noise_rate):
    print('noise rate=',noise_rate)
    P=torch.empty(size=[num_class,num_class])
    P[:,:]=noise_rate/(num_class-1)
    for i in range(num_class):
        P[i,i]=1-noise_rate
    return P.cuda()

def get_noise_or_not(img_source,path_list):
    idx=img_source.find('noised_')+len('noised_')
    img_source=img_source[:idx]+'cleaned_train.csv'
    noise_or_not=[]
    clean_path_list=[]
    with open(img_source, "r") as f:
        for line in f:
            _path, _ = re.split(r",", line.strip())
            clean_path_list.append(_path)
    noise_or_not=[True if i in clean_path_list else False for i in path_list]
    return np.array(noise_or_not)
def reidx(label_list):
    new_class=[]
    new_class_d={}
    for i in range(len(label_list)):
        old_class=label_list[i]
        if old_class not in new_class_d:
            new_class_d[old_class]=str(len(new_class_d))
        new_class.append(new_class_d[old_class])
    label_list=new_class
    return label_list

def get_original_img_source(img_source):
    if 'noise' in img_source and 'cleaned' not in img_source:
        idx=img_source.find('noised_')+len('noised_')
        img_source=img_source[:idx]+'cleaned_train.csv'
        return img_source
    else:
        return None
class MetricDataSet(Dataset):
    """
    Basic Dataset read image path from img_source
    img_source: list of img_path and label
    """

    def __init__(self, img_source, transform=None, target_transform=None,
                     mode="RGB",leng=-1,train=False):
        self.mode = mode
        self.transforms = transform
        self.target_transform = target_transform
        self.root = os.path.dirname(img_source)

        if not os.path.exists(img_source):
            img_source=img_source.replace('/home/liuchang','/root')
        if not train:
            original_img_source=get_original_img_source(img_source)
            if original_img_source is not None:
                img_source=original_img_source
        assert os.path.exists(img_source), f"{img_source} NOT found."
        self.img_source = img_source

        self.label_list = list()
        self.path_list = list()
        self._load_data()
        self.label_index_dict = self._build_label_index_dict()
        if leng!=-1:
            self.path_list=self.path_list[:leng]
            self.label_list=self.label_list[:leng]
        self.label_list=reidx(self.label_list)
        if 'FOOD' in img_source or 'CARSN' in img_source:
            self.noise_or_not=None
        elif train:
            self.noise_or_not=get_noise_or_not(img_source,self.path_list)
        
        self.nb_classes=len(set(self.label_list))
        if train:
            original_img_source=get_original_img_source(img_source)
            if original_img_source is not None:
                self.P=generate_P(original_img_source,num_class=self.nb_classes,noise_rate=float(img_source.split('_')[1].split('noise')[0].split('split')[0]))
        # import pickle
        # title=os.path.basename(img_source)[:-len('_train.csv')]
        # with open('{}_sop_label_path.pkl'.format(title),'wb') as f:
        #     pickle.dump((self.label_list,self.path_list),f)

    def __len__(self):
        return len(self.label_list)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"| Dataset Info |datasize: {self.__len__()}|num_labels: {len(set(self.label_list))}|"

    def _load_data(self):
        with open(self.img_source, "r") as f:
            for line in f:
                _path, _label = re.split(r",", line.strip())
                self.path_list.append(_path)
                self.label_list.append(_label)

    def _build_label_index_dict(self):
        index_dict = defaultdict(list)
        for i, label in enumerate(self.label_list):
            index_dict[label].append(i)
        return index_dict

    def __getitem__(self, index):
        path = self.path_list[index]
        img_path = os.path.join(self.root, path)
        label = self.label_list[index]

        img = read_image(img_path, mode=self.mode)
        if self.transforms is not None:
            img = self.transforms(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        # print(index,img.shape,label)
        return img, label, index
