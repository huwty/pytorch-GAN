from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
from data.base_dataset import get_transform, get_params
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder


class Clothes(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, opt, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.opt = opt
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.train_dataset_ratio=opt.train_dataset_ratio #0.8   # 训练集比例，测试集比例为 1-this
        self.df_all_data_attr= shuffle(pd.read_csv(self.attr_path, sep='\t', header=0))  # 数据读取并打乱
        self.df_selected_attrs=self.df_all_data_attr[self.selected_attrs]
        self.preprocess()
        self.attr_value_types=np.sum(self.df_all_data_attr[self.selected_attrs].nunique())  # 获取所选属性的所有值的种类数


        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)


    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        # file_names = np.array(self.df_selected_attrs.index)
        # file_names = np.array([file_names]).T
        file_names = np.array([self.df_all_data_attr['name'].as_matrix()]).T

        onehot_enc = OneHotEncoder(categories='auto')
        label = onehot_enc.fit_transform(self.df_selected_attrs).toarray()

        dataset=np.concatenate((file_names,label),axis=1) # like [filename,0,1,0..1,1] not [filename, [labels]]
        len_train_dataset=int(len(dataset)*self.train_dataset_ratio)
        self.train_dataset=dataset[:len_train_dataset]
        self.test_dataset=dataset[len_train_dataset:]

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index][0],dataset[index][1:]
        label=label.tolist()
        # print(filename)
        file_path = os.path.join(self.image_dir, filename)
        AB = Image.open(file_path)
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
        transform_params = get_params(self.opt, (A.size))
        A_transform = get_transform(self.opt, transform_params)#, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params)#, grayscale=(self.output_nc == 1))

        A = A_transform(A)        # TODO 怎么转换合适？
        B = B_transform(B)
        return {'A': A, 'B': B, 'attributes':torch.FloatTensor(label), 'A_paths': file_path, 'B_paths': file_path}

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(opt, image_dir, attr_path, selected_attrs, image_size=256,
               batch_size=16, mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    # if mode == 'train':
    #     transform.append(T.RandomHorizontalFlip())
    # transform.append(T.CenterCrop(crop_size))
    # transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))) # TODO
    transform = T.Compose(transform)

    dataset = Clothes(opt, image_dir, attr_path, selected_attrs, transform, mode)



    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader, dataset.attr_value_types