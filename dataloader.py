import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
#from sklearn.model_selection import train_test_split

import torchvision.transforms as transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from utils.score import cal_all_score


class Dataload(Dataset):
    def __init__(self, dir_path, image_shape=(224, 224), data_type='train'):
        self.total_number = None
        file_path = dir_path + '/image/'
        gt_path = dir_path + '/mask/'

        self.file_path = file_path
        self.gt_path = gt_path
        self.image_shape = image_shape
        self.data_type = data_type
        self.photo_set = []
        self.set_gan(True)

        if data_type == 'train':
            self.load_data(file_path, gt_path)
        elif data_type == 'coco':
            self.load_data_coco(file_path)
        else:
            self.load_data(file_path, file_path)
        
    def check_dir(self, path):
        return os.path.exists(path)

    def create_dir(self, path):
        if not self.check_dir(path):
            os.mkdir(path)

    def read_image_data(self, file_path, need_gray = False):
        if need_gray:
            image = cv2.imread(file_path, 0)
            # image = np.stack([image, image, image], axis=2)
        else:
            image = cv2.imread(file_path)
        if(image is None):
            return None
            #raise RuntimeError('image can \'t read:' + file_path)
        return image

    def set_gan(self, set_none = None, method_list = None ):
        # cifar_norm_mean = (0.5)
        # cifar_norm_std = (0.5)
        if method_list is None:
            method_list = [
                transforms.ToPILImage(),
                transforms.Resize(self.image_shape),
                transforms.RandomCrop(size = self.image_shape, padding=(10, 20)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),      
                transforms.Normalize(0,1),     
            ]
        self.datagen = transforms.Compose(method_list)
        self.datagen_gt = transforms.Compose(method_list) # 预留 可能不一样
        if set_none is None:
            self.datagen = None
            self.datagen_gt = None

    def load_data(self, file_path, gt_path):
        # 对应文件夹的地址
        photo_path_set = []
        #check 路径 
        assert self.check_dir(file_path),'{} path not exist'.format(file_path)
        assert self.check_dir(gt_path),'{} path not exist'.format(gt_path)
        if not (file_path[-1] == '/' or  file_path[-1] == '\\'):
            file_path = file_path + '/'
        if not (gt_path[-1] == '/' or  gt_path[-1] == '\\'):
            gt_path = gt_path + '/'
        assert os.listdir(file_path) == os.listdir(gt_path), 'train and gt path dir not equ'

        middle_dir = os.listdir(file_path)
        for image_index in middle_dir:
            image_path = file_path + image_index
            gt_image_path = gt_path + image_index
            photo_path_set.append({
                    "image": image_path,
                    "gt": gt_image_path,
                })

        self.photo_set = photo_path_set
        self.total_number = len(self.photo_set)
        print("total:", self.total_number)

    def load_data_coco(self, file_path):
        # 对应文件夹的地址
        photo_path_set = []
        #check 路径 
        assert self.check_dir(file_path),'{} path not exist'.format(file_path)
        if not (file_path[-1] == '/' or  file_path[-1] == '\\'):
            file_path = file_path + '/'
     
        image_path_list = os.listdir(file_path)
        for image_path in image_path_list:
            image_path = file_path + image_path
            photo_path_set.append({
                    "image":image_path,
                    "gt":image_path,
                })

        self.photo_set = photo_path_set
        self.total_number = len(self.photo_set)
        print("total:", self.total_number)
    
    def get_edge(self, x):
        x = cv2.Canny(x, 20, 100, L2gradient=True)
        return x
    
    def add_noise(self, x):
        a = cv2.Sobel(x, cv2.CV_64F, 1, 0)
        alpha = np.random.random() * 0.5
        y = alpha * a + (1 - alpha) * x 
        return y

    def __getitem__(self, index):
        """
        获取对应index的图像,并视情况进行数据增强
        """
        if index >= self.total_number:
            raise StopIteration
        try:
            re_index = index
            if len(self.photo_set) > 0:
                image_src_path, image_gt_path = self.photo_set[re_index]['image'], self.photo_set[re_index]['gt']
                if self.data_type == 'coco':
                    image_src = self.read_image_data(image_src_path) #read color img 
                else:
                    image_src = self.read_image_data(image_src_path, True)

                if self.datagen is not None:
                    seed = torch.random.seed()
                    torch.random.manual_seed(seed)
                    if np.random.random() > 0.5:
                        image_src = self.add_noise(image_src)
                    image_src = self.datagen(image_src)

                image_mask = self.read_image_data(image_gt_path, True) 
                image_edge_mask = self.get_edge(image_mask)
                if self.datagen_gt is not None:
                    torch.random.manual_seed(seed)
                    image_mask = self.datagen_gt(image_mask)
                    torch.random.manual_seed(seed)
                    image_edge_mask = self.datagen_gt(image_edge_mask)
                            

            return image_src, [image_mask, image_edge_mask] #

        except Exception as e:
            print("发现异常")
            print(e.__class__.__name__)
            print(e)

    def __len__(self):
        return len(self.photo_set)


if __name__ == '__main__':

    batch_size = 16

    dataloader = Dataload(r"H:\DATASET\teech\train", image_shape = (160,320),data_type='train')
    dataloader.set_gan(True)

    a,b = dataloader[35]
    plt.subplot(121)
    plt.imshow(a[0].numpy())
    plt.title('image')
    plt.subplot(122)
    plt.imshow(b[0].numpy())
    plt.title('gt')
    plt.show()
    print(a.shape, b.shape)

    all_dataloader = DataLoader(
        dataset = dataloader,
        batch_size = 12,
        shuffle = True,
        drop_last = True,
    )
    
    for data in all_dataloader:
        img, label = data
        print("img shape:{}  label:shape{}".format(img.shape, label.shape))
        break
    train_size = 0.8
    validate_size = 0.2
    train_dataset, validate_dataset = torch.utils.data.random_split(dataloader
                                                                    , [train_size, validate_size])
    print("train: {} test: {} , ".format(train_size, validate_size))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    validate_loader = DataLoader(
        dataset=validate_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    