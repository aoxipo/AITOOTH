import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import traceback
#from sklearn.model_selection import train_test_split

import torchvision.transforms as transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from utils.score import cal_all_score
from model_server.util import get_level_set
from torch.utils.data import Subset

class CustomSubset(Subset):
    '''A custom subset class'''
    def __init__(self, dataset, indices, dtype = "train", method_list = []):
        super().__init__(dataset, indices)
        self.dtype = dtype
        
    def __getitem__(self, idx): 
        if self.dtype == "train":
            x, y = self.dataset[self.indices[idx]] 
        else:
            x, y = self.dataset.getvalitem(self.indices[idx]) 
        return x, y 

    def __len__(self): 
        return len(self.indices)

class Dataload(Dataset):
    def __init__(self, dir_path, image_shape=(224, 224), data_type='train', need_gray = True, data_aug = 10):
        self.total_number = None
        file_path = dir_path + '/image/'
        gt_path = dir_path + '/mask/'

        self.need_gray = need_gray
        self.file_path = file_path
        self.gt_path = gt_path
        self.image_shape = image_shape
        self.data_type = data_type
        self.photo_set = []
        self.set_gan(True)
        self.use_mosaic = True
        
        

        if data_type == 'train':
            self.load_data(file_path, gt_path)
        elif data_type == 'coco':
            self.load_data_coco(file_path)
        else:
            self.load_data(file_path, file_path)
            
        d = [ self.photo_set for i in range(data_aug)]
        self.photo_set = np.array(d).flatten()
        self.indices = [ i for i in range(len(self.photo_set)) ]
        self.total_number = len(self.photo_set)
        print("total:", self.total_number)
        
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
        if set_none is None:
            self.datagen = None
            self.datagen_gt = None
        else:
            if method_list is None:
                self.datagan_read = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(self.image_shape),
                ])
                self.datagan_guass = transforms.Compose([
                    transforms.GaussianBlur(7),
                ])
                self.datagan_random = transforms.Compose([
                    transforms.RandomCrop(size = self.image_shape, padding=(10, 20)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(10),
                ])
                self.datagan_normal = transforms.Compose([
                    transforms.ToTensor(),      
                    transforms.Normalize(0,1),  
                ])
        
        method_list = [
            transforms.ToPILImage(),
            transforms.Resize(self.image_shape),
            transforms.ToTensor(),      
            transforms.Normalize(0,1),     
        ]
        self.datagen = True
        self.datagen_gt = True
        self.datagen_val = transforms.Compose(method_list)
     
    def load_mosaic(self, index):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        
        random.shuffle(indices)
        img4 = np.zeros(  (2 * self.image_shape[0], 2 * self.image_shape[1]) if self.need_gray else (2 * self.image_shape[0], 2 * self.image_shape[1], 3) ,dtype = np.uint8)
        mask4 = np.zeros( (2 * self.image_shape[0], 2 * self.image_shape[1]), dtype = np.uint8)
       
        for i, index in enumerate(indices):
            # Load image
            re_index = index
            image_src_path, image_gt_path = self.photo_set[re_index]['image'], self.photo_set[re_index]['gt']
            # print( image_src_path, image_gt_path )
            img = self.read_image_data(image_src_path, self.need_gray)
            mask = self.read_image_data(image_gt_path, True)
            img = cv2.resize(img, (self.image_shape[1],self.image_shape[0]))
            mask = cv2.resize(mask, (self.image_shape[1],self.image_shape[0]))
            # print(img.shape, mask.shape, img4.shape, mask4.shape)
            # place img in img4
            if i == 0:  # top left
                x1a, x2a, y1a, y2a = 0, self.image_shape[0], 0, self.image_shape[1] 
            elif i == 1:  # top right
                x1a, x2a, y1a, y2a = 0, self.image_shape[0], self.image_shape[1], 2 * self.image_shape[1]
            elif i == 2:  # bottom left
                x1a, x2a, y1a, y2a = self.image_shape[0], 2 * self.image_shape[0], 0, self.image_shape[1] 
            elif i == 3:  # bottom right
                x1a, x2a, y1a, y2a = self.image_shape[0], 2 * self.image_shape[0], self.image_shape[1], 2 * self.image_shape[1]
            # print(img.shape, mask.shape, x1a, x2a, y1a, y2a)
            img4[x1a:x2a, y1a:y2a] = img # img4[ymin:ymax, xmin:xmax]
            mask4[x1a:x2a, y1a:y2a] = mask

        a = int(random.uniform(self.image_shape[0], 2 * self.image_shape[0]))
        b = int(random.uniform(self.image_shape[1], 2 * self.image_shape[1]))

        img_ = img4[int(a - self.image_shape[0]) : a, int(b - self.image_shape[1]) : b]
        mask_ = mask4[int(a - self.image_shape[0]) : a, int(b - self.image_shape[1]) : b]
        return img_, mask_

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
        print( file_path, len(os.listdir(file_path)), gt_path, len(os.listdir(gt_path)))
        assert len(os.listdir(file_path)) == len(os.listdir(gt_path)), 'train and gt path dir not equ'

        middle_dir = os.listdir(file_path)
        for image_index in middle_dir:
            image_path = file_path + image_index
            gt_image_path = gt_path + image_index
            photo_path_set.append({
                    "image": image_path,
                    "gt": gt_image_path,
                })

        self.photo_set = photo_path_set
        
    
    def load_index_data(self, index):
        
        if self.use_mosaic and np.random.random() > 0.5:
            image_src, image_mask = self.load_mosaic(index)
        else:
            image_src_path, image_gt_path = self.photo_set[index]['image'], self.photo_set[index]['gt']
            
            if self.data_type == 'coco':
                image_src = self.read_image_data(image_src_path) #read color img 
            else:
                image_src = self.read_image_data(image_src_path, self.need_gray)

            image_mask = self.read_image_data(image_gt_path, True) 

        return image_src, image_mask

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
        a_normal =  (a - np.min(a))/(np.max(a) - np.min(a))
        alpha = np.random.random() * 0.2 + 0.4
        y = np.array( (alpha * a_normal + (1 - alpha) * x/255) * 255 , dtype = np.uint8)
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
                
                image_src, image_mask  = self.load_index_data(re_index)

                if self.datagen is not None:
                    # if np.random.random() > 0.5:
                    #     image_src = self.add_noise(image_src)

                    image_src = self.datagan_read(image_src)
                    # image_src = self.datagan_guass(image_src)
                    seed = torch.random.seed()
                    torch.random.manual_seed(seed)
                    image_src = self.datagan_random(image_src)
                    image_src = self.datagan_normal(image_src)
                
                
                image_edge_mask = self.get_edge(image_mask)
                if self.datagen_gt is not None:
                    
                    image_mask = self.datagan_read(image_mask)
                    level_set = get_level_set(image_mask)

                    torch.random.manual_seed(seed)
                    image_mask = self.datagan_random(image_mask)
                    image_mask = self.datagan_normal(image_mask)

                    torch.random.manual_seed(seed)
                    level_set = self.datagan_random(level_set)
                    level_set = self.datagan_normal(level_set)

                    image_edge_mask = self.datagan_read(image_edge_mask)
                    torch.random.manual_seed(seed)
                    image_edge_mask = self.datagan_random(image_edge_mask)
                    image_edge_mask = self.datagan_normal(image_edge_mask)
                            
            return image_src, [ image_mask, image_edge_mask, level_set] #
        except Exception as e:
            print("发现异常")
            print(e.__class__.__name__)
            print(e)
            print(index)
            print(traceback.print_exc())

    def getvalitem(self, index):
        if index >= self.total_number:
            raise StopIteration
        try:
            re_index = index
            if len(self.photo_set) > 0:
                image_src_path, image_gt_path = self.photo_set[re_index]['image'], self.photo_set[re_index]['gt']
                if self.data_type == 'coco':
                    image_src = self.read_image_data(image_src_path) #read color img 
                else:
                    image_src = self.read_image_data(image_src_path, self.need_gray)

                if self.datagen_val is not None:
                    
                    # if np.random.random() > 0.5:
                    #     image_src = self.add_noise(image_src)
                    image_src = self.datagen_val(image_src)
                
                image_mask = self.read_image_data(image_gt_path, True) 
                image_edge_mask = self.get_edge(image_mask)
                if self.datagen_val is not None:

                    image_mask = self.datagen_val(image_mask)
                    image_edge_mask = self.datagen_val(image_edge_mask)
                            

            return image_src, [image_mask, image_edge_mask] #

        except Exception as e:
            print("发现异常")
            print(e.__class__.__name__)
            print(e)

    def __len__(self):
        return len(self.photo_set)


if __name__ == '__main__':

    batch_size = 16

    dataloader = Dataload(r"H:\DATASET\teech\train", image_shape = (160,320), data_type='train')
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
        batch_size = 1,
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
    