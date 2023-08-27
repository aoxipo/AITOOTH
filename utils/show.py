import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from .score import cal_all_score
import cv2
import numpy as np

def display_progress(cond, real, fake, edge = None, current_epoch = 0, figsize=(20,15), save = False, save_path = 'D:'):
    """
    Save cond, real (original) and generated (fake)
    images in one panel 
    """
    score = cal_all_score(real.unsqueeze(0), fake.unsqueeze(0))
    cond = cond.detach().cpu().permute(1, 2, 0).numpy()   
    real = real.detach().cpu().permute(1, 2, 0).numpy()
    fake = fake.detach().cpu().permute(1, 2, 0).numpy()
    images = [cond, real, fake]
    titles = ['input','real','generated']
    
    avg_score = np.round(score[0].cpu().numpy(), 4)
    
    if edge is not None:
        edge = edge.detach().cpu().permute(1, 2, 0).numpy()
        images.append(edge)
        titles.append("edge")
    print(f'Epoch: {current_epoch}')
    if edge is None:
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        for idx,img in enumerate(images):
            if idx == 0:
                imgan = images[0]
            else:
                imgan = img
            imgan = np.array((imgan - np.min(imgan))/(np.max(imgan) - np.min(imgan)) * 255, dtype = np.uint8)
            imgan = cv2.cvtColor(imgan, cv2.COLOR_RGB2BGR)
            ax[idx].imshow(imgan)
            ax[idx].axis("off")
        for idx, title in enumerate(titles):    
            ax[idx].set_title('{}'.format(title))
        if save:
            f = plt.gcf()  #获取当前图像
            f.savefig(save_path + '/{}_{}.png'.format(current_epoch, avg_score))
            f.clear()  #释放
        else:
            plt.show()
    else:
        fig, ax = plt.subplots(1, 4, figsize=figsize)
        for idx,img in enumerate(images):
            if idx == 0:
                imgan = images[0]
            else:
                imgan = img
            imgan = np.array((imgan - np.min(imgan))/(np.max(imgan) - np.min(imgan)) * 255, dtype = np.uint8)
            imgan = cv2.cvtColor(imgan, cv2.COLOR_RGB2BGR)
            ax[idx].imshow(imgan)
            ax[idx].axis("off")
        for idx, title in enumerate(titles):    
            ax[idx].set_title('{}'.format(title))
        if save:
            f = plt.gcf()  #获取当前图像
            f.savefig(save_path + '/{}_{}.png'.format(current_epoch, avg_score))
            f.clear()  #释放
        else:
            plt.show()