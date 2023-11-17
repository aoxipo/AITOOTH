import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from .office_score import evaluateof2d
from .score_numpy import positive_recall, negative_recall
import cv2
import numpy as np

def display_progress(clean, label, pred, edge = None, current_epoch = 0, figsize=(20,15), save = False, save_path = 'D:'):
    """
    Save clean, label (original) and generated (pred)
    images in one panel 
    """
    clean = clean.detach().cpu().permute(1, 2, 0).numpy()   
    label = label.detach().cpu().permute(1, 2, 0).numpy()
    pred = pred.detach().cpu().permute(1, 2, 0).numpy()
    pred_image = np.array( pred*255, dtype = np.uint8), 
    gt_image = np.array( label*255, dtype = np.uint8)
    ScoreDICE, ScoreIOU, ScoreHD = evaluateof2d( pred_image, gt_image)
    pr = positive_recall(gt_image, pred_image)
    nr = negative_recall(gt_image, pred_image)
    images = [clean, label, pred]
    titles = ['input','gt','predict']
    
    avg_score = 0.4*ScoreDICE+0.3*ScoreIOU+ 0.3*ScoreHD
    log_str = f'Epoch:{current_epoch}, SCORE:{avg_score:.4f}, DICE:{ScoreDICE:.4f}, IOU:{ScoreIOU:.4f}, HD:{ScoreHD:.4f}, NR:{nr:.4f}, PR:{pr:.4f}\n'
    print(log_str)
    if edge is not None:
        edge = edge.detach().cpu().permute(1, 2, 0).numpy()
        images.append(edge)
        titles.append("edge")
   
    # print(f'Epoch: {current_epoch}')

    if edge is None:
        fig, ax = plt.subplots(3, 1, figsize=figsize)
    else:
        fig, ax = plt.subplots(4, 1, figsize=figsize)
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
        fig.suptitle(log_str, fontsize = 12)
        f = plt.gcf()  #获取当前图像
        f.savefig(save_path + '/{}_{:.4f}.png'.format(current_epoch, avg_score))
        f.clear()  #释放
    else:
        plt.show()