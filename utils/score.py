from .fid import FID
from .cdc import CDC
import numpy as np
from .adploss import HausdorffDTLoss

class Cal_Score():
    def __init__(self, batch_size) -> None:
        self.fid = FID(batch_size)
        self.cdc = CDC()
        self.fid_score_list = []
        self.cdc_score_list = []

    def cal_fid_and_cdc(self, pred, fid):
        fid_score = self.fid.cal_fid_score(pred, fid)
        pred_numpy = pred.cpu().permute(0,2,3,1).numpy()
        cdc_score = self.cdc.cal_cdc_score(pred_numpy)
        return fid_score, cdc_score
    
    def update(self, pred, fid):
        fid_score, cdc_score = self.cal_fid_and_cdc(pred, fid)
        self.fid_score_list.append(fid_score)
        self.cdc_score_list.append(cdc_score)

    def get_score(self):
        if len(self.fid_score_list) == 0:
            fid_avg = 0
            cdc_avg = 0
        else:
            fid_avg, cdc_avg = np.mean(self.fid_score_list), np.mean(self.cdc_score_list)
        self.fid_score_list = []
        self.cdc_score_list = []
        return fid_avg, cdc_avg
import torch
hd_loss = HausdorffDTLoss()
def cal_dice(mask, gt):
    smooth = 1.
    num = mask.size(0)
    m1 = mask.view(num, -1)  # Flatten
    m2 = gt.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def cal_iou(mask, gt):
    return  torch.mean(torch.sum(mask == gt, [1,2,3])/(320*640) )
    
def cal_Hausdorff(mask, gt):
    global hd_loss
    return hd_loss(mask, gt)

def cal_all_score(mask, gt):
    mask = mask.clone()
    mask[mask<0.5] = 0
    mask[mask>=0.5] = 1
    dice = cal_dice(mask, gt)
    iou = cal_iou(mask, gt)
    hausdorff = cal_Hausdorff(mask, gt)
    score = 0.4 * dice + 0.3 * iou + 0.3 * (1 - hausdorff)
    return [score, dice, iou, hausdorff]