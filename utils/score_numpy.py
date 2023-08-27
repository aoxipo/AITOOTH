import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
import matplotlib.pyplot as plt

def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1) 
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum  - intersection 
    
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1) # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    
    smooth = .001
    dice = 2*(intersection + smooth)/(mask_sum + smooth)
    return dice

def positive_recall(y_true, y_pred, **kwargs):
    """
    compute positive recall for binary segmentation map via numpy
    """
    axes = (0, 1) # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes)

    smooth = .001
    se = (intersection + smooth) / (mask_sum + smooth)
    return se

def negative_recall(y_true, y_pred, **kwargs):
    """
    compute negative recall for binary segmentation map via numpy
    """
    axes = (0, 1) # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    pred_sum = np.sum(np.abs(y_pred), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes)
    tn = y_pred.shape[0] * y_pred.shape[1] - pred_sum - mask_sum + intersection
    fp = pred_sum - intersection

    smooth = .001

    sp = (tn + smooth) / (tn + fp + smooth)
    return sp

