import os
import cv2
import numpy as np
from scipy import stats
import torch

def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * stats.entropy(p, M) + 0.5 * stats.entropy(q, M)


def compute_JS_bgr(input_dir, dilation=1):
    input_img_list = os.listdir(input_dir)
    input_img_list.sort()
    # print(input_img_list)

    hist_b_list = []   # [img1_histb, img2_histb, ...]
    hist_g_list = []
    hist_r_list = []
    
    for img_name in input_img_list:
        # print(os.path.join(input_dir, img_name))
        img_in = cv2.imread(os.path.join(input_dir, img_name))
        H, W, C = img_in.shape
        
        hist_b = cv2.calcHist([img_in], [0], None, [256], [0,256]) # B
        hist_g = cv2.calcHist([img_in], [1], None, [256], [0,256]) # G
        hist_r = cv2.calcHist([img_in], [2], None, [256], [0,256]) # R
        
        hist_b = hist_b / (H * W)
        hist_g = hist_g / (H * W)
        hist_r = hist_r / (H * W)
        
        hist_b_list.append(hist_b)
        hist_g_list.append(hist_g)
        hist_r_list.append(hist_r)
    
    JS_b_list = []
    JS_g_list = []
    JS_r_list = []
    
    for i in range(len(hist_b_list)):
        if i + dilation > len(hist_b_list) - 1:
            break
        hist_b_img1 = hist_b_list[i]
        hist_b_img2 = hist_b_list[i + dilation]     
        JS_b = JS_divergence(hist_b_img1, hist_b_img2)
        JS_b_list.append(JS_b)
        
        hist_g_img1 = hist_g_list[i]
        hist_g_img2 = hist_g_list[i+dilation]     
        JS_g = JS_divergence(hist_g_img1, hist_g_img2)
        JS_g_list.append(JS_g)
        
        hist_r_img1 = hist_r_list[i]
        hist_r_img2 = hist_r_list[i+dilation]     
        JS_r = JS_divergence(hist_r_img1, hist_r_img2)
        JS_r_list.append(JS_r)
        
    return JS_b_list, JS_g_list, JS_r_list


def calculate_cdc(input_folder, dilation=[1, 2, 4], weight=[1/3, 1/3, 1/3]):
    input_folder_list = os.listdir(input_folder)
    input_folder_list.sort()
    input_folder_list = [folder for folder in input_folder_list if os.path.isdir(os.path.join(input_folder, folder))]
    # print(input_folder_list)

    JS_b_mean_list, JS_g_mean_list, JS_r_mean_list = [], [], []   # record mean JS
    for i, folder in enumerate(input_folder_list):
        input_path = os.path.join(input_folder, folder)
        mean_b, mean_g, mean_r = 0, 0, 0
        
        for d, w in zip(dilation, weight):
            JS_b_list_one, JS_g_list_one, JS_r_list_one = compute_JS_bgr(input_path, d)
            mean_b += w * np.mean(JS_b_list_one)
            mean_g += w * np.mean(JS_g_list_one)
            mean_r += w * np.mean(JS_r_list_one)
            
        JS_b_mean_list.append(mean_b)
        JS_g_mean_list.append(mean_g)
        JS_r_mean_list.append(mean_r)
        
    cdc = np.mean([float(np.mean(JS_b_mean_list)), float(np.mean(JS_g_mean_list)), float(np.mean(JS_r_mean_list))])
    return cdc


class CDC():
    def __init__(self) -> None:
        self.dilation = [1, 2, 4]
        self.weight = [1/3, 1/3, 1/3]
    
    def cal_cdc_score(self, pred):
        JS_b_mean_list = []
        JS_g_mean_list = []
        JS_r_mean_list = []
        # B C H W -> B H W C
        #pred = pred.permute(0,2,3,1) 
        mean_b, mean_g, mean_r = 0, 0, 0
        for d, w in zip(self.dilation, self.weight):
            JS_b_list_one, JS_g_list_one, JS_r_list_one = self.compute_JS_bgr(pred, d)
            mean_b += w * np.mean(JS_b_list_one)
            mean_g += w * np.mean(JS_g_list_one)
            mean_r += w * np.mean(JS_r_list_one)
            JS_b_mean_list.append(mean_b)
            JS_g_mean_list.append(mean_g)
            JS_r_mean_list.append(mean_r)
            
        cdc = np.mean([float(np.mean(JS_b_mean_list)), float(np.mean(JS_g_mean_list)), float(np.mean(JS_r_mean_list))])
        return cdc

    def compute_JS_bgr(self, pred, dilation):
        hist_b_list = []   # [img1_histb, img2_histb, ...]
        hist_g_list = []
        hist_r_list = []
        
        for img in pred:
            H, W, C = img.shape
            hist_b = cv2.calcHist([img], [0], None, [256], [0,256]) # B
            hist_g = cv2.calcHist([img], [1], None, [256], [0,256]) # G
            hist_r = cv2.calcHist([img], [2], None, [256], [0,256]) # R
            hist_b = hist_b / (H * W)
            hist_g = hist_g / (H * W)
            hist_r = hist_r / (H * W)

            hist_b_list.append(hist_b)
            hist_g_list.append(hist_g)
            hist_r_list.append(hist_r)
        JS_b_list = []
        JS_g_list = []
        JS_r_list = []
        
        for i in range(len(hist_b_list)):
            if i + dilation > len(hist_b_list) - 1:
                break
            hist_b_img1 = hist_b_list[i]
            hist_b_img2 = hist_b_list[i + dilation]     
            JS_b = JS_divergence(hist_b_img1, hist_b_img2)
            JS_b_list.append(JS_b)
            
            hist_g_img1 = hist_g_list[i]
            hist_g_img2 = hist_g_list[i + dilation]     
            JS_g = JS_divergence(hist_g_img1, hist_g_img2)
            JS_g_list.append(JS_g)
            
            hist_r_img1 = hist_r_list[i]
            hist_r_img2 = hist_r_list[i + dilation]     
            JS_r = JS_divergence(hist_r_img1, hist_r_img2)
            JS_r_list.append(JS_r)
            
        return JS_b_list, JS_g_list, JS_r_list

# torch
# class CDC_torch():
#     def __init__(self) -> None:
#         self.dilation = [1, 2, 4]
#         self.weight = [1/3, 1/3, 1/3]
#         self.init_parameter()
    
#     def init_parameter(self):
#         self.JS_b_mean_list = []
#         self.JS_g_mean_list = []
#         self.JS_r_mean_list = []

#     def cal_cdc(self, pred, gt):
#         mean_b, mean_g, mean_r = 0, 0, 0
#         for d, w in zip(self.dilation, self.weight):
#             JS_b_list_one, JS_g_list_one, JS_r_list_one = self.compute_JS_bgr(pred, d)
#             mean_b += w * torch.mean(JS_b_list_one)
#             mean_g += w * torch.mean(JS_g_list_one)
#             mean_r += w * torch.mean(JS_r_list_one)

#     def JS_divergence(self, p, q):
#         p = p.softmax(-1)
#         q = q.softmax(-1)
#         M = (p + q) / 2
#         return 0.5 * torch.sum(p* torch.log(p/M)) + 0.5 * torch.sum(q* torch.log(q/M))
    
#     def compute_JS_bgr(self, pred, dilation):
#         hist_b_list = []   # [img1_histb, img2_histb, ...]
#         hist_g_list = []
#         hist_r_list = []
        
#         for img in pred:
#             C, H, W = img.shape
#             hist_b = torch.histc(img[0,:,:], bins=256, min=1, max=256) # B
#             hist_g = torch.histc(img[1,:,:], bins=256, min=1, max=256) # G
#             hist_r = torch.histc(img[2,:,:], bins=256, min=1, max=256) # R
#             hist_b = hist_b / (H * W)
#             hist_g = hist_g / (H * W)
#             hist_r = hist_r / (H * W)

#             hist_b_list.append(hist_b)
#             hist_g_list.append(hist_g)
#             hist_r_list.append(hist_r)

        
#         JS_b_list = []
#         JS_g_list = []
#         JS_r_list = []
        
#         for i in range(len(hist_b_list)):
#             if i + dilation > len(hist_b_list) - 1:
#                 break
#             hist_b_img1 = hist_b_list[i]
#             hist_b_img2 = hist_b_list[i + dilation]     
#             JS_b = self.JS_divergence(hist_b_img1, hist_b_img2)
#             JS_b_list.append(JS_b)
            
#             hist_g_img1 = hist_g_list[i]
#             hist_g_img2 = hist_g_list[i + dilation]     
#             JS_g = self.JS_divergence(hist_g_img1, hist_g_img2)
#             JS_g_list.append(JS_g)
            
#             hist_r_img1 = hist_r_list[i]
#             hist_r_img2 = hist_r_list[i + dilation]     
#             JS_r = self.JS_divergence(hist_r_img1, hist_r_img2)
#             JS_r_list.append(JS_r)
            
#         return JS_b_list, JS_g_list, JS_r_list