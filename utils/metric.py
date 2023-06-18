import numpy
import torch
import torch.nn as nn

def get_corresponding_explanation_map(image, cam_map):
    result = torch.mul(cam_map, image)
    return result

def get_avg_drop(Y_score, O_score):
    return torch.sum((nn.ReLU(inplace=True)(Y_score - O_score))/Y_score) * 100

def increase_confidence(Y_score, O_score, class_number):
    return torch.sum(torch.sum(Y_score < O_score,1) == class_number)/Y_score.size()[0] * 100

def win(Y_score, O_score):
    return 1 - torch.sum(Y_score > O_score)/(Y_score.size()[0]*Y_score.size()[1])

image = torch.randn(1,100,100)
image.size()

cam_map = torch.zeros(1,100,100)
cam_map.size()

class_number = 10
total_image = 5
# 总patch = 5, 类别数量置信度 = 10
Y_score = torch.randn(total_image,class_number,1) # 直接将原图作为输入 使用模型分类得到的结果
#O_score = resnet(get_corresponding_explanation_map) 通过原图与对应生成的热力图cam_map相乘的图 作为输入 使用模型再次分类得到的结果
O_score = torch.randn(total_image,class_number,1)
avg_dorp_value = get_avg_drop(Y_score, O_score)
increase_confidence_value = increase_confidence(Y_score, O_score, class_number)
win_score = win(Y_score, O_score)

print(avg_dorp_value,increase_confidence_value,win_score)