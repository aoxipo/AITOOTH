
import json
import sys
import os
import zipfile
import shutil
#from pycocotools.mask import *
import numpy as np
import time
import zipfile
# import SimpleITK as sitk
from medpy import metric
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from PIL import Image


# 错误字典
error_msg = {
    1: "Bad input file",
    2: "Wrong input file format",
    3: "Wrong Mode"
}


def dump_2_json(info, path):
    with open(path, 'w') as output_json_file:
        json.dump(info, output_json_file)


def report_error_msg(detail, showMsg, out_p):
    error_dict = dict()
    error_dict['errorDetail'] = detail
    error_dict['errorMsg'] = showMsg
    error_dict['score'] = 0
    error_dict['scoreJson'] = {}
    error_dict['success'] = False
    dump_2_json(error_dict, out_p)


def report_score(score_map, out_p):
    result = dict()
    result['success'] = True
    result['score'] = score_map['score']

    # 这里{}里面的score注意保留，但可以增加其他key，比如这样：
    # result['scoreJson'] = {'score': score, 'aaaa': 0.1}
    result['scoreJson'] = score_map

    dump_2_json(result, out_p)


class Evaluateof3D():

    def __init__(self, infer_path=None, label_path=None):
        self.infer_path = infer_path
        self.label_path = label_path

    def calculate_hd(self, pred_masks, true_masks, mask):
        hd = metric.binary.hd(pred_masks, true_masks)
        a, b, c = mask.shape  # 获取图像深度、高度和宽度信息，mask只要是3维的图像即可，可以考虑加在数据读入的时候直接获取成一个数组，二维同理
        return hd / np.sqrt(a * a + b * b + c * c)

    def calculate_dice(self, pred_data, label_data):
        intersection = np.logical_and(pred_data, label_data)
        tp = np.sum(intersection)
        fp = np.sum(pred_data) - tp
        fn = np.sum(label_data) - tp

        dice = (2 * tp) / (2 * tp + fp + fn)

        return dice

    def calculate_miou(self, pred_masks, true_masks, num_classes=2):
        num_masks = len(pred_masks)
        intersection = np.zeros(num_classes)
        union = np.zeros(num_classes)

        for i in range(num_masks):
            pred_mask = pred_masks[i]
            true_mask = true_masks[i]

            for cls in range(num_classes):
                pred_cls = pred_mask == cls
                true_cls = true_mask == cls

                intersection[cls] += np.logical_and(pred_cls, true_cls).sum()
                union[cls] += np.logical_or(pred_cls, true_cls).sum()

        iou = intersection / union
        miou = np.mean(iou)

        return miou

    def read_nifti(self, path):
        itk_img = sitk.ReadImage(path)
        itk_arr = sitk.GetArrayFromImage(itk_img)
        return itk_arr

    def get_result(self):
        dice_avg = 0
        hd_avg = 0
        iou_avg = 0
        num = 0
        for file in os.listdir(os.path.join(self.label_path)):
            infer_path = os.path.join(self.infer_path, file)
            label_path = os.path.join(self.label_path, file)  # 可能需要针对数据集位置等信息修改，同2D
            pred = self.read_nifti(infer_path)
            label = self.read_nifti(label_path)
            pred_1 = (pred == 1)
            label_1 = (label == 1)
            if pred_1.sum() > 0 and label_1.sum() > 0:
                asd = metric.binary.asd(pred == 1, label == 1)
                dice = self.calculate_dice(pred == 1, label == 1)
                hd = self.calculate_hd(pred_1 == 1, label_1 == 1, label)
                iou = self.calculate_miou(pred_1 == 1, label_1 == 1)
            dice_avg += dice
            hd_avg += hd
            iou_avg += iou
            num = num + 1

        dice_avg = dice_avg / num
        hd_avg = hd_avg / num
        iou_avg = iou_avg / num
        return dice_avg, hd_avg, iou_avg


class Evaluateof2D():
    def __init__(self, pre_path=None, gt_path=None):
        self.pre_path = pre_path
        self.gt_path = gt_path
        self.threshold_confusion = 0.5
        self.target = None
        self.output = None
        # output --- predicted
        # target --- groundtruth

    def add_batch(self, batch_tar, batch_out):

        self.target = batch_tar.flatten()
        self.output = batch_out.flatten()

    # 求混淆矩阵和IoU
    def confusion_matrix(self):
        # Confusion matrix
        y_pred = self.output >= self.threshold_confusion
        confusion = confusion_matrix(self.target, y_pred)
        # print(confusion)
        iou = 0
        if float(confusion[0, 0] + confusion[0, 1] + confusion[1, 0]) != 0:
            iou = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1] + confusion[1, 0])

        return confusion, iou

    # calculating dice
    # 与f1_score相同

    def dice(self):
        pred = self.output >= self.threshold_confusion
        dice = f1_score(self.target, pred, labels=None, average='binary', sample_weight=None)
        return dice

    def HausdorffDistance(self, pre, gt):

        if np.any(pre != 0):
            hd = metric.binary.hd(pre, gt)
            a, b = gt.shape  # 获取图像高度和宽度信息
            return hd / np.sqrt(a * a + b * b)
        else:
            return 1




    def get_result(self):
        dice_avg = 0
        hd_avg = 0
        iou_avg = 0
        num = 0

        for file in os.listdir(os.path.join(self.gt_path)):
            pre_path = os.path.join(self.pre_path, file)
            gt_path = os.path.join(self.gt_path, file)  # 可能需要进行修改
            # print(pre_path)

            x = Image.open(pre_path)
            y = Image.open(gt_path)

            pre = np.array(x)
            gt = np.array(y)

            hd = self.HausdorffDistance(pre, gt)
            
            self.add_batch(gt, pre)
            dice = self.dice()
            confusion, iou = self.confusion_matrix()

            dice_avg += dice
            hd_avg += hd
            iou_avg += iou
            num = num + 1

        dice_avg = dice_avg / num
        hd_avg = hd_avg / num
        iou_avg = iou_avg / num
        # print(hd_avg)

        return dice_avg, hd_avg, iou_avg

def getHD(pre, gt):
    if np.any(pre != 0):
        hd = metric.binary.hd(pre, gt)
        a, b = gt.shape  # 获取图像高度和宽度信息
        return hd / np.sqrt(a * a + b * b)
    else:
        return 1

def getDICE(pre, gt):
    output = pre.flatten()
    target = gt.flatten()
    pred = output >= 0.5
    dice = f1_score(target, pred, labels=None, average='binary', sample_weight=None)
    return dice

def getIOU(pre, gt):
    # Confusion matrix
    output = pre.flatten()
    target = gt.flatten()
    y_pred = output >= 0.5
    confusion = confusion_matrix(target, y_pred)
 
    iou = 0
    if float(confusion[0, 0] + confusion[0, 1] + confusion[1, 0]) != 0:
        iou = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1] + confusion[1, 0])

    return confusion, iou

def evaluateof2d( x, y):
    pre = np.array(x)
    gt = np.array(y)
    ScoreHD = getHD(pre, gt)
    ScoreDICE = getDICE(pre, gt)
    ScoreIOU = getIOU(pre, gt)
    return ScoreDICE, ScoreIOU, 1 - ScoreHD
    
class MyException(Exception):
    def __init__(self, error_code=None):
        self.error_code = error_code


if __name__ == "__main__":
    '''
      online evaluation
    '''
    in_param_path = "./input_param.json"
    out_path = "./requirements.txt"

    # read submit and answer file from first parameter
    with open(in_param_path, 'r') as load_f:
        input_params = json.load(load_f)

    # 标准答案路径
    standard_file = input_params["fileData"]["standardFilePath"]
    print("Read standard from %s" % standard_file)

    # 选手提交的结果文件路径,不包括文件名
    submit_file = input_params["fileData"]["userFilePath"]
    print("Read user submit file from %s" % submit_file)

    try:
        # TODO: 执行评测逻辑

        # standard_file 代表标准答案的路径
        if os.path.isdir('./standard') and len(os.listdir('./standard')) > 0:
            print("no need to unzip %s", standard_file)
        else:
            with zipfile.ZipFile(standard_file, "r") as zip_ref:
                zip_ref.extractall("./standard")
                zip_ref.close()

        # submit_file 表示选手提交的文件路径
        submit_file_dir = os.path.join("./submit/", "")
        if os.path.isdir(submit_file_dir):
            shutil.rmtree(submit_file_dir)
        with zipfile.ZipFile(submit_file, "r") as zip_data:
            zip_data.extractall(submit_file_dir)
            zip_data.close()

        submit_path = os.path.join(submit_file_dir, 'infers')
        standard_path = os.path.join('standard', 'label')
        #print(standard_path, submit_path)

        # # 查询评估模式，并修改mask地址
        # for file in os.listdir(os.path.join(submit_path, 'infers')):
        #     if os.path.splitext(file)[-1]=='.png':
        #         evaluate_mode = "2D"
        #         standard_path = standard_path[0]
        #
        #     elif os.path.splitext(file)[-1] == '.nii.gz':
        #         evaluate_mode = "3D"
        #         standard_path = standard_path[0]
        #
        #     else:
        #         raise MyException(2)

        # # 2D 评估
        # if evaluate_mode == "2D":
        eval = Evaluateof2D(submit_path, standard_path)
        Dice_avg, hausdorff_distance_avg, iou_avg = eval.get_result()
        # 3D 评估
        # eval = Evaluateof3D(submit_path, standard_path)
        # Dice_avg, hausdorff_distance_avg, iou_avg = eval.get_result()

        # else:
        #     raise MyException(3)

        # 加权评分
        score = Dice_avg * 0.4 + iou_avg * 0.3 + (1 - hausdorff_distance_avg) * 0.3
        score_map = {}
        score_map['score'] = score
        score_map['dice'] = Dice_avg
        score_map['iou'] = iou_avg
        score_map['hausdorff_distance'] = hausdorff_distance_avg
        print(score_map)
        report_score(score_map, out_path)

    except MyException as e:
        check_code = e
        report_error_msg(error_msg[check_code], error_msg[check_code], out_path)

