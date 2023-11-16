import sys 
sys.path.append("..") 
from train import *
from utils.score import cal_all_score
from dataloader import Dataload
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from .utils.office_score import evaluateof2d
from utils.score_numpy import mean_iou_np, mean_dice_np, positive_recall, negative_recall
# %matplotlib inline
import datetime

device = 'cuda:0'
def predict( gt_path, file_path, parameter_path, name):

    ####################### define train ######################
    trainer = Train(
            1, (224),
            name = name,
            method_type = 42,
            is_show = False,
            batch_size = 10,
            device_ = "cuda:0",
    )
    trainer.load_parameter( parameter_path )
    ###################### define dataloader ###################
    all_dataloader = Dataload( file_path, (320,640), data_type = "train", data_aug = 1) # (320,640)
    save_path = './test/'
    save_path = save_path + trainer.name
    all_dataloader.create_dir(save_path )
    save_path = save_path + "/infers/"
    print("验证集合大小: {} ".format( len( all_dataloader )))
    all_dataloader.create_dir(save_path )
    print( "save path:" , save_path, "gt path:", gt_path)
    print( "load parameter path:" , save_path)
    ###########################################################
    dice = []
    iou = []
    hd = []
    pr = []
    nr = []

    total = len(all_dataloader)
    score_list = []
    index = 0
    start_time = datetime.datetime.now() 
    for image_dict in all_dataloader.photo_set:
        image_path = image_dict["image"]
        image = all_dataloader.read_image_data(image_path, True)
        batch_image = all_dataloader.datagen_val( image )
        file_name = image_path.split("/")[-1]
        image_save_name = save_path + file_name
        # batch_image = batch_image.cpu()
        
        batch_image = torch.cat([batch_image, batch_image], 0).unsqueeze(1).to("cuda:0")
        out = trainer.predict_batch(batch_image)
        #     with torch.no_grad():
        #         out = trainer.model.forward_consist(batch_image, 1)
        mask = out["mask"].squeeze().cpu().numpy()
        mask = mask[0]
       
        mask = (mask - np.min(mask))/(np.max(mask) - np.min(mask))
        mask[mask>=0.5] = 1
        mask[mask<0.5] = 0

        # mask = np.abs(mask + np.ones_like( mask ) * -1)
        imgan = np.array(mask * 255, dtype = np.uint8)
        image_path = image_dict["gt"]
        gt = all_dataloader.read_image_data(image_path, True)
        ScoreDICE, ScoreIOU, ScoreHD = evaluateof2d(imgan, gt)
        dice.append( ScoreDICE)
        iou.append( ScoreIOU)
        hd.append( ScoreHD)
        nr.append(negative_recall(gt/255, imgan/255))
        pr.append(positive_recall(gt/255, imgan/255))
        index += 1
    end_time = datetime.datetime.now() 
  

    dice_avg = np.mean(dice)
    iou_avg = np.mean(iou)
    hd_avg = np.mean(hd)
    nr_avg = np.mean(nr)
    pr_avg = np.mean(pr)
    score = dice_avg * 0.4 + iou_avg * 0.3 + hd_avg * 0.3
    avg_time = (end_time - start_time).seconds/index
    print(f"Score:{score:.6f}, DICE:{dice_avg:.6f}, IOU:{iou_avg:.6f}, HD:{hd_avg:.6f},NR:{nr_avg:.6f},PR:{pr_avg:.6f}, CostTime:{avg_time:.6f}\n")

if __name__ == '__main__':
    gt_path = "./"
    file_path = "./"
    parameter_path = ""
    name = "unet"
    predict( gt_path, file_path, parameter_path, name)
