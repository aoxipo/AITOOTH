from train import *
from utils.score import cal_all_score
from dataloader import Dataload
from torch.utils.data import DataLoader
import cv2
from utils.office_score import evaluateof2d
import torchvision.transforms as transforms
from utils.score_numpy import mean_iou_np, mean_dice_np, positive_recall, negative_recall, normalized
import torch.nn as nn
# %matplotlib inline
import datetime
import os
revert = transforms.Compose(
    [
    transforms.Resize((320, 640)),
    ]
)
class MinPool(nn.Module):
    def __init__(self, kernel_size, ndim=2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super(MinPool, self).__init__()
        self.pool = getattr(nn, f'MaxPool{ndim}d')(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                                  return_indices=return_indices, ceil_mode=ceil_mode)
    @torch.no_grad()
    def forward(self, x):
        x = self.pool(-x)
        return -x

erode = MinPool(3,2,1).cuda()

def getedge(image):
    mask = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    mask1 = nn.functional.pad(mask, (1,1,1,1))
    edge = mask - erode(mask1)
    edge = (1 - edge) * 255
    # cv2.imwrite( save_path + name + ".png_edge.png" , edge.squeeze().numpy() )
    return edge.squeeze().numpy()

def getlevelset_hotmap( levelset, save_path):
    levelset[levelset==0] = np.nan
    plt.figure(figsize = (12,12))
    plt.imshow(levelset)
    plt.axis('off')
    plt.savefig(save_path,bbox_inches='tight',pad_inches=0.0)
    # plt.show()

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

device = 'cuda:0'

def saveTop(Top_list, save_path):
    create_dir(save_path)
    for index in range(len(Top_list)-1):
        obj = Top_list[index]
        name = obj["Path"].replace("\\", "/").split("/")[-1]
        mask = np.array(normalized(obj["Mask"]) * 255, dtype = np.uint8)
        image = np.array(normalized(obj["image"]) * 255, dtype = np.uint8)
        levelset = np.array(normalized(obj["LevelSet"]) * 255, dtype = np.uint8)
        edge = np.array(normalized(obj["Edge"]) * 255, dtype = np.uint8)
        path = save_path + name
        cv2.imwrite(path, image) 
        mask_name = f"{path[:-3]}_{obj['score']:.4f}_{obj['DICE']:.4f}_{obj['IOU']:.4f}_{obj['HD']:.4f}_mask.png"
        cv2.imwrite(mask_name, mask) 
        cv2.imwrite(path[:-3]+"_levelset.png", levelset) 
        getlevelset_hotmap( normalized(obj["LevelSet"]), path[:-3]+"_levelset.png")
        edge = getedge(mask/255)
        cv2.imwrite(path+"_edge.png", edge) 


def insertTop(Top_list, obj, TopK = 5):
    Top_list.insert(0, obj)
    if len(Top_list)>TopK:
        Top_list.pop()
    return Top_list

def predict( gt_path, file_path, parameter_path, name, method_index = 43, Topk = 5):
    Top_list = [
                {
                    "score":0,
                    "DICE":0,
                    "IOU":0,
                    "HD":0,
                    "image":None,
                    "Mask":None,
                    "Path":None,
                    "LevelSet":None,
                    "Edge":None,
                }
            ]
    bottom_list = [
                {
                    "score":999,
                    "DICE":0,
                    "IOU":0,
                    "HD":0,
                    "image":None,
                    "Mask":None,
                    "Path":None,
                    "LevelSet":None,
                    "Edge":None,
                }
            ]
    ####################### define train ######################
    trainer = Train(
            1, (224),
            name = name,
            method_type = method_index,
            is_show = False,
            batch_size = 10,
            device_ = "cuda:0",
    )
    trainer.load_parameter( parameter_path )
    ###################### define dataloader ###################
    all_dataloader = Dataload( file_path, (320,320), data_type = "train", data_aug = 1) # (320,640)
    save_path = './test/'
    save_path = save_path + trainer.name
    all_dataloader.create_dir(save_path )
    save_path = save_path + "/infers/"
    all_dataloader.create_dir(save_path )
    print("val size:{} ".format( len( all_dataloader )))
    # all_dataloader.create_dir(save_path )
    print( "save path:" , save_path, "gt path:", gt_path)
    print( "load parameter path:" , parameter_path)
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
        mask = revert(out["mask"]).squeeze().cpu().numpy()
        mask = mask[0]
        # mask[mask >= 0.5] = 1
        # mask[mask< 0.5] = 0
        # mask[:25, :] = 0
        # mask[300:, :] = 0
        # mask[:, :25] = 0
        # mask[:, -25:] = 0
        image_path = image_dict["gt"]
        gt = all_dataloader.read_image_data(image_path, True)
        imgan = mask
        if np.sum(mask) != 0:
            imgan = np.array(normalized( imgan ))
        if np.sum(gt) != 0:
            gt = np.array(normalized(gt)) 

        # print(np.sum(mask), np.sum(gt))
        if np.sum(mask) and np.sum(gt):
            ScoreDICE, ScoreIOU, ScoreHD = evaluateof2d(imgan, gt)
        else:
            ScoreDICE, ScoreIOU, ScoreHD = 1,1,1

        Score = 0.4 * ScoreDICE + 0.3 * ScoreIOU + 0.3 * ScoreHD
        nrScore = negative_recall(gt, imgan)
        prScore = positive_recall(gt, imgan)
        dice.append( ScoreDICE)
        iou.append( ScoreIOU)
        hd.append( ScoreHD)
        nr.append(nrScore)
        pr.append(prScore)
        if Score > Top_list[0]["score"] or len(Top_list)<Topk:
            obj = {
                "score": Score,
                "DICE": ScoreDICE,
                "IOU": ScoreIOU,
                "HD": ScoreHD,
                "pr": round(prScore, 6),
                "nr": round(nrScore, 6),
                "image": image,
                "Mask": mask,
                "gt":gt,
                "Path": image_save_name,
                "LevelSet": revert(out["levelset"])[0].squeeze().cpu().numpy(),
                "Edge":revert(out["edge"])[0].squeeze().cpu().numpy(),
            }
            insertTop( Top_list, obj, Topk)
        
        if Score < bottom_list[0]["score"]:
            obj = {
                "score": Score,
                "DICE": ScoreDICE,
                "IOU": ScoreIOU,
                "HD": ScoreHD,
                "pr": round(prScore, 6),
                "nr": round(nrScore, 6),
                "image": image,
                "Mask": mask,
                "gt":gt,
                "Path": image_save_name,
                "LevelSet": revert(out["levelset"])[0].squeeze().cpu().numpy(),
                "Edge": revert(out["edge"])[0].squeeze().cpu().numpy(),
            }
            insertTop( bottom_list, obj, Topk)
        
        index += 1
        if index % 100 == 0:
            print("process ", index/total, "...")
    end_time = datetime.datetime.now() 
  

    dice_avg = np.mean(dice)
    iou_avg = np.mean(iou)
    hd_avg = np.mean(hd)
    nr_avg = np.mean(nr)
    pr_avg = np.mean(pr)
    score = dice_avg * 0.4 + iou_avg * 0.3 + hd_avg * 0.3
    avg_time = (end_time - start_time).seconds/index
    print(f"Score:{score:.6f}, DICE:{dice_avg:.6f}, IOU:{iou_avg:.6f}, HD:{hd_avg:.6f},NR:{nr_avg:.6f},PR:{pr_avg:.6f}, CostTime:{avg_time:.6f}\n")
    saveTop(Top_list, save_path + "top/")
    saveTop(bottom_list, save_path + "bottom/")


if __name__ == '__main__':
    gt_path = r"/T2004100/data/tooth/r-train/labelled/mask"
    file_path = r"/T2004100/data/tooth/r-train/labelled/"

    parameter_path = r"/T2004100/project/upload/save/GTU_pvt_mlp_test_edge/best.pkl"
    parameter_path = r"/T2004100/project/upload/save/GTU_pvt_mlp_without_enchance/best.pkl"
    parameter_path = r"/T2004100/project/upload/save/GTU_pvt_mlp_without_enchance_no_sk/best.pkl"
    parameter_path = r"/T2004100/project/upload/save/GTU_pvt_mlp_without_bonudary/best.pkl"
    parameter_path = r"/T2004100/project/upload/save/GTU_pvt_mlp_without_bonudary/best.pkl"
    parameter_path = r"/T2004100/project/upload/save/GTU_pvt_mlp/best.pkl"
    parameter_path = r"/T2004100/project/upload/save/GTU/best.pkl"
    parameter_path = r"/T2004100/project/upload/save_best/DCMTDUNet_train/best.pkl"
    name = "DCMTDUNet_train"
    predict( gt_path, file_path, parameter_path, name, 44)

