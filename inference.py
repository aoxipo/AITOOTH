# import sys 
# sys.path.append("..") 
from train import *
from utils.score import cal_all_score
from dataloader import Dataload
from torch.utils.data import DataLoader
import cv2

from predict import getedge, getlevelset_hotmap, revert
from utils.score_numpy import normalized
from utils.office_score import Evaluateof2D
# %matplotlib inline
import datetime


def save_edge_and_levelset( levelset, mask, save_path):
    getlevelset_hotmap(normalized(levelset), save_path+"_levelset.png"),
    edge = getedge(mask)
    cv2.imwrite(save_path+"_edge.png", edge) 


device = 'cuda:0'

batch_size = 10
gt_path = "./"
parameter_path_dict = {
    921:r"H:/parameter_FPN/save_best/FL_DETR/best.pkl",
    922:r"H:/parameter_FPN/save_best/FL_DETR_2/best.pkl",
    923:r"H:\parameter_FPN\FL_DETR_reverse_1024/best.pkl",
    # 4:r"H:\parameter_FPN\GTU_gan/best.pkl",
    41:r"H:/parameter_FPN/FL_GTU/best.pkl",
    42:r"H:/parameter_FPN/gtu_pvt_test/best2.pkl",
    43:r"./save/GTU_pvt_mlp_test_edge/best.pkl",
    44:r"/T2004100/project/upload/save_best/DCMTDUNet_train/best.pkl",
    # H:\parameter_FPN\gtu_pvt_test
    #4:r"H:/parameter_FPN/GTU_decode_2/best.pkl",
    # 4:r"H:/parameter_FPN/GTU_0/best.pkl",
    # 4:r"H:/parameter_FPN/GTU_0_1/best.pkl",
    4:r"H:/parameter_FPN/GTU_dsk/best.pkl",
    # 4:r"H:/parameter_FPN/GTU_d_2/best.pkl",
}
parameter_path = parameter_path_dict[44]
batch_size = 10
method_dict = {
    0: "Unet",
    1: "RESUNet",
    2: 'RU',
    4:"GTU",
    41:"FL_GTU",
    43:"GTU pvt mlp",
    5: "FL",
    8: "FL tiny",
    9: "FL FPN",
    91: "FL FPN 4 8",
    911: "FL FPN 911",
    92:"DETR",
    921:"FL_DETR B",
    922:"FL_DETR_2",
    923:"FL_DETR_reverse",
}

trainer = Train(
        1, (224),
        name = "DCMTDUNet_train",
        method_type = 44,
        is_show = False,
        batch_size = batch_size,
        device_ = "cuda:0",
)
trainer.load_parameter( parameter_path )
parameter_path
data_path = r'/T2004100/data/tooth/val-image/'
# data_path = r'/T2004100/data/tooth/test/'
all_dataloader = Dataload(data_path, (320,320), data_type = "val", data_aug = 1) # (320,640)
save_path = './test/'
save_path = save_path + trainer.name
all_dataloader.create_dir(save_path )
save_path = save_path + "/infers/"
# save_path_attech = save_path + "/attech/"
# all_dataloader.create_dir(save_path_attech )
print("val size: {} ".format( len( all_dataloader )))
all_dataloader.create_dir(save_path )
print( "save path:" , save_path, "gt path:", gt_path)
print( "load parameter path:" , save_path)

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
    # mask = out["mask"].squeeze().cpu().numpy()
    mask = revert(out["mask"]).clone().squeeze().cpu().numpy()
    #levelset = revert( out["levelset"] ).squeeze().cpu().numpy()
    
    mask = mask[0]
    # print(mask.shape)
    # mask = (mask - np.min(mask))/(np.max(mask) - np.min(mask))
    mask[mask>=0.5] = 1
    mask[mask<0.5] = 0
    # mask = np.abs(mask + np.ones_like( mask ) * -1)
    # mask = cv2.resize(mask, (640, 320))
    #save_edge_and_levelset( mask, levelset, save_path_attech)
    imgan = np.array(mask * 255, dtype = np.uint8)
    
    imgan = cv2.cvtColor(imgan, cv2.COLOR_RGB2BGR)
    imGray = cv2.cvtColor(imgan, cv2.COLOR_BGR2GRAY)/255

    cv2.imwrite(image_save_name, imGray, [cv2.IMWRITE_PNG_BILEVEL, 1])
    if index % 100 == 0:
        print("ETA: ", index," / " ,np.round(( total - index )*(datetime.datetime.now() - start_time).seconds / 60, 3) , end='\r')
        start_time = datetime.datetime.now() 
    index += 1

print("done  500  /  0 ")

#eval = Evaluateof2D(save_path, gt_path)

#Dice_avg, hausdorff_distance_avg, iou_avg = eval.get_result()


#score = Dice_avg * 0.4 + iou_avg * 0.3 + (1 - hausdorff_distance_avg) * 0.3
#print(f"Score:{score:.6f}, DICE:{Dice_avg:.6f}, IOU:{iou_avg:.6f}, HD:{1 - hausdorff_distance_avg:.6f}")
