import os
import torch
import shutil
import GPUtil
from tqdm import tqdm
from dataloader import Dataload
from torch.utils.data import DataLoader
from model_server.datawarper import DataWarper
from utils.logger import Logger

use_gpu = torch.cuda.is_available()
torch.cuda.manual_seed(3407)
if (use_gpu):
    deviceIDs = GPUtil.getAvailable(order='first', limit=1, maxLoad=0.8, maxMemory=0.8, includeNan=False, excludeID=[],
                                    excludeUUID=[])
    if (len(deviceIDs) != 0):
        deviceIDs = GPUtil.getAvailable(order='first', limit=1, maxLoad=1, maxMemory=1, includeNan=False, excludeID=[],
                                        excludeUUID=[])
        print(deviceIDs)
        print("detect set :", deviceIDs)
        device = torch.device("cuda:" + str(deviceIDs[0]))
else:
    device = torch.device("cpu")
print("use gpu:", use_gpu)

from model_server.GTU.models.ddpm.dd import GaussianDiffusionTrainer
from model_server.GTU.models.GT_UNet import GT_U_DC_PVTNet, GT_U_DC_PVTNet_with_ddpm_boundary
device = "cuda:0"
from utils.logger import Logger

if __name__ == "__main__":
    name = "DCMTDUNet_20"
    
    logger =Logger( 
        file_name = f"log_{name}.txt", 
        file_mode = "w+", 
        should_flush = True
    )
    print(device)
    save_path = f"./save/{name}/"
    # ddpm
    step = 20
    # dataset
    batch_size = 1000
    image_size = 320
    train_path = r'/T2004100/data/tooth/train/'
    
    param_path = r"/T2004100/project/upload/save/DCMTDUNet/best.pkl"
    All_dataloader = Dataload(
        train_path, 
        image_shape =  (image_size, image_size), #(240, 480), # (320, 640), #(256,256), #(320, 640),
        data_type = "train",
        need_gray = True,
        data_aug = 1,
        )
    dataloader = DataLoader(
        dataset = All_dataloader,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,
    )
    All_dataloader.create_dir(save_path)
    dataWarper = DataWarper().to(  device)
    # model setup
    net_model = GT_U_DC_PVTNet_with_ddpm_boundary(
        1, 1, 
        need_return_dict = True,
        need_supervision = False,
        decode_type = "mlp"
    ).to(device)
    
    # net_model.load_state_dict(torch.load(param_path, map_location = device))
    # print("load param:", param_path)
    
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=0.0004, weight_decay=1e-4)
    # cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    trainer = GaussianDiffusionTrainer( net_model, 1e-4, 0.028, step).to(device)
    
    # start training
    for e in range(51 ):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                X_train = images.to(device)
                y_gt = labels[0].to(device)
                X_train, level_set = dataWarper( X_train, y_gt)
                loss = trainer(X_train, level_set).sum() / 1000.
                loss.backward()
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": X_train.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        if e%10 == 0 and e:
            torch.save(net_model.state_dict(), os.path.join(save_path, 'ckpt_' + str(e) + "_.pt"))
    torch.save(net_model.state_dict(), os.path.join(save_path, 'best.pkl'))

    shutil.copy(f'./log_{name}.txt', f'./save/{name}/')
    