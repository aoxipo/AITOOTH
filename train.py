from dataloader import Dataload, CustomSubset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
import os
import torch
import numpy as np
import datetime
import GPUtil
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt

from utils.adploss import AdpLoss, diceCoeffv2, Cal_HausdorffDTLoss
from utils.score import cal_all_score
from model_server.datawarper import DataWarper
from utils.score_numpy import mean_iou_np, mean_dice_np, positive_recall, negative_recall

from utils.office_score import evaluateof2d
from utils.show import display_progress
from model_server.util import crop_tensor, cat_tensor
from utils.utils import normalized

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


class Train():
    def __init__(self, in_channles, image_size = 320, name = 'dense', method_type = 0, 
                 is_show = True, batch_size = 1, device_ = None, split = False):
        self.in_channels = in_channles
        self.batch_size = batch_size
        self.image_size = image_size
        self.name = name
        self.device = device_
        self.method_type = method_type
        self.lr = 0.0001
        self.history_acc = []
        self.history_loss = []
        self.history_test_acc = []
        self.history_test_loss = []
        self.history_score = []
        self.split = split
        self.create(is_show)

    def create(self, is_show):
        self.data_warper = DataWarper()
        batch_size = self.batch_size
        if self.device is not None:
            device = self.device
        if (self.method_type == 0):
            from model_server.model import Unet as Model
            self.model = Model()
            print("build Unet model")
            
        elif self.method_type == 1:
            from model_server.RESUNet import RESUNet as Model
            self.model = Model()
            print("build RESUNet model")
        elif self.method_type == 2:
            from model_server.model import RUnet as Model
            self.model = Model()
            print("build RU model")

        elif self.method_type == 3:
            from model_server.FL_DETR import FPN as Model
            self.model = Model(
                    middle_channel = [32, 64, 128, 256],
                    need_return_dict = True
                )
            print("build FPN model")
        
        elif self.method_type == 4:
            from model_server.GTU.models.GT_UNet import GT_U_DCNet as Model
            self.model = Model(
                   1, 1, 
                   middle_channel = [64, 128, 256, 512, 768, 1024],
                   encode_len = 6,
                   need_return_dict = True,
                   need_supervision = False,
                   decode_type = "conv"
                )
            print(f"build {self.model.__class__.__name__} model")
        
        elif self.method_type == 42:
            from model_server.GTU.models.GT_UNet import GT_U_DC_PVTNet as Model
            self.model = Model(
                   1, 1, 
                   need_return_dict = True,
                   need_supervision = False,
                   decode_type = "conv"
                )
            print(f"build {self.model.__class__.__name__} model")
        elif self.method_type == 43:
            from model_server.GTU.models.GT_UNet import GT_U_DC_PVTNet as Model
            self.model = Model(
                   1, 1, 
                   need_return_dict = True,
                   need_supervision = False,
                   decode_type = "mlp"
                )
            print(f"build {self.model.__class__.__name__} model")

        elif self.method_type == 41:
            from model_server.FL_GTU import FL_GTU as Model
            self.model = Model(
                   1, 1, need_return_dict = True
                )
            print("build FPN_GTU model")

        elif self.method_type == 5:
            from model_server.FL3D import FL3D as Model
            self.model = Model(1, 1)
            print("build Focus on local")
        elif self.method_type == 6:
            from model_server.FPN import FL2D as Model
            self.model = Model( 
                in_channel = 1,
                out_channel = 1,
                middle_channel = 1, 
                embed_shape = ( 2, 4),
                nstack = 2,
                batch_size = batch_size,
                need_return_dict = True
            )
            print("build FL 2D local")
        elif self.method_type == 7:
            from model_server.FL_base import FL_base as Model
            self.model = Model( 
                in_channel = 1,
                out_channel = 1,
                middle_channel = 1, 
                embed_shape = ( 2, 4),
                batch_size = batch_size,
                need_return_dict = True
            )
            print("build FL_base local")
        elif self.method_type == 8:
            from model_server.FL_seris import FL_tiny as Model
            self.model = Model( 
                in_channel = 1,
                out_channel = 1,
                middle_channel = 1, 
                embed_shape = ( 2, 4),
                batch_size = batch_size,
                need_return_dict = True
            )
            print("build FL_tiny local")
        elif self.method_type == 9:
            from model_server.FPN import FL_FPN as Model
            self.model = Model( 
                in_channel = 1,
                out_channel = 1,
                middle_channel = 1, 
                embed_shape = ( 2, 4),
                batch_size = batch_size,
                need_return_dict = True
            )
            print("build FL_FPN local")
        elif self.method_type == 911:
            from model_server.FPN_BEST import FL_FPN as Model
            self.model = Model( 
                in_channel = 1,
                out_channel = 1,
                middle_channel = 1, 
                embed_shape = ( 2, 4),
                batch_size = batch_size,
                need_return_dict = True
            )
            print("build FL_FPN local")
        elif self.method_type == 91:
            from model_server.FPN import FL_FPN as Model
            self.model = Model( 
                in_channel = 1,
                out_channel = 1,
                middle_channel = 1, 
                embed_shape = ( 3, 6),
                batch_size = batch_size,
                need_return_dict = True
            )
            print("build FL_FPN 3, 6 local")
        elif self.method_type == 92:
            from model_server.FL_DETR import FL_DETR as Model
            self.model = Model( 
                in_channel = 1,
                out_channel = 1,
                encode_len = 4, 
                need_return_dict = True
            )
            print("build FL_DETR local")
        elif self.method_type == 921:
            from model_server.detr.FL_DETR_B import FL_DETR as Model
            self.model = Model( 
                in_channel = 1,
                out_channel = 1,
                encode_len = 4, 
                need_return_dict = True
            )
            print("build FL_DETR B local")
        elif self.method_type == 922:
            from model_server.detr.FL_DETR_2 import FL_DETR as Model
            self.model = Model( 
                in_channel = 1,
                out_channel = 1,
                encode_len = 4, 
                need_return_dict = True
            )
            print("build FL_DETR_2 local")
        elif self.method_type == 923:
            from model_server.detr.FL_DETR_reverse import FL_DETR as Model
            self.model = Model( 
                in_channel = 1,
                out_channel = 1,
                encode_len = 4, 
                need_return_dict = True
            )
            print("build FL_DETR_reverse local")
        else:
            raise NotImplementedError
        
        self.cost = torch.nn.MSELoss()
        self.encropy_cost = torch.nn.CrossEntropyLoss()
        self.l1 = torch.nn.L1Loss()
        self.downsample = torch.nn.MaxPool2d(2)
        if (use_gpu):
            self.model = self.model.to(device)
            self.cost = self.cost.to(device)
            self.encropy_cost = self.encropy_cost.to(device)
            self.l1 = self.l1.to(device)
            self.downsample = self.downsample.to(device)
            self.data_warper = self.data_warper.to(device)
            
        if (is_show):
            summary(self.model, (self.in_channels, self.image_size * 2, self.image_size))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.adp = AdpLoss()

    def train_and_test(self, n_epochs, data_loader_train, data_loader_test):
        best_loss = 1000000
        es = 0
        
        self.save_parameter()
        for epoch in range(n_epochs):
            sc = False
            start_time = datetime.datetime.now()
            print("Epoch {}/{}".format(epoch, n_epochs))
            print("-" * 10)

            epoch_train_loss = self.train(data_loader_train)
            if epoch % 10 == 0:
                sc = True
            epoch_test_loss = self.test(data_loader_test, sc)

            self.history_acc.append(0)
            self.history_loss.append(epoch_train_loss)
            self.history_test_acc.append(0)
            self.history_test_loss.append(epoch_test_loss)
            print(
                "Train Loss is:{:.4f}\nTest Loss is:{:.4f}\ncost time:{:.4f} min, ETA:{:.4f}".format(
                    epoch_train_loss,
                    epoch_test_loss,
                    (datetime.datetime.now() - start_time).seconds / 60,
                    (n_epochs - 1 - epoch) * (datetime.datetime.now() - start_time).seconds / 60,
                )
            )
            if epoch % 10 == 0:
                for image, mask in data_loader_test:
                    mask = mask[0].to(device)
                    output = self.predict_batch(image)
                    break
                display_progress(image[0], mask[0][0].unsqueeze(0), output['mask'][0], edge = output['edge'][0], current_epoch = epoch, save = True, save_path = "./save/" + self.name + "/")
            if (epoch <= 4):
                continue

            if (epoch_test_loss < best_loss):
                best_loss = epoch_test_loss
                es = 0
                self.save_parameter("./save_best/", "best")
            else:
                es += 1
                print("Counter {} of 10".format(es))
                if es > 50:
                   
                    print("Early stopping with best_loss: ", best_loss, "and val_acc for this epoch: ", epoch_test_loss,
                            "...")
                        
                    break
        self.save_history()
        self.save_parameter()

    def test(self, data_loader_test, need_score = False):
        self.model.eval()
        running_loss = 0
        test_index = 0
        iou = []
        dice = []
        hd = []
        nr = []
        pr = []
        with torch.no_grad():
            for data in data_loader_test:
                X_test, y_test = data
                y_gt = y_test[0]
                y_edge = y_test[1]
                X_test, y_gt, y_edge = Variable(X_test).float(), Variable(y_gt), Variable(y_edge)
                if (use_gpu):
                    X_test = X_test.to(device)
                    y_gt = y_gt.to(device)
                    y_edge = y_edge.to(device)

                if self.split:
                    w = 2
                    X_test = crop_tensor(X_test, w, 2*w, axis = 0)
                    y_gt = crop_tensor(y_gt, w, 2*w, axis = 0)
                    y_edge = crop_tensor(y_edge, w, 2*w, axis = 0)
                # print(X_test.shape)

                X_test, level_set = self.data_warper(X_test, y_gt)
                outputs = self.model(X_test)

                loss1 = self.cost(outputs["mask"], y_gt)
                loss_dice_1 = diceCoeffv2( outputs["mask"], y_gt )
                #loss_level_set = self.l1(outputs["levelset"], level_set)
                #loss2 = self.cost(outputs["edge"], y_edge)

                loss = 0.4 * loss_dice_1 + 0.3 * loss1 # + 0.2 * loss2 + loss_level_set 

                running_loss += loss.data.item()
                test_index += 1
                if need_score:
                    gt = y_gt.cpu()
                    mask = outputs["mask"].detach().cpu()

                    for i in range(len(mask)):
                        pred = np.array(normalized( mask[i].squeeze().numpy()) * 255, dtype=np.uint8)
                        label = np.array(normalized( gt[i].squeeze().numpy()) * 255, dtype=np.uint8)
                        score_dice, score_iou, score_hd = evaluateof2d(pred, label)
                        iou.append(score_iou)
                        dice.append(score_dice)
                        hd.append(score_hd)

                    nr.append(negative_recall(gt, mask))
                    pr.append(positive_recall(gt, mask))
        # mean_iou_np(), mean_dice_np(), positive_recall(), negative_recall()
        if need_score:
            score = 0.4* np.mean( dice ) + 0.3* np.mean( iou ) + 0.3* np.mean( hd )
            log_str = "SCORE:{:.4f}, DICE:{:.4f}, IOU:{:.4f}, HD:{:.4f}, NR:{:.4f}, PR:{:.4f}\n".format(
               score, np.mean( dice ), np.mean( iou ),np.mean( hd ),  np.mean(nr), np.mean(pr)
            )
            self.history_score.append(log_str)
            print( log_str )

        epoch_loss = running_loss / (test_index + 1)
        return epoch_loss

    def train(self, data_loader_train):
        self.model.train()
        train_index = 0
        running_loss = 0.0

        for data in data_loader_train:
            X_train, y_train = data
            y_gt = y_train[0]
            y_edge = y_train[1]
            X_train, y_gt, y_edge = Variable(X_train).float(), Variable(y_gt), Variable(y_edge)
            if (use_gpu):
                X_train = X_train.to(device)
                y_gt = y_gt.to(device)
                y_edge = y_edge.to(device)

            if self.split:
                w = 2
                X_train = crop_tensor(X_train, w, 2*w, axis = 0)
                y_gt = crop_tensor(y_gt, w, 2*w, axis = 0)
                y_edge = crop_tensor(y_edge, w, 2*w, axis = 0)
            # print("训练中 train {}".format(X_train.shape))
            self.optimizer.zero_grad()
            X_train, level_set = self.data_warper(X_train, y_gt)
            outputs = self.model(X_train)
            loss1 = self.cost(outputs["mask"], y_gt)
            loss2 = self.l1(outputs["levelset"], level_set)
            loss_dice_1 = diceCoeffv2( outputs["mask"], y_gt )
            loss_cross = self.encropy_cost(outputs["mask"], y_gt)
           
            # loss_hd =  0.5 * loss_hd_1 + 0.5 * loss_hd_3
            # loss_dice = loss_dice_1 + 0.3 * loss1
            # loss_iou = 0.5 * loss1 # + 0.5 * loss3 
            # if loss_dice.item() < 0.1: 
            #     # loss_iou += 0.1 * loss2
            #     loss_dice += 0.1 * loss_dice_2
                # loss_hd += 0.1 * loss_hd_2

            loss = 0.5 * loss1 + 0.5 * loss2 + 0.3 * loss_dice_1 + 0.5 * loss_cross 
            if loss1 < 0.1:
                loss += 0.1 * self.cost(outputs["edge"], y_edge) + self.encropy_cost(outputs["edge"], y_edge)
            # super_gt = y_gt
            # for index in range(len(supervision) - 1):
            #     super_gt = self.downsample(super_gt)
            #     # print( super_gt.shape, supervision[ len(supervision) - index - 1 ].shape )
            #     loss += diceCoeffv2( supervision[ len(supervision) - index - 1 ], super_gt)
            
            # loss = loss.float()
            # with torch.autograd.set_detect_anomaly(True):
            loss.backward()
            self.optimizer.step()

            running_loss += loss.data.item()
            train_index += 1

        epoch_train_loss = running_loss / train_index
        return epoch_train_loss

    def predict_batch(self, image):
        if (type(image) == np.ndarray):
            image = torch.from_numpy(image)
        if (len(image.size()) == 3):
            image.unsqueeze(1)
        self.model.eval()
        with torch.no_grad():
            image = Variable(image).float()
            if (use_gpu):
                image = image.to(device)
            # print(image.shape)
            image = self.data_warper.clahe(image)
            output = self.model(image)
        return output
        # return output

    def save_history(self, file_path='./save/'):
        file_path = file_path + self.name + "/"
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        fo = open(file_path + "loss_history.txt", "w+")
        fo.write(str(self.history_loss))
        fo.close()
        fo = open(file_path + "acc_history.txt", "w+")
        fo.write(str(self.history_acc))
        fo.close()
        fo = open(file_path + "loss_test_history.txt", "w+")
        fo.write(str(self.history_test_loss))
        fo.close()
        fo = open(file_path + "test_history.txt", "w+")
        fo.write(str(self.history_test_acc))
        fo.close()
        fo = open(file_path + "test_score.txt", "w+")
        fo.write(str(self.history_score))
        fo.close()

    def save_parameter(self, file_path='./save/', name=None):
        file_path = file_path + self.name + "/"
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        if name == None:
            file_path = file_path + "model_" + str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(
                "-", "_").replace(".", "_") + ".pkl"
        else:
            file_path = file_path + name + ".pkl"
        torch.save(obj=self.model.state_dict(), f=file_path)

    def load_parameter(self, file_path='./save/'):
        print(f"load:{file_path}")
        self.model.load_state_dict(torch.load(file_path, map_location = device))


device = "cuda:0"
if __name__ == "__main__":
    batch_size = 12
    image_size = 320
    train_path = r'/data0/lijunlin_data/teech/train/'
   
    All_dataloader = Dataload(
        train_path, 
        image_shape =  (320, 640), #(240, 480), # (320, 640), #(256,256), #(320, 640),
        data_type = "train",
        need_gray = True,
        data_aug = 2,
        )
    
    train_size = int(len(All_dataloader) * 0.8)
    print("size :", train_size)
 
    train_subset = CustomSubset(All_dataloader, np.arange(train_size))
    val_subset = CustomSubset(All_dataloader, np.arange(train_size, len(All_dataloader)), dtype = "val")
    print("train size: {} test size: {} , ".format(len(train_subset), len( val_subset )))
    train_loader = DataLoader(
        dataset = train_subset,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,
    )
    validate_loader = DataLoader(
        dataset = val_subset,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,
    )
    method_dict = {
        0: "Unet",
        1: "RESUNet",
        2: 'RU',
        4: "GTU",
        41:"FL GTU",
        42:"GTU PVT",
        5: "FL",
        8: "FL tiny",
        9: "FL FPN",
        91: "FL FPN 4 8",
        92: "FL DETR ",
    }

    trainer = Train( 
        1, image_size,
        name = "GTU_pvt_mlp",
        method_type = 43,
        is_show = False,
        batch_size = batch_size,
        device_ = "cuda:3",
        split = False,
    )
    print(device)
    # trainer.load_parameter( "./save_best/GTU_pvt/best.pkl" )

    trainer.train_and_test(100, train_loader, validate_loader)


