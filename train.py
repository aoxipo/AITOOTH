from dataloader import Dataload
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

from utis.adploss import AdpLoss
from utils.score import cal_all_score
from utils.show import display_progress

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
    def __init__(self, in_channles, image_size = 320, name = 'dense', method_type = 0, is_show = True):
        self.in_channels = in_channles

        self.image_size = image_size
        self.name = name
        self.method_type = method_type
        self.lr = 0.0001
        self.history_acc = []
        self.history_loss = []
        self.history_test_acc = []
        self.history_test_loss = []
        self.create(is_show)

    def create(self, is_show):

        if (self.method_type == 0):
            from model.model import Unet as Model
            self.model = Model()
            print("build Unet model")
            
        elif self.method_type == 1:
            from model.RESUNet import RESUNet as Model
            self.model = Model()
            print("build RESUNet model")
        elif self.method_type == 2:
            from model.model import RUnet as Model
            self.model = Model()
            print("build RU model")
        else:
            raise NotImplementedError
        
        self.cost = torch.nn.MSELoss()
        if (use_gpu):
            self.model = self.model.to(device)
            self.cost = self.cost.to(device)
        if (is_show):
            summary(self.model, (self.in_channels, self.image_size * 2, self.image_size))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.adp = AdpLoss()

    def train_and_test(self, n_epochs, data_loader_train, data_loader_test):
        best_loss = 1000000
        es = 0

        for epoch in range(n_epochs):
            start_time = datetime.datetime.now()
            print("Epoch {}/{}".format(epoch, n_epochs))
            print("-" * 10)

            epoch_train_loss = self.train(data_loader_train)
            epoch_test_loss = self.test(data_loader_test)

            self.history_acc.append(0)
            self.history_loss.append(epoch_train_loss)
            self.history_test_acc.append(0)
            self.history_test_loss.append(epoch_test_loss)
            print(
                "Loss is:{:.4f}\nLoss is:{:.4f}\ncost time:{:.4f} min, ETA:{:.4f}".format(
                    epoch_train_loss,
                    epoch_test_loss,
                    (datetime.datetime.now() - start_time).seconds / 60,
                    (n_epochs - 1 - epoch) * (datetime.datetime.now() - start_time).seconds / 60,
                )
            )
            if epoch % 10 == 0:
                for image, mask in data_loader_test:
                    mask = mask[0].to(device)
                    result1, result2 = self.predict_batch(image)
                    break
                display_progress(image[0], mask[0][0].unsqueeze(0), result1[0], current_epoch=epoch, save=True, save_path='perdict')
            if (epoch <= 4):
                continue
            if (epoch_test_loss < best_loss):
                best_loss = epoch_test_loss
                es = 0
                self.save_parameter("./save_best/", "best")
            else:
                es += 1
                print("Counter {} of 10".format(es))
                if es > 10:
                    print("Early stopping with best_loss: ", best_loss, "and val_acc for this epoch: ", epoch_test_loss,
                          "...")
                    break
        self.save_history()
        self.save_parameter()

    def test(self, data_loader_test):
        self.model.eval()
        running_loss = 0
        test_index = 0
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

                outputs = self.model(X_test)

                loss1 = self.cost(outputs["mask"], y_gt)
                loss2 = self.cost(outputs["edge"], y_edge)
                loss = loss2 + loss1

                running_loss += loss.data.item()
                test_index += 1

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
            # print("训练中 train {}".format(X_train.shape))
            self.optimizer.zero_grad()

            outputs = self.model(X_train)
            loss1 = self.cost(outputs["mask"], y_gt)
            loss2 = self.cost(outputs["edge"], y_edge)
            loss = self.adp(loss_main = loss1, loss_other = loss2)
            # loss = loss2 + loss1
            # loss = loss.float()
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
            print(image.shape)
            output = self.model(image)
        return output['mask'], output['edge']
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
        self.model.load_state_dict(torch.load(file_path))


def trainTest():
    batch_size = 32
    image_size = 224
    data_path = r'/data0/lijunlin_data/teech/train/'

    All_dataloader = Dataload(data_path)

    train_size = int(All_dataloader.__len__() * 0.8)
    validate_size = All_dataloader.__len__() - train_size

    train_dataset, validate_dataset = torch.utils.data.random_split(All_dataloader
                                                                    , [train_size, validate_size])

    print("训练集大小: {} 测试集大小: {} , ".format(train_size, validate_size))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )
    validate_loader = DataLoader(
        dataset=validate_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    trainer = Train(3, 8, image_size, False)

    # trainer =  Train(3,25,image_size,False)
    # print(len(train_loader), len(test_loader))
    print("开始训练")
    trainer.train(train_loader)
    # trainer.train_and_test(100, train_loader, validate_loader)
    # trainer.test(validate_loader)


def perdit():
    trainer = Train(1, 128, False)
    trainer.load_parameter(r"F:\PyTestProject\AliProject\save_best\SCSUNet_D\best.pkl")
    img = torch.ones([3, 128, 128])
    res = trainer.predict_batch(img)
    print(res)


if __name__ == "__main__":
    batch_size = 32
    image_size = 224
    train_path = r'/data0/lijunlin_data/teech/train/'
    # label_path = r'E:\Data\Frame\train\src\frames'
    # All_dataloader = Dataload(r'H:\DATASET\COLORDATA\train\train_frame', r'H:\DATASET\COLORDATA\train_gt\train_gt_frame')
    # All_dataloader = Dataload(train_path, label_path, image_shape=(image_size, image_size))

    All_dataloader = Dataload(train_path)
    train_size = int(All_dataloader.__len__() * 0.8)
    validate_size = All_dataloader.__len__() - train_size

    train_dataset, validate_dataset = torch.utils.data.random_split(All_dataloader
                                                                    , [train_size, validate_size])

    print("训练集大小: {} 测试集大小: {} , ".format(train_size, validate_size))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    validate_loader = DataLoader(
        dataset=validate_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    method_dict = {
        0: "densecoord",
        1: "mixfpn",
        2: 'SCSUNet',
        3: "SCSUNet_D",
    }

    trainer = Train(
        1, image_size,
        name="UNet",
        method_type=0,
        is_show=False
    )
    trainer.train_and_test(100, train_loader, validate_loader)

    # perdit()
