
import torch
import torch.nn as nn
import pytorch_lightning as pl
from dataloader import Dataload

import os
from torch.utils.data import DataLoader
from utils.score import cal_all_score
from utils.show import display_progress

class DFGAN(pl.LightningModule):

    def __init__(self, in_channels, out_channels, method_type = 0, learning_rate=0.0002, lambda_recon=100, display_step=10, lambda_gp=10, lambda_r1=10, middle_channel =  [64,128,256,512]):
        super().__init__()
        self.save_hyperparameters()
        self.display_step = display_step
        self.nstack = 2
        if method_type == 0:
            self.generator = FLHD(
                in_channel = in_channels,
                channel = 128,
                n_res_block = 2,
                n_res_channel = 128,
                n_coder_blocks = 2,
                embed_dim = 64,
                n_codebooks = 2,
                stride = 2,
                decay = 0.99,
                loss_name = "mse",
                vq_type = "dq",
                beta = 0.25,
                n_hier = [64, 128, 256],
                n_logistic_mix = 10,
            )
        if method_type == 1:
            from model.RESUNet import RESUNet
            self.generator = RESUNet(1,1)
        else:
            raise NotImplementedError
        self.critic = Critic(out_channels)
        self.generator.apply(self._weights_init)
        self.critic.apply(self._weights_init)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
        self.optimizer_C = torch.optim.Adam(self.critic.parameters(), lr=learning_rate, betas=(0.5, 0.9))
        self.scaler_G = torch.cuda.amp.GradScaler(enabled = True )
        self.scaler_C = torch.cuda.amp.GradScaler(enabled = True )
        self.lambda_recon = lambda_recon
        self.lambda_gp = lambda_gp
        self.lambda_r1 = lambda_r1
        self.recon_criterion = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.generator_losses, self.critic_losses  = [],[]
        self.save_path = "./save/FLHD/"
        # self.cal_score = Cal_Score(batch_size)
    
    def _weights_init(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def configure_optimizers(self):
        return [self.optimizer_C, self.optimizer_G]
        
    def generator_step(self, real_images, conditioned_images):
        # WGAN has only a reconstruction loss
        self.optimizer_G.zero_grad()

        conditioned_images = conditioned_images.to(torch.float16)
        real_images_mask = real_images[0].to(torch.float16)
        real_images_edge = real_images[1].to(torch.float16)

        real_images = real_images_mask

        with torch.cuda.amp.autocast(enabled=True):
            fake_images, great_image = self.generator(conditioned_images)
        
        # recon_loss = 0
        # beta = (self.nstack + 1) * self.nstack / 2
        # for i in range(self.nstack):
        #     alpha = (i+1)/self.nstack
        #     # label = alpha * real_images + ( 1 - alpha ) * conditioned_images
        #     recon_loss += ( alpha / beta ) * self.recon_criterion(fake_images[i], real_images)
        # assert False,(great_image.shape, real_images_224.shape)
        recon_loss = self.recon_criterion(fake_images, real_images) + self.recon_criterion(great_image, real_images_edge)

        self.scaler_G.scale(recon_loss).backward()
        self.scaler_G.unscale_(self.optimizer_G)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 2)
        self.scaler_G.step(self.optimizer_G)
        self.scaler_G.update()
        #recon_loss.backward()
        #self.optimizer_G.step()
        
        # Keep track of the average generator loss
        self.generator_losses += [recon_loss.item()/self.nstack]

    def predict_step(self, conditioned_images):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                fake_images,_ = self.generator(conditioned_images)
        return fake_images
    

    def predict_batch(self, image):
        if(type(image) == np.ndarray):
            image = torch.from_numpy(image)
        if(len(image.size()) == 3 ):
            image = image.unsqueeze(0)
        self.generator.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                image = image.to(device)
                image = image.to(torch.float16)
                fake_images,_ = self.generator(image)
        return fake_images.detach().squeeze(0).permute(1,2,0).cpu().numpy()
        
    def critic_step(self, real_images, conditioned_images):
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                conditioned_images = conditioned_images.to(torch.float16)
                fake_images, edge_images , = self.generator(conditioned_images)
            #fake_images = fake_images[-1]
        self.optimizer_C.zero_grad()
        real_images_mask = real_images[0]
        real_images_edge = real_images[1]
        real_images = real_images_mask
        fake_images = fake_images.float()
        fake_logits = self.critic(fake_images)
        real_logits = self.critic(real_images)
            
        # Compute the loss for the critic
        loss_C = real_logits.mean() - fake_logits.mean()

        #Compute the gradient penalty
        alpha = torch.rand(real_images.size(0), 1, 1, 1, requires_grad=True)
        alpha = alpha.to(device)
        interpolated = (alpha * real_images + (1 - alpha) * fake_images.detach()).requires_grad_(True)
        
        interpolated_logits = self.critic(interpolated)
        
        grad_outputs = torch.ones_like(interpolated_logits, dtype=torch.float32, requires_grad=True)
        gradients = torch.autograd.grad(outputs=interpolated_logits, inputs=interpolated, grad_outputs=grad_outputs,create_graph=True, retain_graph=True)[0]

        
        gradients = gradients.view(len(gradients), -1)
        gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        loss_C += self.lambda_gp * gradients_penalty
        
        #Compute the R1 regularization loss
        r1_reg = gradients.pow(2).sum(1).mean()
        loss_C += self.lambda_r1 * r1_reg

        self.scaler_C.scale(loss_C).backward()
        self.scaler_C.unscale_(self.optimizer_C)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 2)
        self.scaler_C.step(self.optimizer_C)
        self.scaler_C.update()

        # Backpropagation
        # loss_C.backward()
        # self.optimizer_C.step()
        self.critic_losses += [loss_C.item()]
        
    def training_step(self, batch, batch_idx, optimizer_idx):
        condition, real = batch
        if optimizer_idx == 0:
            self.critic_step(real, condition)
        elif optimizer_idx == 1:
            self.generator_step(real, condition)
        gen_mean = sum(self.generator_losses[-self.display_step:]) / self.display_step
        crit_mean = sum(self.critic_losses[-self.display_step:]) / self.display_step
        if self.current_epoch % self.display_step==0 and batch_idx==0 and optimizer_idx==1:
            fake, edge, = self.generator(condition)
            # fake = fake[-1].detach()
            torch.save(self.generator.state_dict(), self.save_path +"/ResUnet_"+ str(self.current_epoch) +".pt")
            torch.save(self.critic.state_dict(), self.save_path +"/PatchGAN_"+ str(self.current_epoch) +".pt")
            score_list = cal_all_score(fake, condition)
            log_str = f", score:{score_list[0]}, dice:{score_list[1]}, iou:{score_list[2]}, hausdorff:{score_list[3]}"
            print(
                f"Epoch {self.current_epoch} : Generator loss: {gen_mean},  Critic loss: {crit_mean} " + log_str
            )
            real = real[0]
            display_progress(condition[0], real[0], fake[0], edge[0], self.current_epoch, (20,15),True, self.save_path)
        
    # def validation_step(self, batch, batch_idx):
    #     condition, real = batch
    #     fake = self.generator(condition)
    #     self.cal_score.update(fake, real)
    #     if self.current_epoch % self.display_step==0 and batch_idx==0:
    #         display_progress(condition[0], real[0], fake[0], self.current_epoch, (20,15),True, self.save_path + '/val/')
    #         torch.save(self.generator.state_dict(), self.save_path +"/val/ResUnet_"+ str(self.current_epoch) +".pt")
    #         torch.save(self.critic.state_dict(), self.save_path +"/val/PatchGAN_"+ str(self.current_epoch) +".pt")
    #         fid_avg, cdc_avg = self.cal_score.get_score()
    #         str_log = f"Epoch {self.current_epoch} : FID score: {fid_avg}, CDC score: {cdc_avg}"
    #         self.log('val',str_log)
    #         print(str_log)

def predict_from_dataloader(test_loader, cwgan, save_path = './image/'):
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            real, condition = batch
            condition = condition.to(torch.float16)
            condition = condition.to(device)
            pred = cwgan.generator(condition).detach().squeeze().permute(1, 2, 0)
            pred = np.array((pred - np.min(pred))/(np.max(pred) - np.min(pred)) * 255, dtype = np.uint8)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            cv2.imwrite("{}/f{:03}.png".format(save_path, batch_idx), pred)
        
def vis(folder_path, save_path = r'E:\Data\Frame\test_frames\result_ADQAE/'):
    image_size = 224
    train_path = r'E:\Data\Frame\train\dark\frames'
    label_path = r'E:\Data\Frame\train\src\frames'

    cwgan = DFGAN(in_channels = 3, out_channels = 3 ,learning_rate=2e-4, lambda_recon=100, display_step=5, middle_channel= [32,64,128,256])
    cwgan = cwgan.to('cuda:0')
    cwgan.generator.load_state_dict(torch.load('./save/dqgan/val/ResUnet_45.pt'))
    cwgan.critic.load_state_dict(torch.load('./save/dqgan/val/PatchGAN_45.pt'))
    
    dataloader = Dataload(train_path, label_path, image_shape = (image_size, image_size))
    dir_dict = dataloader.load_data_new(folder_path)
    dataloader.create_dir(save_path)
    print(len(dir_dict))
    for middle_file_name, image_path_list in dir_dict.items():
        save_path_folder = save_path + middle_file_name
        print(save_path_folder)
        dataloader.create_dir(save_path_folder)
        for index in range(len(image_path_list)):
            image_path = image_path_list[index]
            image = dataloader.read_image_data( image_path['image'] )
            image = dataloader.datagen(image)
            pred = cwgan.predict_batch(image)
            #print(pred.shape)
            pred = np.array((pred - np.min(pred))/(np.max(pred) - np.min(pred)) * 255, dtype = np.uint8)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            cv2.imwrite("{}/f{:03}.png".format(save_path_folder, index), pred)


def train():
    batch_size = 32
    image_size = 256
    train_path = r'/data0/lijunlin_data/teech/train/'
    
    All_dataloader = Dataload(
        train_path, 
        image_shape = (image_size, image_size),
        data_type = "train"
    )
    train_size = int(len(All_dataloader.photo_set) * 0.8)
    validate_size = len(All_dataloader.photo_set) - train_size

    train_dataset, validate_dataset = torch.utils.data.random_split(
        All_dataloader, 
        [train_size, validate_size],
    )
    print("train :{} test :{} , ".format(train_size, validate_size))

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

    cwgan = DFGAN(
        method_type = 1,
        in_channels = 1, 
        out_channels = 1,
        learning_rate=2e-5, 
        lambda_recon=100, 
        display_step = 10, 
        middle_channel= [32,64,128,256]
    )
    cwgan.save_path = "./save/resUnet/"
    print(cwgan.save_path)
    if not os.path.exists(cwgan.save_path):
        os.mkdir(cwgan.save_path)
    if not os.path.exists(cwgan.save_path + 'val'):
        os.mkdir(cwgan.save_path + 'val')
    trainer = pl.Trainer(max_epochs=51, gpus = [int(device[-1])])
    #cwgan.generator.load_state_dict(torch.load('./save/ResUnet_90.pt'))
    #cwgan.critic.load_state_dict(torch.load('./save/PatchGAN_90.pt'))
    trainer.fit(cwgan, train_loader, validate_loader)
    torch.save(cwgan.generator.state_dict(), cwgan.save_path +"ResUnet_END.pt")
    torch.save(cwgan.critic.state_dict(), cwgan.save_path  +"PatchGAN_END.pt")


def train_coco():
    batch_size = 16
    image_size = 224
    train_path = r'/data0/lijunlin_data/train2017/'
    label_path = '' #r'E:\Data\Frame\train\src\frames'
    #All_dataloader = Dataload(r'H:\DATASET\COLORDATA\train\train_frame', r'H:\DATASET\COLORDATA\train_gt\train_gt_frame')
    All_dataloader = Dataload(train_path,label_path, image_shape = (image_size, image_size), data_type='coco')
    train_size = int(len(All_dataloader.photo_set) * 0.1)
    validate_size = len(All_dataloader.photo_set) - train_size

    train_dataset, validate_dataset = torch.utils.data.random_split(All_dataloader
                                                                    , [train_size, validate_size])
    print("train {} test {} , ".format(train_size, validate_size))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    validate_loader = DataLoader(
        dataset=validate_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    cwgan = DFGAN(
        in_channels = 3, 
        out_channels = 3 ,
        learning_rate=2e-4, 
        lambda_recon=100, 
        display_step=5, 
        middle_channel= [32,64,128,256]
    )
    cwgan.save_path = "./save/PHDAE/"
    if not os.path.exists(cwgan.save_path):
        os.mkdir(cwgan.save_path)
    if not os.path.exists(cwgan.save_path + 'val'):
        os.mkdir(cwgan.save_path + 'val')
    trainer = pl.Trainer(max_epochs=71, gpus = [int(device[-1])])
    #cwgan.generator.load_state_dict(torch.load('/home/lijunlin/Project/AIColor/save/dgan_256_256/ResUnet_END.pt'))
    #cwgan.critic.load_state_dict(torch.load('/home/lijunlin/Project/AIColor/save/dgan_256_256/PatchGAN_END.pt'))
    trainer.fit(cwgan, train_loader, validate_loader)
    torch.save(cwgan.generator.state_dict(), cwgan.save_path +"ResUnet_END.pt")
    torch.save(cwgan.critic.state_dict(), cwgan.save_path  +"PatchGAN_END.pt")
    
device = 'cuda:0'
if __name__ == '__main__':
    # file_path = r'E:\Data\Frame\test_frames\test'
    # save_path = r'E:\Data\Frame\test_frames\result_ADQAE/'
    # vis(file_path, save_path)
    train()
    #train_coco()
    
