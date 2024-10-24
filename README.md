# DDPM evolution in LevelSet Function
see [./assert/ddpm_evolution_boundar.gif](https://github.com/aoxipo/AITOOTH/blob/main/assert/ddpm_evolution_boundar.gif)
![./assert/ddpm_evolution_boundar.gif](https://github.com/aoxipo/AITOOTH/blob/main/assert/ddpm_evolution_boundar.gif)

# AITOOTH
AITOOTH is our code for MICCAI 2023 Challenges ：STS-基于2D 全景图像的牙齿分割任务
Our paper "Diffusion-Based Conv-Former Dual-Encode U-Net: DDPM for Level Set Evolution Mapping - MICCAI STS 2023 Challenge"
are onlined at ![https://link.springer.com/chapter/10.1007/978-3-031-72396-4_8] 
![https://github.com/aoxipo/AITOOTH/blob/main/assert/DDPM.png](https://github.com/aoxipo/AITOOTH/blob/main/assert/DDPM.png)

# Good Case
![good case](https://github.com/aoxipo/AITOOTH/blob/main/assert/img1.png)

# Model

1. We propose the CMT module, which can learn the potential features of dif-
   ferent distributions and frequencies in the features, and extract and reconstruct
   them
2. We propose a boundary learning model based on DDPM, where the level
   set is constructed by predicting the level set function of the projection in the high
   latitude space at the boundary. Subsequently, we employ the DDPM model to
   find the optimal zero level set, which enabling us to effectively learn all potential
   boundary features and select an appropriate partition boundary as our final
   outcome

# Train
note: if you just want to add boundary to your model, by this way [boundary module helper](./boundry/README.md)
### Modify configuration file

```python
batch_size = 12
image_size = 320
train_path = r'/data0/lijunlin_data/teech/train/'
# replace your dataset path here
All_dataloader = Dataload(
    train_path, 
    image_shape =  (320, 640), #(240, 480), # (320, 640), #(256,256), #(320, 640),
    data_type = "train",
    need_gray = True,
    data_aug = 1,
    )
# set right image shape
```

### definition model 1 

```python
method_dict = {
        0: "Unet",
        4: "GTU",
        43:"DCMTDUNet",
       	44:"DCMTDUNet_boundry",
    }
trainer = Train( 
        1, image_size,
        name = "DCMTDUNet_boundry",
        method_type = 43,
        is_show = False,
        batch_size = batch_size,
        device_ = "cuda:3",
        split = False,
)
print(device)
# trainer.load_parameter( "./save_best/DCMTDUNet_boundry/best.pkl" )
trainer.train_and_test(100, train_loader, validate_loader)
```

#### Run 
```shell
python train.py
```

### definition broundry 

```python
logger =Logger( 
   file_name = f"log_{name}.txt", 
   file_mode = "w+", 
   should_flush = True
)
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
net_model = DCFDU_with_ddpm_boundary(
   1, 1, 
   need_return_dict = True,
   need_supervision = False,
   decode_type = "mlp"
).to(device)
```

#### Run 
```shell
python train_ddpm.py
```

# Inference

```shell
cd ./test
jupyter notebook
% open predict.ipynb to inference the model
```

# Results

![https://github.com/aoxipo/AITOOTH/blob/main/assert/result.png](https://github.com/aoxipo/AITOOTH/blob/main/assert/result.png)

Here is our result in tianchi rank board result and we got 129/839 rank on first board and got 36 on final board, got Top 8 on online board.

# cite 
```
@InProceedings{10.1007/978-3-031-72396-4_8,
author="Li, Junlin
and Tian, Weixin
and Li, Junliang
and He, Yuan
and Ke, Wanglin",
editor="Wang, Yaqi
and Chen, Xiaodiao
and Qian, Dahong
and Ye, Fan
and Wang, Shuai
and Zhang, Hongyuan",
title="Diffusion-Based Conv-Former Dual-Encode U-Net: DDPM for Level Set Evolution Mapping MICCAI STS 2023 Challenge",
booktitle="Semi-supervised Tooth Segmentation",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="83--95",
abstract="We propose the Diffusion-Based Conv-Former U-Net (DCFDU-Net) model for panoramic CT segmentation tasks. Our model primarily employs a dual-encode structure, comprising the CMT and PVT modules. To enhance boundary precision, we incorporated a novel boundary learning module inspired by DDPM and level set. This module constructs the level set function by initially predicting the boundary in a high-dimensional space projection. Then uses the DDPM model to evolve this projection, facilitating the accurate delineation of the zero level set. Finally, the boundaries and mask outcomes are refined through an efficient, cost-effective network architecture. Our method achieved an average DICE score of 91.81{\%} and an average IOU score of 96.35{\%}, with an average HD distance of 0.0332 for teeth segmentation on the validation set using an NVIDIA GeForce RTX 3090 GPU. The average running time was 0.91 s per image. The code is available at https://github.com/aoxipo/AITOOTH.",
isbn="978-3-031-72396-4"
}
```

