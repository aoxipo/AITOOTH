# how to add boundry in your model

# first add module in your model to predict the levelset function
```python
# add a function in your model 
from util import MinPool
    class Unet():
        self.erode = MinPool(2,2,1)
        self.select = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d( 32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d( 16, 1, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.last_decode_levelset = nn.Sequential(
            nn.Conv2d( middle_channel, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size = 1, stride = 1, padding = 0),
            nn.ReLU(),
        )
    ...
    ...
    # your old forward
    def old_forward(x, edge):
        ...
        ...
        out = self.last_decode(outp) #　middle feature
        out_level_set = self.last_decode_levelset(outp)
        return x, out_level_set

    def after_forward(x, edge):
        x = torch.cat([x, edge], 1)
        x = self.select(x)

    def forward(x):
        x, levelset = self.old_forward(x)
        levelset[levelset>0] = 1           # for first train we just use default index or N what ever you like
        edge = self.get_edge(levelset) 
        x = self.after_forward(x, edge)
        return x

    @torch.no_grad()
    def pre_with_boundary_hepler(self, x1):
        x = x1.clone().detach()
        level_set_index = self.ddpmSamplerhelper(x).squeeze()
        B,_,_,_ = x.shape
        for index in range(B):
            x[index][x[index] >= level_set_index[index].item() ] = 1
            x[index][x[index] < level_set_index[index].item()  ] = 0
        edge = self.get_edge(x)
        # cv2.imwrite( './test.png',np.array(edge[0].detach().cpu().squeeze().numpy() * 255, np.uint8)) # show work or not
        return edge
    
    def get_edge(self, x):
        edge = nn.functional.pad(x, (1, 0, 1, 0))
        edge = x - self.erode(edge)
        return edge 
    
    def band_boundary(self, helper):
        self.ddpmSamplerhelper = helper

    @torch.no_grad()
    def inference(self, x):
        x, levelset = self.old_forward(x)
        edge = self.pre_with_boundary_hepler(levelset)
        x = self.after_forward(x, edge)
        return x
```

# second add datawarper for in your train step to generate the levelset label
```python
# just like this
# define the datawarper
from datawarper import DataWarper
self.data_warper = DataWarper()
self.data_warper = self.data_warper.to(device)

# if you dont want to use CLAHE  just use it like _, level_set = self.data_warper(X_train, y_gt)
X_train, level_set = self.data_warper(X_train, y_gt)
outputs = self.model(X_train)
loss1 = self.cost(outputs["mask"], y_gt)
loss2 = self.l1(outputs["levelset"], level_set)
```

# third train the boundary module 
```python
# declare the trainer in train_ddpm.py
from dd import Boundary, GaussianDiffusionTrainer
name = "DCMTDUNet"
logger =Logger( 
    file_name = f"log_{name}.txt", 
    file_mode = "w+", 
    should_flush = True
)
print(device)
save_path = f"./save/{name}/" # for save path
# ddpm
step = 20
# dataset
batch_size = 1000
image_size = 320

########################### declare your dataloader
train_path = r'/T2004100/data/tooth/train/'
All_dataloader = Dataload(
    train_path, 
    image_shape =  (image_size, image_size), #(240, 480), # (320, 640), #(256,256), #(320, 640),
    data_type = "train",
    need_gray = True,
    data_aug = 1,
    )
dataloader = DataLoader(
    dataset = All_dataloader, # YOUR DATALOADER
    batch_size = batch_size,
    shuffle = True,
    drop_last = True,
)
############################

dataWarper = DataWarper().to(  device)
# model setup
net_model = Boundary(1024).to(device)
optimizer = torch.optim.AdamW(
    net_model.parameters(), lr=0.0004, weight_decay=1e-4)
trainer = GaussianDiffusionTrainer( net_model, 1e-4, 0.028, step).to(device)
```
and just run it

```shell
python train_ddpm.py
```
# four add the boundary to your model and load the parameter
```python
from dd import GaussianDiffusionSampler, Boundary
file_path = "./boundary/best.pkl" #your path
b_model = Boundary(1024).to(device)
b_model.load_state_dict(torch.load(file_path, map_location = device))
ddpm_helper = GaussianDiffusionTrainer( b_model, 1e-4, 0.028, step).to(device)

file_path = "./Unet/best.pkl" #your path
model = Unet() # your model
model.load_state_dict(torch.load(file_path, map_location = device))

model.band_boundary(ddpm_helper)
with torch.no_grad():
    image = torch.zeros((1,1,320,320)).cuda()
    result = model.inference(image)
    
```