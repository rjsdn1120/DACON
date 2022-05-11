#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import *
import numpy as np
import random
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm
import timm
import albumentations as A
from sklearn.model_selection import KFold
import cv2
import os
from glob import glob
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
import argparse

# In[2]:


from torch.utils.data.sampler import SubsetRandomSampler


# In[3]:


random_seed = 1120
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


# In[4]:

parser = argparse.ArgumentParser()

parser.add_argument('-lr')
parser.add_argument('-gpu')

args=parser.parse_args()


# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.GaussianBlur(7),
#     transforms.RandomHorizontalFlip(p=0.2),
#     transforms.RandomVerticalFlip(p=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,0.5,0.5),(0.224,0.224,0.224))
# ])


transform = A.Compose([
    A.Resize(224,224),
    A.Blur(),
     A.GaussNoise(),
     A.HorizontalFlip(p=0.2),
     A.ColorJitter(),
    A.Normalize(),
    ToTensorV2(),
])
valt=A.Compose([
  A.Resize(224,224),
    A.Normalize(),
    ToTensorV2(),
])

# RandomResizeCrop
# Random GaussianNoise
# color jitter -> b/s/h/c  0.2 ~ 3  ||  128
# RandomHorzital


# In[5]:


cnt=0
ctoi={k : idx for idx,k in enumerate(os.listdir('/media/data2/rjsdn/dacon/basic_image_classification/train'))}
itoc = {v: k for k, v in ctoi.items()}


# In[6]:


ctoi


# In[7]:


class Mydataset(Dataset):
    def __init__(self,path,transform=None):
        super(Mydataset).__init__()
        self.path=path
        self.transform=transform
        
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self,idx):
        img = self.path[idx]
        label = img.split('/')[-2]
        label = ctoi[label]
        
        img = cv2.imread(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        img = transform(image=img)['image']
            
        return img,label
        
        


# In[8]:


def lite_save(state, epoch, save_dir, model):
    os.makedirs(save_dir, exist_ok=True)
    
    target_path = f'{save_dir}/{state}.path.tar'
    
    with open(target_path, "wb") as f:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),}, f)


# In[9]:



# learning rate scheduler -> StepLR || Cosine Annealing

# optimzier Adam || AdamW || AadmP || AdamB ? 

# Regularizer -> 써보고


# In[10]:


def train(train_param):
    res = {
        'train_acc' :[],
        'train_loss':[],
        'val_loss':[],
        'val_acc': []
    }


    trainloader = train_param['trainloader']
    valloader = train_param['valloader']
    model = train_param['model']
    opt = train_param['opt']
    criterion = train_param['criterion']
    scheduler = train_param['scheduler']
    SAVE_PATH = train_param['SAVE_PATH']
    
    scaler=torch.cuda.amp.GradScaler()
    
    best_acc = 0
    for epoch in range(epochs):
        train_loss=0
        train_acc=0 
        train_size=0
        loader_size = len(trainloader)
        model.train()
        for data in tqdm(trainloader):
            x = data[0].to(device)
            y = data[1].to(device)
            train_size+=len(y)

            opt.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(x)
                loss = criterion(output,y)
#                 loss.backward()
#                 opt.step()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            _,output = output.max(1)
            train_acc += (output==y).sum().item()
            train_loss += loss.item()
        

        train_acc /= train_size
        train_loss /= loader_size
        print(f'[{epoch+1}/{epochs}], train_acc={train_acc:.2f} train_loss={train_loss:.2f}')

        val_acc =0
        cnt = 0
        val_loss=0
        val_size=0
        loader_size = len(valloader)
        model.eval()
        with torch.no_grad():
            for data in tqdm(valloader):
                x = data[0].to(device)
                y = data[1].to(device)
                val_size+=len(y)
                output = model(x)
                loss = criterion(output,y)
                output = torch.argmax(output,1)
                val_loss+=loss.item()
                val_acc += (output==y).sum().item()

        val_acc /= val_size
        val_loss /= loader_size
        
        scheduler.step(val_loss)
        
        print(f'val acc={val_acc}, val_los={val_loss}')
        print(f'\t Current valid Acc | BEST Acc: [{val_acc}| {best_acc}]')
        if val_acc > best_acc:
            print(f'\t Best Acc changed [{best_acc} ---> {val_acc}]')
            best_acc = val_acc

            lite_save('best', epoch, SAVE_PATH, model)
        lite_save('last', epoch, SAVE_PATH, model)

        res['train_loss'].append(train_loss)
        res['train_acc'].append(train_acc)
        res['val_loss'].append(val_loss)
        res['val_acc'].append(val_acc)
        
        
    import pandas as pd    
    pd.DataFrame(res).to_csv(SAVE_PATH+'/res.csv', index=False)


# In[11]:


train_path = '/media/data2/rjsdn/dacon/basic_image_classification/train/'
trainlist = [y for x in os.walk(train_path) for y in sorted(glob(os.path.join(x[0],'*.jpg')))]
cv=KFold(n_splits=5,random_state=1120,shuffle=True)


# In[12]:


trainlist[20000]


# In[13]:


BATCHSIZE=64
lr=float(args.lr)
device = torch.device('cuda:0')
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu


# In[30]:



# In[31]:


idx=0
epochs=50
random.shuffle(trainlist)
for t,v in cv.split(trainlist):
    idx+=1
    print('Fold :',idx)
    trainset = Mydataset(trainlist,transform)
    valset = Mydataset(trainlist,valt)
    
    trainloader = DataLoader(trainset,batch_size=BATCHSIZE,num_workers=4,sampler=t)
    valloader = DataLoader(valset,batch_size=BATCHSIZE,num_workers=4,sampler=v)
    
    model_name = 'efficientnet_b3'
    
    model=timm.create_model(model_name,num_classes=10)   # Efficient-noisystudent
    model.to(device)
    criterion=nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-6)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=lr/1e-1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode = 'min', patience = 2, factor = 0.5, min_lr = 5e-5)
    
    MODEL=f'{model_name}_224_a_{idx}'
    SAVE_PATH = f'weights/{MODEL}/'
    os.makedirs(SAVE_PATH,exist_ok=True)
    
    train_param={
        'trainloader':trainloader,
        'valloader':valloader,
        'model':model,
        'opt':opt,
        'criterion':criterion,
        'scheduler':scheduler,
        'SAVE_PATH':SAVE_PATH
    }
    
    train(train_param)
    


# In[ ]:




