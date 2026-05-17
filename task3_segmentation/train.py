import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,random_split,Dataset
import wandb
import numpy as np
import random
from PIL import Image
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
seed_everything(42)
class TransformSubset(Dataset):
    def __init__(self,subset,transform=None,target_transform=None):
        self.subset=subset
        self.transform=transform
        self.target_transform=target_transform
    def __getitem__(self,idx):
        img,mask=self.subset[idx]
        if self.transform!=None:
            img=self.transform(img)
        if self.target_transform!=None:
            mask=self.target_transform(mask)
        return img,mask
    def __len__(self):
        return len(self.subset)
class DiceLoss(nn.Module):
    def forward(self,pred,target):
        pred=torch.softmax(pred,dim=1)
        target_onehot=nn.functional.one_hot(target,num_classes=3).permute(0,3,1,2).float()
        inter=(pred*target_onehot).sum(dim=(0,2,3))
        union=pred.sum(dim=(0,2,3))+target_onehot.sum(dim=(0,2,3))
        return 1-(2.*inter.sum()+1e-6)/(union.sum()+1e-6)
class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(in_ch,out_ch,3,padding=1),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True),nn.Conv2d(out_ch,out_ch,3,padding=1),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True))
    def forward(self,x):
        return self.conv(x)
class UNet(nn.Module):
    def __init__(self,n_classes=3):
        super().__init__()
        self.pool=nn.MaxPool2d(2)
        self.e1=DoubleConv(3,64)
        self.e2=DoubleConv(64,128)
        self.e3=DoubleConv(128,256)
        self.e4=DoubleConv(256,512)
        self.b=DoubleConv(512,1024)
        self.up4=nn.ConvTranspose2d(1024,512,2,2)
        self.d4=DoubleConv(1024,512)
        self.up3=nn.ConvTranspose2d(512,256,2,2)
        self.d3=DoubleConv(512,256)
        self.up2=nn.ConvTranspose2d(256,128,2,2)
        self.d2=DoubleConv(256,128)
        self.up1=nn.ConvTranspose2d(128,64,2,2)
        self.d1=DoubleConv(128,64)
        self.out=nn.Conv2d(64,n_classes,1)
    def forward(self,x):
        e1=self.e1(x)
        e2=self.e2(self.pool(e1))
        e3=self.e3(self.pool(e2))
        e4=self.e4(self.pool(e3))
        b=self.b(self.pool(e4))
        d4=self.d4(torch.cat([self.up4(b),e4],1))
        d3=self.d3(torch.cat([self.up3(d4),e3],1))
        d2=self.d2(torch.cat([self.up2(d3),e2],1))
        d1=self.d1(torch.cat([self.up1(d2),e1],1))
        return self.out(d1)
def compute_miou(pred,target,n_classes=3):
    pred=torch.argmax(pred,dim=1)
    iou=[]
    for cls in range(n_classes):
        p=(pred==cls)
        t=(target==cls)
        inter=(p&t).sum().float()
        union=(p|t).sum().float()
        iou.append(inter/(union+1e-6))
    return torch.mean(torch.tensor(iou)).item()
def train(model,loader,ce,dice,loss_type,opt,device):
    model.train()
    loss_sum,miou_sum=0.,0.
    for img,mask in loader:
        img,mask=img.to(device),mask.to(device).long()
        opt.zero_grad()
        pred=model(img)
        if loss_type=="CE":
            loss=ce(pred,mask)
        elif loss_type=="DICE":
            loss=dice(pred,mask)
        else:
            loss=ce(pred,mask)+dice(pred,mask)
        loss.backward()
        opt.step()
        loss_sum+=loss.item()
        miou_sum+=compute_miou(pred,mask)
    return loss_sum/len(loader),miou_sum/len(loader)
def val(model,loader,ce,dice,loss_type,device):
    model.eval()
    loss_sum,miou_sum=0.,0.
    with torch.no_grad():
        for img,mask in loader:
            img,mask=img.to(device),mask.to(device).long()
            pred=model(img)
            if loss_type=="CE":
                loss=ce(pred,mask)
            elif loss_type=="DICE":
                loss=dice(pred,mask)
            else:
                loss=ce(pred,mask)+dice(pred,mask)
            loss_sum+=loss.item()
            miou_sum+=compute_miou(pred,mask)
    return loss_sum/len(loader),miou_sum/len(loader)
if __name__=="__main__":
    device=torch.device("cuda"if torch.cuda.is_available()else"cpu")
    img_size=224
    batch_size=16
    epochs=100
    LOSS_TYPE="CE_DICE"
    train_transform=transforms.Compose([transforms.Resize((256,256)),transforms.RandomResizedCrop(img_size,scale=(0.8,1.0)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    val_transform=transforms.Compose([transforms.Resize((img_size,img_size)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    def mask_trans(pic):
        pic=pic.resize((img_size,img_size),Image.NEAREST)
        return torch.from_numpy(np.array(pic)).long()-1
    full=datasets.OxfordIIITPet(root="../data/oxford-iiit-pet",split="trainval",target_types="segmentation",download=False)
    train_size=int(0.9*len(full))
    val_size=len(full)-train_size
    train_sub,val_sub=random_split(full,[train_size,val_size])
    train_set=TransformSubset(train_sub,train_transform,mask_trans)
    val_set=TransformSubset(val_sub,val_transform,mask_trans)
    test_set=TransformSubset(datasets.OxfordIIITPet(root="../data/oxford-iiit-pet",split="test",target_types="segmentation",download=False),val_transform,mask_trans)
    train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=2,pin_memory=True)
    val_loader=DataLoader(val_set,batch_size=batch_size,shuffle=False,num_workers=2,pin_memory=True)
    test_loader=DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=2,pin_memory=True)
    model=UNet().to(device)
    ce=nn.CrossEntropyLoss()
    dice=DiceLoss()
    opt=optim.Adam(model.parameters(),lr=1e-4)
    ckpt_path=f"ckpt_{LOSS_TYPE}.pth"
    wandb.init(project="pet-seg",name=f"task3-{LOSS_TYPE}",resume="allow",id=wandb.util.generate_id(),mode="offline")
    start_epoch=0
    best_miou=0.
    if os.path.exists(ckpt_path):
        ckpt=torch.load(ckpt_path,map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_epoch=ckpt["epoch"]+1
        best_miou=ckpt["best_miou"]
    for epoch in range(start_epoch,epochs):
        train_loss,train_miou=train(model,train_loader,ce,dice,LOSS_TYPE,opt,device)
        val_loss,val_miou=val(model,val_loader,ce,dice,LOSS_TYPE,device)
        wandb.log({"epoch":epoch+1,"train_loss":train_loss,"train_miou":train_miou,"val_loss":val_loss,"val_miou":val_miou})
        if val_miou>best_miou:
            best_miou=val_miou
            torch.save(model.state_dict(),f"best_{LOSS_TYPE}.pth")
        torch.save({"epoch":epoch,"model":model.state_dict(),"opt":opt.state_dict(),"best_miou":best_miou},ckpt_path)
    model.load_state_dict(torch.load(f"best_{LOSS_TYPE}.pth"))
    test_miou=val(model,test_loader,ce,dice,LOSS_TYPE,device)[1]
    wandb.log({"test_miou":test_miou})
    print(f"Final test mIoU:{test_miou:.4f}")
    wandb.finish()
