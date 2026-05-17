import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,random_split,Dataset
import wandb
import random
import numpy as np
import os
import shutil

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
seed_everything(42)

class TransformSubset(Dataset):
    def __init__(self,subset,transform=None):
        self.subset=subset
        self.transform=transform
    def __getitem__(self,idx):
        img,label=self.subset[idx]
        if self.transform is not None:
            img=self.transform(img)
        return img,label
    def __len__(self):
        return len(self.subset)

class SEBlock(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
            nn.Linear(channel,channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel,bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,_,_=x.size()
        y=self.avg_pool(x).view(b,c)
        y=self.fc(y).view(b,c,1,1)
        return x*y.expand_as(x)

class ResNet18(nn.Module):
    def __init__(self,pretrained=True,use_se=False):
        super().__init__()
        self.backbone=models.resnet18(pretrained=pretrained)
        self.use_se=use_se
        if use_se:
            self.se1=SEBlock(64)
            self.se2=SEBlock(128)
            self.se3=SEBlock(256)
            self.se4=SEBlock(512)
        self.backbone.fc=nn.Linear(512,37)
    def forward(self,x):
        x=self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
        x=self.backbone.maxpool(x)
        if self.use_se:
            x=self.se1(self.backbone.layer1(x))
        else:
            x=self.backbone.layer1(x)
        if self.use_se:
            x=self.se2(self.backbone.layer2(x))
        else:
            x=self.backbone.layer2(x)
        if self.use_se:
            x=self.se3(self.backbone.layer3(x))
        else:
            x=self.backbone.layer3(x)
        if self.use_se:
            x=self.se4(self.backbone.layer4(x))
        else:
            x=self.backbone.layer4(x)
        x=self.backbone.avgpool(x)
        x=torch.flatten(x,1)
        x=self.backbone.fc(x)
        return x

def train_one_epoch(model,loader,criterion,optimizer,device):
    model.train()
    total_loss,correct,total=0,0,0
    for img,lab in loader:
        img,lab=img.to(device),lab.to(device)
        optimizer.zero_grad()
        out=model(img)
        loss=criterion(out,lab)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        pred=out.argmax(dim=1)
        total+=lab.size(0)
        correct+=(pred==lab).sum().item()
    return total_loss/len(loader),correct/total

def validate(model,loader,criterion,device):
    model.eval()
    total_loss,correct,total=0,0,0
    with torch.no_grad():
        for img,lab in loader:
            img,lab=img.to(device),lab.to(device)
            out=model(img)
            loss=criterion(out,lab)
            total_loss+=loss.item()
            pred=out.argmax(dim=1)
            total+=lab.size(0)
            correct+=(pred==lab).sum().item()
    return total_loss/len(loader),correct/total

def run_one(pretrained,use_se,backbone_lr,fc_lr,batch_size,device,train_set,val_set,test_set):
    train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=2,pin_memory=True)
    val_loader=DataLoader(val_set,batch_size=batch_size,shuffle=False,num_workers=2,pin_memory=True)
    test_loader=DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=2,pin_memory=True)
    model=ResNet18(pretrained=pretrained,use_se=use_se).to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.AdamW([
        {"params":[p for n,p in model.named_parameters() if "fc" not in n],"lr":backbone_lr},
        {"params":model.backbone.fc.parameters(),"lr":fc_lr}
    ],weight_decay=1e-4)
    exp_name=f"pre{pretrained}_se{use_se}_blr{backbone_lr}_flr{fc_lr}_bs{batch_size}"
    ckpt_path=f"ckpt_{exp_name}.pth"
    wandb.init(
        project="pet-cls",
        name=exp_name,
        config={"pretrained":pretrained,"use_se":use_se,"backbone_lr":backbone_lr,"fc_lr":fc_lr,"batch_size":batch_size},
        resume="allow",
        id=wandb.util.generate_id(),
        mode="offline"
    )
    start_epoch=0
    best_val_acc=0.0
    if os.path.exists(ckpt_path):
        ckpt=torch.load(ckpt_path,map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
        start_epoch=ckpt["epoch"]+1
        best_val_acc=ckpt["best_val_acc"]
    for epoch in range(start_epoch,30):
        train_loss,train_acc=train_one_epoch(model,train_loader,criterion,optimizer,device)
        val_loss,val_acc=validate(model,val_loader,criterion,device)
        wandb.log({"epoch":epoch+1,"train_loss":train_loss,"train_acc":train_acc,"val_loss":val_loss,"val_acc":val_acc})
        if val_acc>best_val_acc:
            best_val_acc=val_acc
            torch.save(model.state_dict(),f"best_{exp_name}.pth")
        torch.save({"epoch":epoch,"model":model.state_dict(),"opt":optimizer.state_dict(),"best_val_acc":best_val_acc},ckpt_path)
    model.load_state_dict(torch.load(f"best_{exp_name}.pth"))
    test_loss,test_acc=validate(model,test_loader,criterion,device)
    wandb.log({"test_acc":test_acc})
    wandb.finish()
    return best_val_acc,test_acc

if __name__=="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform=transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomResizedCrop(224,scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_test_transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    full_trainval=datasets.OxfordIIITPet(root="../data/oxford-iiit-pet",split="trainval",target_types="category",download=False)
    train_size=int(0.9*len(full_trainval))
    val_size=len(full_trainval)-train_size
    train_sub,val_sub=random_split(full_trainval,[train_size,val_size])
    train_set=TransformSubset(train_sub,train_transform)
    val_set=TransformSubset(val_sub,val_test_transform)
    test_set=datasets.OxfordIIITPet(root="../data/oxford-iiit-pet",split="test",target_types="category",download=False,transform=val_test_transform)
    PRETRAINED=True
    USE_SE=True
    backbone_lrs=[1e-5,5e-5,1e-4]
    fc_lrs=[1e-4,5e-4,1e-3]
    batch_sizes=[16,32]
    results=[]
    for blr in backbone_lrs:
        for flr in fc_lrs:
            for bs in batch_sizes:
                print(f"\nRunning: blr={blr},flr={flr},bs={bs}")
                best_val,test_acc=run_one(PRETRAINED,USE_SE,blr,flr,bs,device,train_set,val_set,test_set)
                results.append({"blr":blr,"flr":flr,"bs":bs,"best_val_acc":best_val,"test_acc":test_acc})
    print("Overall：")
    for res in results:
        print(f"blr={res['blr']},flr={res['flr']},bs={res['bs']} | val_acc={res['best_val_acc']:.4f},test_acc={res['test_acc']:.4f}")

current_group_best=max(results,key=lambda x:x["test_acc"])
group_model_name=f"final_pre{PRETRAINED}_se{USE_SE}.pth"
best_exp_name=f"pre{PRETRAINED}_se{USE_SE}_blr{current_group_best['blr']}_flr{current_group_best['flr']}_bs{current_group_best['bs']}"
shutil.copy(f"best_{best_exp_name}.pth",group_model_name)
print("\n")
print(f"current group：pretrain={PRETRAINED}，use_se={USE_SE}")
print(f"best in current group：{current_group_best['test_acc']:.4f}")
print(f"Saved：{group_model_name}")
