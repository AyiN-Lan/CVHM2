# 计算机视觉期中作业（CVHM2 2026）
本仓库包含图像分类、目标检测与跟踪、语义分割三大任务代码、实验结果及完整运行说明，所有代码基于PyTorch实现，使用GPU加速训练。

## 一、环境配置
### 1. 基础环境
- Python 3.8~3.10
- 训练硬件：NVIDIA GPU + CUDA（代码自动调用可用GPU）
- 推荐CUDA版本：11.7 / 11.8

### 2. 依赖安装
pip install -r requirements.txt

本次实验依赖包：
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
wandb>=0.15.0

## 二、数据与模型准备（本地配置）
本仓库不上传数据集、模型权重、视频文件，需自行下载并配置本地路径：
1. 任务1/任务3共用数据集：Oxford-IIIT Pet Dataset，需下载至本地并在代码中修改数据集路径
2. 任务2数据集：VisDrone数据集，需在VisDrone.yaml中配置本地数据集根路径
3. 任务2预训练权重：下载yolov8s.pt，放置于task2_detection_tracking文件夹下
4. 所有训练、测试需先在代码中指定本地数据路径，无法直接运行

## 三、仓库文件结构
<pre>
CV-Midterm/
├── data/                      # 数据集存放目录
│   ├── Oxford-IIIT_Pet/
│   └── VisDrone/
├── README.md
├── requirements.txt           # 环境依赖文件
├── task1_classification/
│   ├── train.py
│   ├── task1_test_acc.png
│   ├── task1_train_acc.png
│   ├── task1_train_loss.png
│   ├── task1_val_acc.png
│   └── task1_val_loss.png
├── task2_detection_tracking/
│   ├── train.py
│   ├── track_count.py
│   ├── VisDrone.yaml
│   ├── track_count.log
│   ├── results.csv # YOLOv8 训练完整指标
│   ├── results.png # 综合训练曲线
│   ├── BoxPR_curve.png # 各类别 PR 曲线
│   ├── BoxP_curve.png # 精确率 - 置信度曲线
│   ├── BoxR_curve.png # 召回率 - 置信度曲线
│   ├── BoxF1_curve.png # F1 - 置信度曲线
│   ├── confusion_matrix.png # 原始混淆矩阵
│   ├── confusion_matrix_normalized.png # 归一化混淆矩阵
│   └── labels.jpg # 数据集统计分析图
└── task3_segmentation/
    ├── train.py
    ├── task3_test_miou.png
    ├── task3_train_loss.png
    ├── task3_train_miou.png
    ├── task3_val_loss.png
    └── task3_val_miou.png
</pre>
## 四、运行指令
### 任务1：图像分类（ResNet-18+注意力机制）
cd task1_classification<br>
python train.py
- 需提前配置本地Oxford-IIIT Pet数据集路径
- 训练日志离线模式保存在本地，需手动同步至WandB
- 实验曲线已保存至当前文件夹
- 需在代码中手动修改全局pretrained和use_se变量参数True/False
- 运行指令后会开始训练模型，生成各个超参数下的最优模型权重，此外还会生成存档点权重文件
### 任务2：目标检测与跟踪（YOLOv8+ByteTrack）
cd task2_detection_tracking
# 训练YOLOv8检测模型
python train.py
# 执行跟踪与越线计数
python track_count.py
- 需配置VisDrone.yaml数据集路径以及测试视频路径
- 越线计数结果保存至track_count.log
- 输出视频文件过大，不上传至仓库

### 任务3：语义分割（从零搭建U-Net）
cd task3_segmentation<br>
python train.py
- 支持CE、Dice、CE+Dice三种损失函数，需在代码中对全局变量进行手动修改
- 需提前配置本地Oxford-IIIT Pet数据集路径
- 训练日志离线模式保存在本地，需手动同步至WandB
- 实验曲线已保存至当前文件夹
- 运行指令后会开始训练模型，生成当前损失函数下的的最优模型权重，此外还会生成存档点权重文件
## 五、实验结果
### 任务1：图像分类
测试集最高准确率：87.56%

### 任务2：目标检测与跟踪
- 完成手机街拍场景多目标检测、跟踪、双向越线计数
- 分析ByteTrack算法在遮挡场景下的ID跳变与漏检问题
- 计数结果：track_count.log

### 任务3：语义分割（三分类）
| 损失函数 | 达到最优轮数 | 最优验证集mIoU | 测试集mIoU |
| -------- | ------------ | --------------- | ----------- |
| CE       | 79           | 0.5700          | 0.5638      |
| DICE     | 84           | 0.5607          | 0.5579      |
| CE+DICE  | 79           | 0.5668          | 0.5648      |

所有实验曲线截图均来自WandB，已上传至对应任务文件夹。

## 六、补充说明
1. 全程使用GPU加速训练，无GPU环境无法正常训练
2. 数据集、模型权重、输出视频均为大文件，不上传至GitHub
3. 运行代码前必须修改本地数据路径，无法直接执行

## 模型权重下载链接
https://pan.baidu.com/s/1iq4sza62r8Z7JKW3DnpWPg?pwd=8upn 提取码: 8upn
