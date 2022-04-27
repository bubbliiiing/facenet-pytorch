## Facenet：人脸识别模型在Pytorch当中的实现
---

## 目录
1. [仓库更新 Top News](#仓库更新)
2. [相关仓库 Related code](#相关仓库)
3. [性能情况 Performance](#性能情况)
4. [所需环境 Environment](#所需环境)
5. [注意事项 Attention](#注意事项)
6. [文件下载 Download](#文件下载)
7. [预测步骤 How2predict](#预测步骤)
8. [训练步骤 How2train](#训练步骤)
9. [参考资料 Reference](#Reference)

## Top News
**`2022-03`**:**进行了大幅度的更新，支持step、cos学习率下降法、支持adam、sgd优化器选择、支持学习率根据batch_size自适应调整。**  
BiliBili视频中的原仓库地址为：https://github.com/bubbliiiing/facenet-pytorch/tree/bilibili

**`2021-02`**:**创建仓库，支持模型训练，大量的注释，多个可调整参数，lfw数据集评估等。**  

## 相关仓库
| 模型 | 路径 |
| :----- | :----- |
facenet | https://github.com/bubbliiiing/facenet-pytorch
arcface | https://github.com/bubbliiiing/arcface-pytorch
retinaface | https://github.com/bubbliiiing/retinaface-pytorch
facenet + retinaface | https://github.com/bubbliiiing/facenet-retinaface-pytorch

## 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | accuracy |
| :-----: | :-----: | :------: | :------: | :------: |
| CASIA-WebFace | [facenet_mobilenet.pth](https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/facenet_mobilenet.pth) | LFW | 160x160 | 98.23% |
| CASIA-WebFace | [facenet_inception_resnetv1.pth](https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/facenet_inception_resnetv1.pth) | LFW | 160x160 | 98.78% |

## 所需环境
pytorch==1.2.0

## 文件下载
已经训练好的facenet_mobilenet.pth和facenet_inception_resnetv1.pth可以在百度网盘下载。    
链接: https://pan.baidu.com/s/1K20hyxU_UgSej1eZWih0Ag 提取码:anv6

训练用的CASIA-WebFaces数据集以及评估用的LFW数据集可以在百度网盘下载。    
链接: https://pan.baidu.com/s/1qMxFR8H_ih0xmY-rKgRejw 提取码: bcrq    

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，在model_data文件夹里已经有了facenet_mobilenet.pth，可直接运行predict.py输入：
```python
img\1_001.jpg
img\1_002.jpg
```  
2. 也可以在百度网盘下载facenet_inception_resnetv1.pth，放入model_data，修改facenet.py文件的model_path后，输入：
```python
img\1_001.jpg
img\1_002.jpg
```  
### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在facenet.py文件里面，在如下部分修改model_path和backbone使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，backbone对应主干特征提取网络**。  
```python
_defaults = {
    "model_path"    : "model_data/facenet_mobilenet.pth",
    "input_shape"   : (160, 160, 3),
    "backbone"      : "mobilenet",
    "cuda"          : True,
}
```
3. 运行predict.py，输入  
```python
img\1_001.jpg
img\1_002.jpg
```  

## 训练步骤
1. 本文使用如下格式进行训练。
```
|-datasets
    |-people0
        |-123.jpg
        |-234.jpg
    |-people1
        |-345.jpg
        |-456.jpg
    |-...
```  
2. 下载好数据集，将训练用的CASIA-WebFaces数据集以及评估用的LFW数据集，解压后放在根目录。
3. 在训练前利用txt_annotation.py文件生成对应的cls_train.txt。  
4. 利用train.py训练facenet模型，训练前，根据自己的需要选择backbone，model_path和backbone一定要对应。
5. 运行train.py即可开始训练。

## 评估步骤
1. 下载好评估数据集，将评估用的LFW数据集，解压后放在根目录
2. 在eval_LFW.py设置使用的主干特征提取网络和网络权值。
3. 运行eval_LFW.py来进行模型准确率评估。

## Reference
https://github.com/davidsandberg/facenet  
https://github.com/timesler/facenet-pytorch  

