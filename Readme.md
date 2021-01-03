# 环境依赖
matplotlib==3.3.2\
numpy==1.19\
pytorch==1.7.1\
pytorch_lightning==1.1.0\
scipy==1.5.2\
opencv-python==4.4.0
# 目录结构
```python
└─dataset
│      ─train_x                      #训练数据-重建图像
│      ─train_y                      #训练数据-真值图像
│      ─test_x                       #测试数据-重建图像
│      ─test_y                       #测试数据-真值图像
│      ─H_weight.mat           
│ ─fig                               
└─opt
│      ─Data                         #训练参数-数据
│      ─Model                        #训练参数-模型
│      ─trainer                      #训练参数-训练器
│      ─train.yaml
│ ─Data.py                           #数据脚本
│ ─Model.py                          #模型脚本
│ ─Main.py                           #训练脚本
│ ─PGD demo.py                       #投影梯度算法展示脚本
│ ─Readme.md                         #说明文档
```
# 使用方法
## 部署环境
## 了解投影梯度算法
1. 运行 PGD_demo.py
```shell
python .\PGD demo.py
```
2. 效果图
![avatar](./fig/Track%20of%20solutions%20for%20PGD%20demo.png)
## 用PGD重建医学图像
1. 更改./opt/Data/Dataset/train.yaml中的数据集路径
2. 运行 Main.py
```shell
python .\PGD demo.py
```
3. 效果图
![avatar](./fig/comparison%20of%20PGD.png)