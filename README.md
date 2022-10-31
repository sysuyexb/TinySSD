# TinySSD  
**Artificial intelligence principle experiment: sorting out TinySSD code**   
## Author  
叶兴彬  

## Installation  
### Requirements  
1. Python3
2. PyTorch >=1.8
3. matplotlib >=3.5  
4. numpy>=1.20.3  
5. torch>=1.8.1+cu101   
6. torchvision>=0.9.1+cu101  
7. opencv_python_headless>=4.6.0.66  
8. pandas>=1.3.4     

`git clone https://github.com/sysutexb/TinySSD.git  
`cd TinySSD` 
`pip install -r requirements.txt`

## 数据目录介绍  
```  
.
├─data
    ├─background
    │      000012-16347456122a06.jpg
    │	. . .
    │      191328-15136820086f91.jpg
    │      
    ├─one_target_train
    │  │  
    │  └─images
    │          
    ├─target
    │      0.png
    │      1.png
    │      
    ├─test
    │      1.jpg
    │      2.jpg
    │      3.jpg
    │      4.jpg
    │      
    └─two_target_train
        │  
        └─images  
```  
【data目录说明】  
**background**：将你自己准备的背景图片放在这里  
**target**：目标图片放在这里【注意：目标图片的命名按照0.png、1.png…来命名】  
**one_target_train**：生成的训练数据存放在这里（此处对应单目标检测，如若需要多个目标检测，建议新建文件夹存放生成的训练数据，命名只需要将“one”改为对应目标的数目）  
**test**：存放测试图片    


  
  
## 训练流程  
代码结构如下：  
1. create_data.py
2. load_data.py  
3. model.py
4. train.py
5. test.py
6. plot.py
7. util.py
  
其中，代码1，4，5为主要运行代码，其余为辅助代码。  
主要运行代码主函数中有一个变量：target_num，如果设置其为'one'，则对应一个目标；如果设置其为'two'，则对应两个目标。
  

### 1、数据准备  
我们会自己制作目标检测的数据集，目标将被粘入背景图中，并保存目标位置即可。  
在create_data.py中修改对应的target_num之后，运行生成训练数据，target被粘在background文件夹中的每一张图片上，粘贴的位置随机。  
生成的新图片将存入one_target_train/images文件夹中，图片对应粘贴的位置将生成label.csv存入one_target_train文件夹中。    

`python create_train.py`   
  
### 2、模型训练  
在train.py中修改batch_size、epoch以及target_num之后运行。  
训练好的模型存入文件夹net中对应的子文件夹，设置为每10轮存储一次。  

`python train.py`  
  
如果训练成功，运行框将如下图所示：  
![ao](results/train_result.png"训练成功结果")  
  
  
### 3、模型测试  
在test.py中修改对应的target_num之后运行。  
运行结果将展示带有目标检测框的图片。  
  
`python test.py`  
  
测试结果展示（目标检测框左上角给出"类别：置信度"）：  
![ao](results/one_target.png"训练成功结果")  

 








