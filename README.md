# TinySSD  
**Artificial intelligence principle experiment: sorting out TinySSD code**   


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

```
git clone https://github.com/sysutexb/TinySSD.git  
cd TinySSD  
pip install -r requirements.txt  
```
  
  
## Training process  

### 1、Data generation  （TinySSD_create_train.py）  
This file helps us generate a training dataset, which copies the target image to the background image to generate a new picture.    

```
python3 TinySSD_create_train.py  
```
  
If the code runs successfully, the structure of the dataset file is as follows:  
  
```  
.
├─detection
    ├─background
    │      000012-16347456122a06.jpg
    │      000043-157305604379c0.jpg
    │	       .
    │	       .
    │	       .
    │      
    ├─sysu_train
    │  │  
    │  └─label.csv  
    │  │  
    │  └─images
    │         000012-16347456122a06.jpg
    │         000043-157305604379c0.jpg
    │	             .
    │	             .
    │	             .
    │          
    ├─target
    │      0.png
    │      0.jpg
    │      
    └─test
           1.jpg
           2.jpg

```  
  
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

 

## Author  
**叶兴彬   20354148**  






