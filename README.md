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

### Data generation  
这个文件帮助我们生成一个训练数据集，它将目标图像复制到背景图像以生成新的图片。    

```
python3 TinySSD_create_train.py  
```
  
如果代码成功运行，数据集的结构如下:  
  
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
  
### Training  
将合成图像导入TinySSD网络进行训练。
训练流程如下图所示：  
![image](https://github.com/sysuyexb/TinySSD/blob/main/picture/train.png?raw=true)  
  
1）训练图片先导入TinySSD中，生成多尺度的锚框，为每个锚框预测类别和偏移量  
2）然后使用multibox_target函数为每个锚框标注类别和偏移量  
3）接着使用calc_loss函数计算损失（根据类别和偏移量的预测和标注值）,反向传播  
4）最后使用cls_eval和bbox_eval计算类别损失和偏移损失 
5)训练成功的网络保存到net文件夹
  

`python3 TinySSD_train.py`  
  

  
  
### Testing  
将调用训练成功的网络进行目标检测效果测试。  
测试流程如下图所示：  
![image](https://github.com/sysuyexb/TinySSD/blob/main/picture/test.png?raw=true)  
  
1)将测试图片导入predict函数  
2）其内的multibox_detection函数计算出每个锚框的类别索引，置信度，预测边界框坐标  
3）调用非极大值抑制保留置信度大于阈值的锚框  
4)使用display函数输出测试结果
  
`python3 TinySSD_test.py`  
  
使用net_30.pkl的结果如下图所示：  
![image](https://github.com/sysuyexb/TinySSD/blob/main/picture/2.png?raw=true)     
  
    
  
### Improvement  
从数据方面进行效果提升  
对生成的训练数据采用数据增强的方法，包括

 

## Author  
**叶兴彬   20354148**  






