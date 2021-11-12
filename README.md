# CSA-CDGAN: Channel Self-Attention Based Generative Adversarial Network for Change Detection of Remote Sensing Images
A general framework for change detection of remote sensing images 
![image](https://user-images.githubusercontent.com/79884379/141422470-08dbcf3f-f2e2-410a-8383-671fc955570d.png)
## Requirements
```
Python 3.7.0  
Pytorch 1.6.0  
Visdom 0.1.8.9  
Torchvision 0.7.0
```
## Datasets
- CDD dataset
  - paper: [Change detection in remote sensing images using conditional adversarial networks. International Archives of the Photogrammetry](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGV
to-nHrNs9)
  - WHU-CD: [Fully Convolutional Networks for Multisource Building Extraction From an Open Aerial and Satellite Imagery Data Set](http://study.rsgis.whu.edu.cn/pages/download/)
  - LEVIR-CD: [A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection]( https://justchenhao.github.io/LEVIR/)  
You also can download datasets after being processed by us. [[Baiduyun]](https://pan.baidu.com/s/1ptiznHupKRigJwT_NyMxwA) the password is hnbi. or [[GoogleDrive]](https://drive.google.com/drive/folders/16N0Ii5VsouxE2Ak9PNq0Kac4v1801Nmx?usp=sharing)
## Pretrained Model
Pretrained models for CDD, LEVIR-CD and WHU-CD are available. You can download them from the following link.
[[Baiduyun]](https://pan.baidu.com/s/1WxQ52qtGLE-gz2MpZztu7g) the password is yudl. [[GoogleDrive]](https://drive.google.com/drive/folders/1IQDz_s0LiUjw1WtLDrGGmR9rPOkbrgzP?usp=sharing)
## Test
Before test, please download datasets and pretrained models. Revise the data-path in  `constants.py ` to your path. Copy pretrained models to folder `'./dataset_name/outputs/best_weights'`, and run the following command:
```
cd CSA-CDGAN_ROOT
python make_dataset.py
python test.py
```
`make_dataset.py` can generate .txt files for training, validation and test. Not that the dataset structure should be the same as following:
>Custom dataset  
|--train  
&ensp;&ensp;|--file1  
&ensp;&ensp;&ensp;&ensp;|--t0.jpg, t1.jpt, label.jpg  
&ensp;&ensp;|--file2  
&ensp;&ensp;&ensp;&ensp;|--t0.jpg, t1.jpt, label.jpg   
&ensp;&ensp;...  
|--test  
&ensp;&ensp;|--file1  
&ensp;&ensp;&ensp;&ensp;|--t0.jpg, t1.jpt, label.jpg  
&ensp;&ensp;|--file2  
&ensp;&ensp;&ensp;&ensp;|--t0.jpg, t1.jpt, label.jpg   
&ensp;&ensp;...  
|--validation  
&ensp;&ensp;|--file1  
&ensp;&ensp;&ensp;&ensp;|--t0.jpg, t1.jpt, label.jpg  
&ensp;&ensp;|--file2  
&ensp;&ensp;&ensp;&ensp;|--t0.jpg, t1.jpt, label.jpg   
&ensp;&ensp;...  
## Training
```
cd CSA-CDGAN_ROOT
python make_dataset.py
python -m visdom.server
python train.py
```
To display training processing, open 'http://localhost:8097' in your browser.
## Citing TransCD
If you use this repository or would like to refer the paper, please use the following BibTex entry.
```
@inproceddings{TransCD,
title={CSA-CDGAN: Channel Self-Attention Based Generative Adversarial Network for Change Detection of Remote Sensing Images},
author={ZHIXUE WANG, YU ZHANG*, LIN LUO, NAN WANG},
yera={2021},
}
```
## Reference
```
-Akcay, Samet, Amir Atapour-Abarghouei, and Toby P. Breckon. "Ganomaly: Semi-supervised anomaly detection via adversarial training." Asian conference on computer vision. Springer, Cham, 2018.
```
