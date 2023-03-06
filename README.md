# [CSA-CDGAN: Channel Self-Attention Based Generative Adversarial Network for Change Detection of Remote Sensing Images](https://link.springer.com/article/10.1007/s00521-022-07637-z)
A general framework for change detection of remote sensing images  
paper link: https://link.springer.com/article/10.1007/s00521-022-07637-z
<img src="https://user-images.githubusercontent.com/79884379/200501481-ddc81274-2d08-42bc-b2d4-38e0d0f9d11b.png" width="800" height="250">
## Quantitative & Qualitative Results on CDD, WHU-CD, LEVIR-CD Datasets
<img src="https://user-images.githubusercontent.com/79884379/161015442-4a4a8c19-b2d8-484a-9620-ec529fe2d1a0.png" width="800" height="450">  
<img src="https://user-images.githubusercontent.com/79884379/161015514-15326ba5-66e8-4eab-b4c8-52f4fcdcad9c.png" width="700" height="450">  
<img src="https://user-images.githubusercontent.com/79884379/161015569-40f5b2bb-dc20-4183-a9d7-594d8663a217.png" width="700" height="450">  
<img src="https://user-images.githubusercontent.com/79884379/161015612-2eed4cfc-be8e-4adc-88e6-39b6cfbf63e0.png" width="700" height="450">  
<img src="https://user-images.githubusercontent.com/79884379/161015733-89ba4a56-2119-4a7c-90cd-9445140fe174.png" width="700" height="450">  
<img src="https://user-images.githubusercontent.com/79884379/161015764-9e64d3e6-324d-400b-ae8e-48892bfddc26.png" width="700" height="450">  

## Requirements
```
Python 3.7.0  
Pytorch 1.6.0  
Visdom 0.1.8.9  
Torchvision 0.7.0
```
## Datasets
  - CDD: [Change detection in remote sensing images using conditional adversarial networks](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9)
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
To display training processing, copy 'http://localhost:8097' to your browser.
## Citing CSA-CDGAN
If you use this repository or would like to refer the paper, please use the following BibTex entry.
```
@article{wang2022csa,
  title={CSA-CDGAN: channel self-attention-based generative adversarial network for change detection of remote sensing images},
  author={Wang, Zhixue and Zhang, Yu and Luo, Lin and Wang, Nan},
  journal={Neural Computing and Applications},
  pages={1--15},
  year={2022},
  publisher={Springer}
}
```
## Reference
```
-Akcay, Samet, Amir Atapour-Abarghouei, and Toby P. Breckon. "Ganomaly: Semi-supervised anomaly detection via adversarial training." Asian conference on computer vision. Springer, Cham, 2018.
```
## More
[My personal google web](https://scholar.google.com/citations?user=qdkY0jcAAAAJ&hl=zh-TW)
