# OpenImage-ObjectDetection

```python
pwd
```




    '/workspace/Docker/yolov5_openImagev5'




```python
#openImage를 다운받기 위한 git clone
!git clone https://github.com/EscVM/OIDv4_ToolKit.git
```

    fatal: destination path 'OIDv4_ToolKit' already exists and is not an empty directory.



```python
cd OIDv4_ToolKit
```

    /workspace/Docker/yolov5_openImagev5/OIDv4_ToolKit



```python
#관련 library등 install
!pip install -r requirements.txt
```

    Requirement already satisfied: pandas in /opt/conda/lib/python3.8/site-packages (from -r requirements.txt (line 1)) (1.3.0)
    Requirement already satisfied: numpy in /opt/conda/lib/python3.8/site-packages (from -r requirements.txt (line 2)) (1.19.2)
    Collecting awscli
      Downloading awscli-1.20.30-py3-none-any.whl (3.7 MB)
    [K     |████████████████████████████████| 3.7 MB 2.0 MB/s eta 0:00:01
    [?25hRequirement already satisfied: urllib3 in /opt/conda/lib/python3.8/site-packages (from -r requirements.txt (line 5)) (1.25.11)
    Requirement already satisfied: tqdm in /opt/conda/lib/python3.8/site-packages (from -r requirements.txt (line 7)) (4.51.0)
    Requirement already satisfied: opencv-python in /opt/conda/lib/python3.8/site-packages (from -r requirements.txt (line 9)) (4.5.3.56)
    Requirement already satisfied: pytz>=2017.3 in /opt/conda/lib/python3.8/site-packages (from pandas->-r requirements.txt (line 1)) (2020.5)
    Requirement already satisfied: python-dateutil>=2.7.3 in /opt/conda/lib/python3.8/site-packages (from pandas->-r requirements.txt (line 1)) (2.8.2)
    Requirement already satisfied: PyYAML<5.5,>=3.10 in /opt/conda/lib/python3.8/site-packages (from awscli->-r requirements.txt (line 3)) (5.3.1)
    Collecting s3transfer<0.6.0,>=0.5.0
      Downloading s3transfer-0.5.0-py3-none-any.whl (79 kB)
    [K     |████████████████████████████████| 79 kB 4.9 MB/s  eta 0:00:01
    [?25hCollecting docutils<0.16,>=0.10
      Downloading docutils-0.15.2-py3-none-any.whl (547 kB)
    [K     |████████████████████████████████| 547 kB 10.9 MB/s eta 0:00:01
    [?25hRequirement already satisfied: rsa<4.8,>=3.1.2 in /opt/conda/lib/python3.8/site-packages (from awscli->-r requirements.txt (line 3)) (4.7.2)
    Collecting botocore==1.21.30
      Downloading botocore-1.21.30-py3-none-any.whl (7.8 MB)
    [K     |████████████████████████████████| 7.8 MB 10.6 MB/s eta 0:00:01
    [?25hCollecting colorama<0.4.4,>=0.2.5
      Downloading colorama-0.4.3-py2.py3-none-any.whl (15 kB)
    Collecting jmespath<1.0.0,>=0.7.1
      Downloading jmespath-0.10.0-py2.py3-none-any.whl (24 kB)
    Requirement already satisfied: six>=1.5 in /opt/conda/lib/python3.8/site-packages (from python-dateutil>=2.7.3->pandas->-r requirements.txt (line 1)) (1.15.0)
    Requirement already satisfied: pyasn1>=0.1.3 in /opt/conda/lib/python3.8/site-packages (from rsa<4.8,>=3.1.2->awscli->-r requirements.txt (line 3)) (0.4.8)
    Installing collected packages: jmespath, botocore, s3transfer, docutils, colorama, awscli
    Successfully installed awscli-1.20.30 botocore-1.21.30 colorama-0.4.3 docutils-0.15.2 jmespath-0.10.0 s3transfer-0.5.0
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m



```python
pwd
```




    '/workspace/Docker/yolov5_openImagev5'




```python
#!python main.py downloader --classes Apple Orange --type_csv train --limit 5000 --multiclasses 1
# apple orange이미지 5000장 download
```


```python
#object detection을 하기위한 yolov5 git clone
!git clone https://github.com/ultralytics/yolov5.git
```

    fatal: destination path 'yolov5' already exists and is not an empty directory.



```python
#Train
```


```python
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os
```


```python
pwd
```




    '/workspace/Docker/yolov5_openImagev5/yolov5'




```python
!python detect.py --weight /workspace/Docker/yolov5_openImagev5/yolov5/runs/train/yolov5_AppleOrange3/weights/best.pt --img 416 --conf 0.5 --source "/workspace/ssddata/yolov5_openImage_dataset/test/"
```

    [34m[1mdetect: [0mweights=['/workspace/Docker/yolov5_openImagev5/yolov5/runs/train/yolov5_AppleOrange3/weights/best.pt'], source=/workspace/ssddata/yolov5_openImage_dataset/test/, imgsz=416, conf_thres=0.5, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False
    YOLOv5 🚀 v5.0-312-gd17b45e torch 1.7.1 CUDA:0 (NVIDIA GeForce RTX 3090, 24265.3125MB)
    
    Fusing layers... 
    Model Summary: 224 layers, 7056607 parameters, 0 gradients, 16.3 GFLOPs
    image 1/6 /workspace/ssddata/yolov5_openImage_dataset/test/download.jpg: 288x416 1 apple, 1 orange, Done. (0.009s)
    image 2/6 /workspace/ssddata/yolov5_openImage_dataset/test/image1.jpg: 288x416 1 apple, 1 orange, Done. (0.006s)
    image 3/6 /workspace/ssddata/yolov5_openImage_dataset/test/image2.jpg: 256x416 1 apple, 1 orange, Done. (0.008s)
    image 4/6 /workspace/ssddata/yolov5_openImage_dataset/test/image3.jpg: 384x416 1 apple, Done. (0.009s)
    image 5/6 /workspace/ssddata/yolov5_openImage_dataset/test/images (1).jpg: 320x416 5 apples, Done. (0.008s)
    image 6/6 /workspace/ssddata/yolov5_openImage_dataset/test/images.jpg: 192x416 1 apple, 1 orange, Done. (0.010s)
    Results saved to runs/detect/exp9
    Done. (0.087s)



```python
result_img = glob('/workspace/Docker/yolov5_openImagev5/yolov5/runs/detect/exp9/*.jpg')
len(result_img)
```




    6




```python
img = plt.imread(result_img[2])
plt.imshow(img)
plt.axis('off')
plt.show()
```


    
![png](output_12_0.png)
    



```python

```
