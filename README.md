# Magnification-Prior-Self-Supervised-Method Result Review

In this forked reporsitory, a self supervised method, suggested by Chhipa, P., Upadhyay,1 R., Pihlgren, G., Saini, R., Uchida, S., Liwicki, M. (2022, September 08) for 
detecting malign on Histopathological images called Magnification-Prior-Self- Supervised-Method will be reviewed. Project aims to reproduce results explained in the original article and reproduce the results to help researchers focusing similar image classification problem.

Repository is created by Visual Studio Code as per writers and I converted it to a notebook version because of the computation power issue andrun this code on colab by using google drive and loading image sets to Google Drive Folders.

There were some minor changes in code like excluding Accimage since there were no way to install it to Colab and some of the configuration file changes applied.
In this review work, single gpu is used and code run for less epoch compared to original version (1000 Epoch) so conditions are not %100 same.


For following the order of running commands can be found in notebook (Deep_Learning_Final_Project_NotebookVersion.ipynb) however  below table is explaning the experiment of this work and its running order.

![image](https://user-images.githubusercontent.com/119973966/216549715-b28a1562-8550-4c64-8c98-933e1003e170.png)

To run this code, config and bc_config files need to be modified as per the location of the input images and output results to be saved. There are also experiment configuration files for both.

**1 To run data preparation for Break His ./src/data/prepare_data_breakhis.py **
Change following folder configuration

```root = '/home/datasets/BreaKHis_v1/histology_slides/breast'```

**2.#src/bc_config.py**
**Change following folder configuration**

```result_path = '/home/result/results_bc_5fold/'```

```tensorboard_path = '/home/logs/tensorboard_bc_5fold/'```

**3. /src/supervised/apply/config.py**
**Change following folder configuration**

```result_path = '/home/result/results_bc/'```

```tensorboard_path = '/home/logs/tensorboard_bc/'```


Original Implemantation and Article is as follows.
Implementation for ['Magnification Prior: A Self-Supervised Method for Learning Representations on Breast Cancer Histopathological Images'] (https://arxiv.org/abs/2203.07707) - Accepted in EEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2023

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/magnification-prior-a-self-supervised-method/breast-cancer-histology-image-classification)](https://paperswithcode.com/sota/breast-cancer-histology-image-classification?p=magnification-prior-a-self-supervised-method)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/magnification-prior-a-self-supervised-method/breast-cancer-histology-image-classification-1)](https://paperswithcode.com/sota/breast-cancer-histology-image-classification-1?p=magnification-prior-a-self-supervised-method)


# Requirement
This repository code is compaitible with Python 3.6 and 3.8, Pytorch 1.2.0, and Torchvision 0.4.0.

# Dataset
**BreakHis** - This is publically available dataset on Breast Cancer Histopathology WSI of several magnifications. Link - https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/. Details from BreakHis website: The Breast Cancer Histopathological Image Classification (BreakHis) is  composed of 9,109 microscopic images of breast tumor tissue collected from 82 patients using different magnifying factors (40X, 100X, 200X, and 400X).  To date, it contains 2,480  benign and 5,429 malignant samples (700X460 pixels, 3-channel RGB, 8-bit depth in each channel, PNG format). This database has been built in collaboration with the P&D Laboratory  – Pathological Anatomy and Cytopathology, Parana, Brazil (http://www.prevencaoediagnose.com.br). We believe that researchers will find this database a useful tool since it makes future benchmarking and evaluation possible.

# Commands

**Data access, prepartion, and processing scripts in src/data package**

**BreakHis dataset** 
```python -m prepare_data_breakhis```


**Self-supervised pretraining on BreakHis Dataset** 

**1. Single GPU implementation for constrained computation - use and customize the config files located in src/self_supervised/experiment_config/single_gpu - example mentioned below** 
```python -m pretrain_mpcs_single_gpu --config experiment_config/single_gpu/mpcs_op_rn50.yaml```
**It choses Ordered Pair smapling method for MPCS pretraining for ResNet50 encoder. Refer config files for cokmplete details and alternatives. Batch size needs to be small in this settings.

**Downstream Task on BreakHis dataset** 
**1. ImageNet supervised transfer learning finetune for malgnancy classification** 
```python -m finetune_breakhis --config experiment_config/breakhis_imagenet_rn50.yaml```
**Refer config files for cokmplete details and alternatives. This scripts runs model finetunung for each fold of 5 folds of dataset on given gpu mappings. Evaluation takes place after finetununbg completed on validation and testset and results are logged. no manual instruction needed.

**2. MPCS self-supervised transfer learning finetune for malgnancy classification** 
```python -m finetune_breakhis --config experiment_config/breakhis_mpcs_rn50.yaml```
**Refer config files for cokmplete details and alternatives and smapling method ordered pair, fixed pair and random pair. This scripts runs model finetunung for each fold of 5 folds of dataset on given gpu mappings. Pretraine models are search, accessed by scripts for given base path of all models autonomously and it fine tune models for each listed pretrained model weights for each batch size available. Evaluation takes place after finetuning completed on validation and testset and results are logged. no manual instruction needed.

