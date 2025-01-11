# Digital Profile of Pediatric Cardiac Function: Automated Echocardiogram Strain Analysis with Robust Validation on Diverse Datasets and Downstream Tasks
### Introduction
This repository provides the detailed code for the paper "Digital Profile of Pediatric Cardiac Function: Automated Echocardiogram Strain Analysis with Robust Validation on Diverse Datasets and Downstream Tasks". The code contains the implementation of the `Motion-Echo` network with the training and inference process.

We collected a pediatric dataset comprising 11,096 videos, covering children diagnosed with solid tumors and DMD. In conjunction with public adult datasets from four cohorts, the datasets total around 22,393 videos. Motion-Echo system is proposed, which   innovatively integrates the context compensation module and motion estimation module with a semi-supervised learning strategy. Myocardial global strain is estimated in a holistic video level based on the cardiac segmentation. The mean absolute deviation for global longitudinal strain (GLS) and global circumferential strain (GCS) are 2.099 and 2.665 with Pearson correlation coefficients of 0.799 and 0.781, respectively. Motion-Echo is validated across various populations, image qualities, and vendors. Ultimately, focusing on the primary concerns in pediatric CVDs, three downstream challenging tasks: CTRCD event predicting, late gadolinium enhancement (LGE) diagnosis and left ventricular ejection fraction (LVEF) decrease prediction are quantitatively evaluated, thereby facilitating the early warning of subclinical cardiac dysfunction. 

![image](https://github.com/rushier/Motion-Echo/blob/main/digital_profile.svg)
-----

### Citation and license
If you use this code or find our research helpful, please cite:

The model and code are available for non-commercial (NC) research purposes only. Please do not distribute this code during the review process. If you modify the code and want to redistribute, please include the CC-BY-NC-SA-4.0 license.

-----

### News
- [x] The training and inference code of Motion-Echo
- [ ] The model weights of Motion-Echo
- [ ] Part of Echocardiograms for the training of Motion-Echo
 
-----

### Requirements:
The code is implemented based on Pytorch 2.0.1 with CUDA 12.4 and Python 3.8.19. It is trained and tested with two NVIDIA GeForce RTX 4090 GPUs. Clone this repository and install all the necessary dependecies written in the `requirements.txt` file with `pip install -r requirements.txt`. 

### Datasets

The raw data should be structured as follows:

```
├── Dataset
    ├── ZZ121836
    │   ├── L5RCI9AO_a2c.npz
    │   ├── L5RCH42C_a3c.npz
    │   ├── L5RCGKA2_a4c.npz
    │   ├── L5RCL1C8_mid.npz
    │   ├── L5RCKJ3U_mv.npz
    ├── YL121426
    │   ├── L3KCJM40_a2c.npz
    │   ├── L3KCJARS_a3c.npz
    │   ├── L3KCG81Q_a4c.npz
    │   ├── L3KCL8T6_mid.npz
    │   ├── L3KCKSSM_mv.npz 
```
The file train_val_test.csv for training and testing should be structured as follows:
```
| path                                         | cycle_number | PID      | Echodate | HR  | image_number | duration | chamber | gls | split | select_slice | gls_cycles            |
|----------------------------------------------|--------------|----------|----------|-----|--------------|----------|---------|-----|-------|--------------|-----------------------|
| data3/GEMS_IMG/2022_FEB/12/ZR113824/M2CBSPR4 | 1            | 10002091 | 20220212 | 108 | 50           | 3.96     | a4c     | 23  | val   | 10           | [24.19, 20.75, 24.06] |
| data3/GEMS_IMG/2022_FEB/12/ZR113824/M2CBSPR4 | 2            | 10002091 | 20220212 | 108 | 50           | 3.96     | a4c     | 23  | val   | 2            | [24.19, 20.75, 24.06] |
| data3/GEMS_IMG/2022_FEB/12/ZR113824/M2CBSPR4 | 3            | 10002091 | 20220212 | 108 | 50           | 3.96     | a4c     | 23  | val   | 8            | [24.19, 20.75, 24.06] |
| data3/GEMS_IMG/2022_FEB/12/ZR113824/M2CBSPR4 | 4            | 10002091 | 20220212 | 108 | 50           | 3.96     | a4c     | 23  | val   | random       | [24.19, 20.75, 24.06] |
```

### Usage

1. Run the following commands for training and testing Motion-Echo.
```python
CUDA_VISIBLE_DEVICES=0 python main_Motion-Echo.py --task seg --dataset huaxi2d --info_csv '/path_to_training_csv/train_val_test.csv' --size 224 --loss DiceLoss --bs 4 --ms [80,160] --epoch 201 --model_save_freq 20  --model_dir 'save_models/' --root_path '/path_to_echocardiograms/qq_raw_cycles/' --forward_uncer 0 --num_classes 3 --temporal_interval 1 --network TransUnet_cst_temporal_flow_woResidual --classify --chamber a4c_a2c_a3c --lr 1e-4 --prior --HistMatch --update_folder '/path_to_echocardiograms/segmentations'
```
```python
CUDA_VISIBLE_DEVICES=0 python main_Motion-Echo.py --task seg --dataset huaxi2d --info_csv '/path_to_training_csv/train_val_test.csv' --size 224 --loss DiceLoss --bs 4 --ms [80,160] --epoch 201 --model_save_freq 20  --model_dir 'save_models/' --root_path '/path_to_echocardiograms/qq_raw_cycles/' --forward_uncer 0 --num_classes 3 --temporal_interval 1 --network TransUnet_cst_temporal_flow_woResidual --classify --chamber a4c_a2c_a3c --lr 1e-4 --prior --HistMatch --update_folder '/path_to_echocardiograms/segmentations' --test
```
2. Run the following commands for training and testing Motion-Echo_2d+t.
```python
CUDA_VISIBLE_DEVICES=0 python main_Motion-Echo_2d+t.py --task seg --dataset Huaxi3d --info_csv '/ailab/group/pjlab-medai/qianyi/jiaorushi/GLS/code/Oncocardiology_huaxi/DeepGLS/load/train_val_test_zhaoli_1129_cycles_labeled_0220.csv' --size 112 --loss DiceLoss --bs 2 --ms [40,80] --epoch 81 --model_save_freq 100  --model_dir 'save_models_aug/' --root_path '/ailab/group/pjlab-medai/qianyi/jiaorushi/GLS/data/qq_raw_cycles/'  --num_classes 3 --network SpaceTimeUnet --chamber a4c_a2c_a3c --lr 1e-4 --prior
```
```python
CUDA_VISIBLE_DEVICES=0 python main_Motion-Echo_2d+t.py --task seg --dataset Huaxi3d --info_csv '/ailab/group/pjlab-medai/qianyi/jiaorushi/GLS/code/Oncocardiology_huaxi/DeepGLS/load/train_val_test_zhaoli_1129_cycles_labeled_0220.csv' --size 112 --loss DiceLoss --bs 1 --ms [40,80] --epoch 81 --model_save_freq 100  --model_dir 'save_models_aug/' --root_path '/ailab/group/pjlab-medai/qianyi/jiaorushi/GLS/data/qq_raw_cycles/'  --num_classes 3 --network SpaceTimeUnet --chamber a4c_a2c_a3c --lr 1e-4 --prior --test
```
