# Digital Profile of Pediatric Cardiac Function: Automated Echocardiogram Strain Analysis with Robust Validation on Diverse Datasets and Downstream Tasks
### Introduction
This repository provides the detailed code for the paper "Digital Profile of Pediatric Cardiac Function: Automated Echocardiogram Strain Analysis with Robust Validation on Diverse Datasets and Downstream Tasks". The code contains the implementation of the `Motion-Echo` network with the training and inference process.

We collected a pediatric dataset comprising 11,096 videos, covering children diagnosed with solid tumors and DMD. In conjunction with public adult datasets from four cohorts, the datasets total around 22,393 videos. Motion-Echo system is proposed, which   innovatively integrates the context compensation module and motion estimation module with a semi-supervised learning strategy. Myocardial global strain is estimated in a holistic video level based on the cardiac segmentation. The mean absolute deviation for global longitudinal strain (GLS) and global circumferential strain (GCS) are 2.099 and 2.665 with Pearson correlation coefficients of 0.799 and 0.781, respectively. Motion-Echo is validated across various populations, image qualities, and vendors. Ultimately, focusing on the primary concerns in pediatric CVDs, three downstream challenging tasks: CTRCD event predicting, late gadolinium enhancement (LGE) diagnosis and left ventricular ejection fraction (LVEF) decrease prediction are quantitatively evaluated, thereby facilitating the early warning of subclinical cardiac dysfunction. 


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
The code is implemented based on Tensorflow 2.4.0 with CUDA 11.4, OpenCV 4.7.0 and Python 3.8.16. It is tested in Ubuntu 18.04.5 with one 24GB GeForce RTX 3090Ti GPU. GPU usage is about 24GB.
Some important required packages include:

* Numpy == 1.19.5
* Nibabel == 5.0.1
* scikit-learn == 1.2.1
* imgaug == 0.4.0
* matplotlib == 3.4.3

### Datasets

The raw data should be structured as follows:

```
├── Dataset
    ├── HCC_001
    │   ├── pre_AP_image.nii.gz
    │   ├── pre_VP_image.nii.gz
    │   ├── post_AP_image.nii.gz
    │   ├── post_VP_image.nii.gz
    │   ├── pre_liver.nii.gz
    │   ├── pre_tumor.nii.gz
    │   ├── post_liver.nii.gz
    │   ├── post_tumor.nii.gz
    ├── HCC_002
    │   ├── pre_AP_image.nii.gz
    │   ├── pre_VP_image.nii.gz
    │   ├── post_AP_image.nii.gz
    │   ├── post_VP_image.nii.gz
    │   ├── pre_liver.nii.gz
    │   ├── pre_tumor.nii.gz
    │   ├── post_liver.nii.gz
    │   ├── post_tumor.nii.gz   
```
You can preprocess the raw data following these files: data_preprocess/preprocess.py, data_preprocess/get_roi.py, data_preprocess/registration.py

### Usage

1. Run the following commands for training.
```python
python train.py --epochs 120 --train_dir '../JSPH_TACE_train_data_STW' --transformer_layers 3 --response_mode 'concat' --survival_mode 'add'
```
2. Run the following commands for testing.
```python
python test.py --train_log 'log.csv' --transformer_layers 3 --response_mode 'concat' --survival_mode 'add'
```
