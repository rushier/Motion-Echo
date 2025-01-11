import cv2
import numpy as np
import math
from skimage import exposure
from PIL import Image
from skimage.restoration import (denoise_wavelet, estimate_sigma)

# 灰度级拉伸
def preprocessing_log(src, konstanta, nilaidaripixel) :
    RGB = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    images = RGB[:, :, 1]

    #semakin besar nilai C (konstanta), kecerahannya juga semakin cerah
    #semakin kecil nilai A (nilai dari pixel), kecerahannya semakin cerah
    g = konstanta * (np.log(nilaidaripixel+np.float64(images)))

    return g*255

def log_transform(src):
    gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    gray = gray(src)
    max_ = np.max(gray)
    return (255/np.log(1+max_)) * np.log(1+gray)


def gammacorrection(src, gamma):
    #nilai dari gamma (G) akan mempengaruhi kecerahan dari citra
    #G>1 akan membuat citra menjadi lebih gelap
    #G<1 akan membuat citra menjadi lebih cerah
    #G=1 tidak akan memberikan efek apapun
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in range(0, 256)]).astype("uint8")
    return cv2.LUT(src, table)


def Autogammacorrection(src):
    #nilai dari gamma (G) akan mempengaruhi kecerahan dari citra
    #G>1 akan membuat citra menjadi lebih gelap
    #G<1 akan membuat citra menjadi lebih cerah
    #G=1 tidak akan memberikan efek apapun
    mean = np.mean(src)
    gamma_val = math.log10(0.5) / math.log10(mean / 255) 
    table = np.array([((i / 255.0) ** gamma_val) * 255
                      for i in range(0, 256)]).astype("uint8")
    return cv2.LUT(src, table)

# 直方图均衡化

def Histogram_equalization(src):
    src = np.uint8(src)

    img_HE = cv2.equalizeHist(src) # 直方图均衡化

    img1 = exposure.equalize_adapthist(src) # 自适应直方图均衡化
    #img_AHE = Image.fromarray(np.uint8(img1 * 255))
    img_AHE = Image.fromarray(np.uint8(img1))
    img_AHE = np.array(img_AHE)

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(3, 3))
    img_CLAHE = clahe.apply(src)   # 将灰度图像和局部直方图相关联, 把直方图均衡化应用到灰度图
    #return img_HE, img_AHE, img_CLAHE
    return img_HE, img_AHE, img_CLAHE

