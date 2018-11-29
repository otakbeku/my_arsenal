from imgaug import augmenters as au
import cv2
import numpy as np
import os
import glob
import re

flipper = au.Fliplr(1.0)
vflipper = au.Flipud(0.9)
gblurer_zero_degree = au.GaussianBlur(0.5)
gblurer_one_degree = au.GaussianBlur(1.0)
gblurer_two_degree = au.GaussianBlur(2.0)
gblurer_three_degree = au.GaussianBlur(3.0)
translater = au.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}) 
scaler1 = au.Affine(scale=0.5)
scaler2 = au.Affine(scale=1.5)
scaler3 = au.Affine(scale=2.0)
superpixeler75 = au.Superpixels(p_replace=1, n_segments=75)
superpixeler100 = au.Superpixels(p_replace=1, n_segments=100)
superpixeler125 = au.Superpixels(p_replace=1, n_segments=125)
adder_min45 = au.Add(value=-45)
adder_min25 = au.Add(value=-25)
adder_plus25 = au.Add(value=25)
adder_plus45 = au.Add(value=45)
addHS_min45 = au.AddToHueAndSaturation(value=-45)
addHS_min25 = au.AddToHueAndSaturation(value=-25)
addHS_plus25 = au.AddToHueAndSaturation(value=25)
addHS_plus45 = au.AddToHueAndSaturation(value=45)
multiplyer1 = au.Multiply(0.5)
multiplyer2 = au.Multiply(1.5)
multiplyer3 = au.Multiply(2.0)
multiplyerE1 = au.MultiplyElementwise(0.5)
multiplyerE2 = au.MultiplyElementwise(1.5)
multiplyerE3 = au.MultiplyElementwise(2.0)
multiplyerE1pc1 = au.MultiplyElementwise(0.5, per_channel=0.5)
multiplyerE1pc2 = au.MultiplyElementwise(0.5, per_channel=1.0)
multiplyerE2pc1 = au.MultiplyElementwise(1.5, per_channel=0.5)
multiplyerE2pc2 = au.MultiplyElementwise(1.5, per_channel=1.0)
multiplyerE3pc1 = au.MultiplyElementwise(2.0, per_channel=0.5)
multiplyerE3pc2 = au.MultiplyElementwise(2.0, per_channel=1.0)
dropouter1 = au.Dropout(0.2)
dropouter1pc1 = au.Dropout(0.2, per_channel=0.2)
dropouter1pc2 = au.Dropout(0.2, per_channel=0.5)
dropouter2 = au.Dropout(0.5)
dropouter2pc1 = au.Dropout(0.5, per_channel=0.2)
dropouter2pc2 = au.Dropout(0.5, per_channel=0.5)
coarsedp1=au.CoarseDropout(0.02, size_percent=0.5)
coarsedp1pc1=au.CoarseDropout(0.02, size_percent=0.5, per_channel=0.5)
coarsedp2=au.CoarseDropout(0.05, size_percent=0.5)
coarsedp2pc1=au.CoarseDropout(0.05, size_percent=0.5, per_channel=0.5)
contrastnorm1 = au.ContrastNormalization(0.5)
contrastnorm1pc1 = au.ContrastNormalization(0.5, per_channel=0.5)
contrastnorm2 = au.ContrastNormalization(1.5)
contrastnorm2pc1 = au.ContrastNormalization(1.5, per_channel=0.5)
contrastnorm3 = au.ContrastNormalization(2.0)
contrastnorm3pc1 = au.ContrastNormalization(2.0, per_channel=0.5)
rotasi1 = au.Affine(rotate=45)
rotasi2 = au.Affine(rotate=-45)
avgbluer1 = au.AverageBlur(k=2)
avgbluer2 = au.AverageBlur(k=5)
avgbluer3 = au.AverageBlur(k=11)
medbluer1 = au.MedianBlur(k=3)
medbluer2 = au.MedianBlur(k=5)
medbluer3 = au.MedianBlur(k=11)
shapener1 = au.Sharpen(alpha=0.1)
shapener2 = au.Sharpen(alpha=0.5)
shapener3 = au.Sharpen(alpha=1.0)
gaussianNoise = au.AdditiveGaussianNoise(scale=(0, 0.05*255))
saltpepper1 = au.SaltAndPepper(p=0.03)
saltpepper2 = au.SaltAndPepper(p=0.1)
saltpepper3 = au.SaltAndPepper(p=0.3)
pppt1 = au.PerspectiveTransform(scale=0.075)
pppt2 = au.PerspectiveTransform(scale=0.1)

operation = {
# 'flip':flipper,
# 'vflip':vflipper,
'gblur1':gblurer_zero_degree,
'gblur2':gblurer_one_degree,
'gblur3':gblurer_two_degree,
'gblur4':gblurer_three_degree,
# 'trans':translater,
# 'scl1':scaler1,
# 'scl2':scaler2,
# 'scl3':scaler3,
# 'sp75':superpixeler75,
# 'sp100':superpixeler100,
# 'sp125':superpixeler125,
'addm45':adder_min45,
'addm25':adder_min25,
'addp25':adder_plus25,
'addp45':adder_plus45,
# 'addhm45':addHS_min45,
# 'addhm25':addHS_min25,
# 'addhp25':addHS_plus25,
# 'addhp45':addHS_plus45,
# 'mt1':multiplyer1,
'mt2':multiplyer2,
# 'mt3':multiplyer3,
# 'mte1':multiplyerE1,
# 'mte1pc1':multiplyerE1,
# 'mte1pc1':multiplyerE1pc1,
# 'mte1pc2':multiplyerE1pc2,
'mte2':multiplyerE2,
# 'mte2pc1':multiplyerE2pc1,
# 'mte2pc2':multiplyerE2pc2,
# 'mte3':multiplyerE3,
# 'mte3pc1':multiplyerE3pc1,
# 'mte3pc2':multiplyerE3pc2,
# 'do1':dropouter1,
# 'do1pc1':dropouter1pc1,
# 'do1pc2':dropouter1pc2,
'do2':dropouter2,
# 'do2pc1':dropouter2pc1,
# 'do2pc2':dropouter2pc2,
'cdp1':coarsedp1,
# 'cdp1pc1':coarsedp1pc1,
# 'cdp2':coarsedp2,
# 'cdp2pc1':coarsedp2pc1,
# 'cnt1':contrastnorm1,
# 'cnt1pc1':contrastnorm1pc1,
# 'cnt2':contrastnorm2,
# 'cnt2pc1':contrastnorm2pc1,
# 'rot1':rotasi1,
# 'rot2':rotasi2,
# 'avgblur1':avgbluer1,
# 'avgblur2':avgbluer2,
# 'avgblur3':avgbluer3,
# 'medblur1':medbluer1,
# 'medblur2':medbluer2,
# 'medblur3':medbluer3,
# 'shpr1':shapener1,
'shpr2':shapener2,
# 'shpr3':shapener3,
'gn':gaussianNoise,
'spt1':saltpepper1,
# 'spt2':saltpepper2,
'spt3':saltpepper3,
'ppt1':pppt1,
# 'ppt2':pppt2,
}
# Testing 
# image = cv2.imread('contoh.jpg')
# img = operation['avgblur2'].augment_image(image)
# cv2.imshow('tes', img)
# cv2.imshow('ori', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(len(operation))

path = 'F:\FSR\dataset\skin-cancer-mnist-ham10000\Locat'
path2='F:\FSR\dataset\skin-cancer-mnist-ham10000\Locataug'

lanjut = input('Jumlah operasi :'+str(len(operation))+' Di lanjutkan? (y/n)')
if lanjut.lower() == 'n':
    exit()
for cat_folder in os.listdir(path):
    print('-', cat_folder)
    for sub_cat_folder in os.listdir(path+'//'+cat_folder):
        print('\t--', sub_cat_folder)
        for image_path in glob.glob(path + "/" + cat_folder + "/" + sub_cat_folder + "/*.jpg"):
            print('\t\t--- ', image_path)
            image_name = image_path.replace(path, path2)
            image_name = image_name.replace('.jpg', '')
            # print('\t\t------ ',image_name)
            image = cv2.imread(image_path)
            # Ngga ada yang bisa dipakai dibawah ini
            # for key1 in operation.keys():
            #     print('\t\t\t--- ', str(key1))
            #     temp = operation[key1].augment_image(image)
            #     cv2.imwrite(image_name+'_'+str(key1)+'.jpg', temp)
            #     for key2 in operation.keys():
            #         print('\t\t\t\t----- ', str(key2))
            #         temp = operation[key1].augment_image(image)
            #         temp = operation[key2].augment_image(temp)
            #         cv2.imwrite(image_name+'_'+str(key1)+'_'+str(key2)+'.jpg', temp)

print('Done')
