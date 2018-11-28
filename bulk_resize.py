import cv2
import os
import glob
# import argparse
import re  # regexp



# parser = argparse.ArgumentParser()
# args = parser.parse_args()

def bulk_resize_image(path, file_extension=".jpg"):
    for cat_folder in os.listdir(path):
        print(cat_folder)
        for sub_cat_folder in os.listdir(path+'//'+cat_folder):
            print('\n', sub_cat_folder)
            for image_path in glob.glob(path + "/" + cat_folder + "/" + sub_cat_folder + "/*.jpg"):
                # image_path = image_path_raw.split(path + "/" + cat_folder + "/" + sub_cat_folder + "/")[1]
                print(image_path)
                img = cv2.imread(image_path)
                y, x, c = img.shape
                dsize = (int(x/2), int(y/2))
                img_resize = cv2.resize(img, dsize)
                # cv2.imwrite(image_path+"_resize"+file_extension=".jpg", img_resize)
                cv2.imwrite(image_path, img_resize)
    print("Done")



bulk_resize_image(path='F:\FSR\dataset\skin-cancer-mnist-ham10000\Locat')
