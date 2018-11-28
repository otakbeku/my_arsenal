import os
import glob

f = open('keterangan.txt', 'w+')
f.write('==============================================\n')
f.write('================ KETERANGAN ==================\n')
f.write('==============================================\n')
def count_image(path, file_extension=".jpg"):
    for cat_folder in os.listdir(path):
        f.write('Kelas: '+cat_folder+'\n')
        print(cat_folder)
        for sub_cat_folder in os.listdir(path+'//'+cat_folder):
            f.write('\t'+sub_cat_folder+' '+str(len(glob.glob(path + "/" + cat_folder + "/" + sub_cat_folder + "/*.jpg")))+'\n')
            print('\t', sub_cat_folder)
            print('\t\t',len(glob.glob(path + "/" + cat_folder + "/" + sub_cat_folder + "/*.jpg")))
            # for image_path in glob.glob(path + "/" + cat_folder + "/" + sub_cat_folder + "/*.jpg"):

    print("Done")
    f.write('==============================================')
    f.close()



count_image(path='F:\FSR\dataset\skin-cancer-mnist-ham10000\Locat')
