import shutil, os
import pandas as pd


data = pd.read_csv('HAM10000_metadata.csv')
data.info()
root_path = 'locat'
img_path = 'HAM10000_images'

for index, row in data.iterrows():
    path = root_path+'\\'+row.localization+'_'+row.dx
    if not os.path.exists(path):
        os.mkdir(path)
        print(path)
        shutil.copy(img_path+'\\'+row.image_id+'.jpg', path)
    else:
        print(path)
        shutil.copy(img_path+'\\'+row.image_id+'.jpg', path)
