# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 18:47:44 2021

@author: Param
"""
import pandas as pd
import os

os.chdir('D:\yolo object detection\OIDv4_ToolKit\OID\csv_folder')
class_data=pd.read_csv('class-descriptions-boxable.csv',header=None)

classes=['Ambulance','Car','Person']

uids=[]
for Class in classes:
    real_class=class_data.loc[class_data[1]==Class]
    uid=real_class.iloc[0][0]
    uids.append(uid)

annotn_data=pd.read_csv('train-annotations-bbox.csv',
                        usecols=['ImageID',
                                 'LabelName',
                                 'XMin',
                                 'XMax',
                                 'YMin',
                                 'YMax'])
new_class_data=annotn_data.loc[annotn_data['LabelName'].isin(uids)].copy()

new_class_data['classNumber']=''
new_class_data['center x']=''
new_class_data['center y']=''
new_class_data['width']=''
new_class_data['height']=''

for i in range(len(uids)):
    new_class_data.loc[new_class_data['LabelName']==uids[i],'classNumber']=i

new_class_data['center x']=(new_class_data['XMax']+new_class_data['XMin'])/2
new_class_data['center y']=(new_class_data['YMax']+new_class_data['YMin'])/2

new_class_data['width']=new_class_data['XMax']-new_class_data['XMin']
new_class_data['height']=new_class_data['YMax']-new_class_data['YMin']

yolo_vals=new_class_data.loc[:,['ImageID','classNumber','center x','center y','width','height']].copy()

os.chdir('D:\yolo object detection\OIDv4_ToolKit\OID\Dataset\train\Person_Car_Ambulance')
for c_dir,dirs,files in os.walk('.'):
    for f in files:
        if f.endswith('.jpg'):
            img_title=f[:-4]
            yolo_file=yolo_vals.loc[yolo_vals['ImageID']==img_title]
            df=yolo_file.loc[:,['classNumber','center x','center y','width','height']]
            save_path=img_title+'.txt'
            df.to_csv(save_path,header=False,index=False,sep=' ')


    
