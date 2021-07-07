# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 20:18:04 2021

@author: Param
"""

import os
image_path='D:\yolo object detection\OIDv4_ToolKit\OID\Dataset\train\Person_Car_Ambulance'
path_list=[]

for c_dir,dirs,files in os.walk('.'):
    for f in files:
        if f.endswith('.jpg'):
            file_loc=image_path+'/'+f
            path_list(file_loc+'\n')

paths_test=path_list[:,int(len(path_list)*0.20)]
path_list=path_list[int(len(path_list)*0.20):]

with open('train.txt','w') as train:
    for path in path_list:
        train.write(path)
        
with open('test.txt','w') as test:
    for path in paths_test:
        test.write(path) 
        
i=0
with open(image_path+'/'+'classes.names','w')as cls, \
     open(image_path+'/'+'classes.txt','r')as text:
     for line in text:
       cls.write(line)
       i+=1
       
with open(image_path+'/'+'image_data.data','w')as data:
  data.write('classes='+str(i)+'\n')
  data.write('train='+image_path+'/'+'train.txt'+'\n')
  data.write('valid='+image_path+'/'+'test.txt'+'\n')
  data.write('names='+image_path+'/'+'classes.names'+'\n')
  data.write('backup=backup')
       







        