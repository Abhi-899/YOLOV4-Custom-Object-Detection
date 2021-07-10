# YOLOV4-Custom-Object-Detection
In this project we will train the YOLOv4 network on 3 classes 'Ambulance' , 'Car' , 'Person' with the Google open image dataset  and run the detection on a real video caught on a moving traffic camera.

Let us look at the YOLOV4 model.
## The Model
YOLO stands for You Only Look Once. It’s an object detection model used in deep learning use cases, of which there are mainly 2 main families:
1) Two-Stage Detectors

2) One-Stage Detectors

YOLO belongs to the family of One-Stage Detectors. In a sliding window + classification approach, you look at the image and classify it for every window.Compared to YOLOv3,  YOLOv4 has improved again in terms of accuracy (average precision) and speed (FPS), the two metrics we generally use to qualify an object detection algorithm as shown in the below graph:
![image](https://user-images.githubusercontent.com/64439578/124858782-565a1a00-dfcc-11eb-9618-bac7b209fb9c.png)
And the best part of the YOLOv4 model is that model training can be done on a single GPU.
Given below is the architecture of the model:
![image](https://user-images.githubusercontent.com/64439578/124859278-40008e00-dfcd-11eb-816c-a0aa09d0fe9b.png)
For more one can look into the YOLOv4 model paper.THe link is given below:

https://arxiv.org/pdf/2004.10934.pdf
## Google OID
A dataset of ~9 million varied images with rich annotations.
The images are very diverse and often contain complex scenes with several objects (8.4 per image on average). It contains image-level labels annotations, object bounding boxes, object segmentations, visual relationships, localized narratives, and more. It contains 

-> 15,851,536 boxes on 600 categories

-> 2,785,498 instance segmentations on 350 categories

-> 3,284,280 relationship annotations on 1,466 relationships

-> 675,155 localized narratives

-> 59,919,574 image-level labels on 19,957 categories

-> Extension - 478,000 crowdsourced images with 6,000+ categories

To download a dataset from OID:

Step 1: Install git

Step 2: Open your command prompt or anaconda prompt and write the following commands:
         ```
         git clone https://github.com/EscVM/OIDv4_Toolkit.git
         python main.py downloader --classes Ambulance Car Person --type_csv train --multiclass 1 --limit 100
         ```
Step3: For running on google colab
       ```
       !git clone https://github.com/EscVM/OIDv4_Toolkit.git
       !python main.py downloader --classes Ambulance Car Person --type_csv train --multiclass 1 --limit 100
       ```
One will get the OID repo here: https://g.co/dataset/open-images       
### NOTE
In case you want to download multiple images at once through google search. Use
```
!pip install icrawler
from icrawler.builtin import GoogleImageCrawler
google_Crawler = GoogleImageCrawler(storage = {'root_dir': r'write the name of the directory you want to save to here'})
google_Crawler.crawl(keyword = 'sad human faces', max_num = 800)
```
## Building the detector
The first thing we have to do is cloning and building the darknet.The following cells will clone darknet from AlexeyAB's famous repository, adjust the Makefile to enable OPENCV and GPU for darknet and then build darknet.
```
 !git clone https://github.com/AlexeyAB/darknet
 ```
Now we have to make the dataset directory structure suitable for YOLOv4. So first we have to run the annotation.py and the( train and test creation.py) which would create text files containing the data about the images.
```
!python annotation.py
!python train and test creation.py
```
Each image file will have a corresponding text file named <image name>.txt along with the train.txt , the test.txt files wuth the absolute paths of the train and test images. It also creates the image_data.data file which contains the no. of classes, paths to train.txt and test.txt and also the backup folder where the weights files for checkpoints after every 1000 steps will be stored along with the weights of the last checkpoint.

## Configuring Files for Training   
Now we need to edit the .cfg to fit our needs based on our object detector. Open it up in a code or text editor to do so.

If we downloaded cfg to google drive we can use the built in Text Editor by going to our google drive and double clicking on yolov4-obj.cfg and then clicking on the Open with drop down and selectin Text Editor.
![image](https://user-images.githubusercontent.com/64439578/124907089-63472f80-e005-11eb-9c87-8ac5a8206cfb.png)
We recommend having batch = 64 and subdivisions = 16 for ultimate results. If we run into any issues then up subdivisions to 32.

Make the rest of the changes to the cfg based on how many classes you are training your detector on.

Note: We set my max_batches = 6000, steps = 4800, 5400, I changed the classes = 1 in the three YOLO layers and filters = 18 in the three convolutional layers before the YOLO layers.

How to Configure Your Variables:

width = 416

height = 416 (these can be any multiple of 32, 416 is standard, you can sometimes improve results by making value larger like 608 but will slow down training)

max_batches = (# of classes) * 2000 (but no less than 6000 so if you are training for 1, 2, or 3 classes it will be 6000, however detector for 5 classes would have max_batches=10000)

steps = (80% of max_batches), (90% of max_batches) (so if your max_batches = 10000, then steps = 8000, 9000)

filters = (# of classes + 5) * 3 (so if you are training for one class then your filters = 18, but if you are training for 4 classes then your filters = 27)         

## Training
First we have to download the pre-trained weights for convolution layers.
```
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
```
Now run the following command to train the model. The -dont_show flag stops streaming during training
```
!./darknet detector train data/Person_Car_Ambulance/image_data.data cfg/yolov4_train.cfg yolov4.conv.137 -dont_show 
```
## Testing
We will be testing our model on a video taken by a moving traffic camera.
```
!./darknet detector demo data/Person_Car_Ambulance/image_data.data cfg/yolov4_test.cfg /content/drive/MyDrive/YOLO_V4/darknet/backup/yolov4_train_last.weights -dont_show /content/drive/MyDrive/YOLO_V4/test.mp4  -i 0 -out_filename /content/drive/MyDrive/YOLO_V4/output.avi   
```
Note that you would have to change your class names and paths given according to your model. Now remember that you have to give permission to your YOLO model for execution. That can be done using the following code:
```
os.chdir('/content/drive/MyDrive/YOLO_V4/darknet')
!sudo chmod +x darknet
!./darknet
```
## OUTPUT:
![Screenshot (6)](https://user-images.githubusercontent.com/64439578/124911234-2467a880-e00a-11eb-82d1-45b3e8f39c92.png)

## NOTE :
The testing could have been done better by adding the -thresh flag. We could easily see after running the output1.avi that a black van was also detected as an ambulance but with a lower threshold. Now that can be improved by:
        
1) Training with more ambulance images
         
2) Setting a higher threshold during testing 
         
3) If more images are not available we can also augment the images and use labelimg to label the images and create the corresponding annotation file
         
Now, YOLOv4 can also return the co-ordinates of the bounding boxes. In order to know how to perform object tracking and counting in a given area, Please click on the link below to my repository on vehicle tracking and counting:
         
https://github.com/Abhi-899/Object-tracking-and-counting-in-given-region

You can output bounding box coordinates for each detection with the flag '-ext_output'. This external outputs flag will give you a few extra details about each detection within an image.
```
!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/person.jpg -ext_output
```         
         
