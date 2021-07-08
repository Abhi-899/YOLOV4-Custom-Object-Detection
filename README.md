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
