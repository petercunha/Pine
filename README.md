# Pine
Pine is an aimbot powered by real-time object detection with neural networks.

This software can be tweaked to work smoothly in CS:GO, Fortnite, and Overwatch. Pine also has built-in support for Nvidia's CUDA toolkit and is optimized to achieve extremely high object-detection FPS. It is GPU accelerated, cross-platform, and blazingly fast.



## Architecture

### YOLOv3: The Internal Object-Detection Engine

At its core, Pine's internal object detection system relies on a modified version of the [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) neural network architecture. The detection network is trained on a combination of video game images and the COCO dataset. It is optimized to recognize human-like objects as quickly as possible. This is useful because the decection engine can be abstracted to many FPS games, including CS:GO, TF2, and more.

![YOLO Neural Network](https://i.imgur.com/0edTFBP.jpg)

YOLO's inference time puts RetinaNet to shame...

![YOLO Network Speed Graph](https://pjreddie.com/media/image/map50blue.png)


### OpenCV: GPU-Enabled Image Processing

OpenCV is at the heart of Pine's image processing capabilities. Not only does it provide an abstraction layer for working with the screen capture data, but it it also allows us to harness the power of GPU hardware by interfacing CUDA and OpenCL. After Pine takes a capture of the user's screen, it uses OpenCV to process that image into a form that is recognizable to the object-detection engine.

![OpenCV Image Processing Architecture](https://i.imgur.com/n3LgS6T.png)
