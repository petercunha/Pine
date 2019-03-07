# Pine
Pine is an aimbot powered by real-time object detection with neural networks.

This software can be tweaked to work smoothly in CS:GO, Fortnite, and Overwatch. Pine also has built-in support for Nvidia's CUDA toolkit and is optimized to achieve extremely high object-detection FPS. It is GPU accelerated, cross-platform, and blazingly fast.



## Architecture

### YOLOv3: The Internal Object-Detection Engine
At its core, Pine's internal object detection system relies on the [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) neural network architecture and is trained on the COCO dataset to recognize human-like objects. This is useful because the decection engine can be abstracted to many FPS games, including CS:GO, TF2, and more.

![YOLO Network Speed Graph](https://pjreddie.com/media/image/map50blue.png)
