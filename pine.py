
# import the necessary packages
import numpy as np
import pyautogui
from termcolor import colored
import imutils
import time
import mss
import cv2
import os
import signal
import sys

print('''
====================================
 Pine: Neural-Network Aimbot (v0.1)
====================================
''')

print("[INFO] press 'q' to quit or ctrl+C in console...")
time.sleep(0.4)

YOLO_DIRECTORY = "models"
CONFIDENCE = 0.35
THRESHOLD = 0.2
SCREENCAP_SIZE = 400

# If true, mouse will be moved to target automatically
ENABLE_MOUSE = True

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([YOLO_DIRECTORY, "coco-dataset.labels"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([YOLO_DIRECTORY, "yolov3-tiny.weights"])
configPath = os.path.sep.join([YOLO_DIRECTORY, "yolov3-tiny.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading neural-network from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

print("[INFO] neural-network is loaded...")

# screenshot tool
sct = mss.mss()

# define screen subspace
W, H = None, None
Wd, Hd = sct.monitors[1]["width"], sct.monitors[1]["height"]
origbox = (Wd/2 - SCREENCAP_SIZE/2, Hd/2 - SCREENCAP_SIZE/2,
           Wd/2 + SCREENCAP_SIZE/2, Hd/2 + SCREENCAP_SIZE/2)

print("[INFO] screen access successful...")

if not ENABLE_MOUSE:
    print("[INFO] aim control disabled, using visualizer only...")
else:
    print(colored("[OKAY] Aimbot enabled!", "green"))


def signal_handler(sig, frame):
    # release the file pointers
    print("\n[INFO] cleaning up...")
    sct.close()
    cv2.destroyAllWindows()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

# print("[INFO] testing for GPU acceleration support...")
build_info = str("".join(cv2.getBuildInformation().split()))

if "OpenCL:YES" in build_info:
    print(colored("[OKAY] OpenCL is working!", "green"))
else:
    print(colored("[WARNING] OpenCL acceleration is disabled!", "yellow"))

if "CUDA:YES" in build_info:
    print(colored("[OKAY] CUDA is working!", "green"))
else:
    print(colored("[WARNING] CUDA acceleration is disabled!", "yellow"))

# loop over frames from the video file stream
while True:

    start = time.process_time()

    frame = np.array(sct.grab(origbox))
    frame = cv2.resize(frame, (SCREENCAP_SIZE, SCREENCAP_SIZE))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[: 2]

    frame = cv2.UMat(frame)

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 260.0, (416, 416),
                                 swapRB=False, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]

            # classID = np.argmax(scores)
            # confidence = scores[classID]
            classID = 0  # person = 0
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0: 4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

    # ensure at least one detection exists
    if len(idxs) > 0:

        # Find best player match
        bestMatch = confidences[np.argmax(confidences)]

        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y),
                          (x + w, y + h), (0, 0, 255) if bestMatch == confidences[i] else color, 2)
            # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            text = "{}: {:.4f}".format("Player", confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if ENABLE_MOUSE and bestMatch == confidences[i]:
                pyautogui.moveTo(origbox[0] + x + w/2, origbox[1] + y + h/4)

    cv2.imshow("Pine", frame)

    elapsed = time.process_time() - start
    # print(int(elapsed * 1000), "ms")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# release the file pointers
print("[INFO] cleaning up...")
sct.close()
cv2.destroyAllWindows()
