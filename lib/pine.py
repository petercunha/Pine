
import numpy as np
import pyautogui
from termcolor import colored
import _thread
import imutils
import time
import mss
import cv2
import os
import signal
import sys

if __name__ == "__main__":
    print("Do not run this file directly.")


def moveMouse(x, y):
    pyautogui.moveTo(x, y)


def start(ENABLE_AIMBOT):

    # Config
    YOLO_DIRECTORY = "models"
    CONFIDENCE = 0.35
    THRESHOLD = 0.2

    #
    #   Size (in pixels) of the screen capture box to feed the neural net.
    #   This box is in the center of your screen. Lower value makes the network faster.
    #
    #   Example: "ACTIVATION_RANGE = 400" means a 400x400 pixel box.
    #
    ACTIVATION_RANGE = 400

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

    # Wait for buffering
    time.sleep(0.4)

    # load our YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    print("[INFO] loading neural-network from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Define screen capture area
    print("[INFO] loading screencapture device...")
    sct = mss.mss()
    W, H = None, None
    Wd, Hd = sct.monitors[1]["width"], sct.monitors[1]["height"]
    origbox = (Wd/2 - ACTIVATION_RANGE/2, Hd/2 - ACTIVATION_RANGE/2,
               Wd/2 + ACTIVATION_RANGE/2, Hd/2 + ACTIVATION_RANGE/2)

    # Log whether aimbot is enabled
    if not ENABLE_AIMBOT:
        print("[INFO] aimbot disabled, using visualizer only...")
    else:
        print(colored("[OKAY] Aimbot enabled!", "green"))

    # Handle Ctrl+C in terminal, release pointers
    def signal_handler(sig, frame):
        # release the file pointers
        print("\n[INFO] cleaning up...")
        sct.close()
        cv2.destroyAllWindows()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # Test for GPU support
    build_info = str("".join(cv2.getBuildInformation().split()))
    if "OpenCL:YES" in build_info:
        print(colored("[OKAY] OpenCL is working!", "green"))
    else:
        print(
            colored("[WARNING] OpenCL acceleration is disabled!", "yellow"))
    if "CUDA:YES" in build_info:
        print(colored("[OKAY] CUDA is working!", "green"))
    else:
        print(
            colored("[WARNING] CUDA acceleration is disabled!", "yellow"))

    # loop over frames from the video file stream
    while True:

        start = time.process_time()

        frame = np.array(sct.grab(origbox))
        frame = cv2.resize(frame, (ACTIVATION_RANGE, ACTIVATION_RANGE))
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

                if ENABLE_AIMBOT and bestMatch == confidences[i]:
                    mouseX = origbox[0] + x + w/2
                    mouseY = origbox[1] + y + h/5
                    # _thread.start_new_thread(moveMouse, (mouseX, mouseY))
                    moveMouse(mouseX, mouseY)

        cv2.imshow("Pine", frame)

        elapsed = time.process_time() - start
        print(int(elapsed * 1000), "ms")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up on exit
    signal_handler(0, 0)
