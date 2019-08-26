import numpy as np
from termcolor import colored
import timeit
import _thread
import imutils
import time
import mss
import cv2
import os
import signal
import sys
import pynput
import ctypes
from lib.grab import grab_screen

sct = mss.mss()
Wd, Hd = sct.monitors[1]["width"], sct.monitors[1]["height"]
SendInput = ctypes.windll.user32.SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]
class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]
class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]
class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]
class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

def set_pos(x, y):
    x = 1 + int(x * 65536./Wd)
    y = 1 + int(y * 65536./Hd)
    extra = ctypes.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    ii_.mi = pynput._util.win32.MOUSEINPUT(x, y, 0, (0x0001 | 0x8000), 0, ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
    command=pynput._util.win32.INPUT(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))
if __name__ == "__main__":
    print("Do not run this file directly.")

def start(ENABLE_AIMBOT):

    # Config
    YOLO_DIRECTORY = "models"
    CONFIDENCE = 0.36
    THRESHOLD = 0.22

    #
    #   Size (in pixels) of the screen capture box to feed the neural net.
    #   This box is in the center of your screen. Lower value makes the network faster.
    #
    #   Example: "ACTIVATION_RANGE = 400" means a 400x400 pixel box.
    #
    ACTIVATION_RANGE = 250

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
    W, H = None, None
    origbox = (int(Wd/2 - ACTIVATION_RANGE/2),
               int(Hd/2 - ACTIVATION_RANGE/2),
               int(Wd/2 + ACTIVATION_RANGE/2),
               int(Hd/2 + ACTIVATION_RANGE/2))

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
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        cv2.ocl.useOpenCL()
        print(colored("[OKAY] OpenCL is working!", "green"))
    else:
        print(
            colored("[WARNING] OpenCL acceleration is disabled!", "yellow"))
    if "CUDA:YES" in build_info:
        print(colored("[OKAY] CUDA is working!", "green"))
    else:
        print(
            colored("[WARNING] CUDA acceleration is disabled!", "yellow"))

    print()

    # loop over frames from the video file stream
    while True:
        start_time = timeit.default_timer()
        frame = np.array(grab_screen(region=origbox))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[: 2]

        frame = cv2.UMat(frame)

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 260, (150, 150),
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

                # draw target dot on the frame
                cv2.circle(frame, (int(x + w / 2), int(y + h / 5)), 5, (0, 0, 255), -1)

                # draw a bounding box rectangle and label on the frame
                # color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y),
                                (x + w, y + h), (0, 0, 255), 2)

                text = "TARGET {}%".format(int(confidences[i] * 100))
                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if ENABLE_AIMBOT and bestMatch == confidences[i]:
                    mouseX = origbox[0] + (x + w/1.5)
                    mouseY = origbox[1] + (y + h/5)
                    set_pos(mouseX, mouseY)

        cv2.imshow("Neural Net Vision (Pine)", frame)
        elapsed = timeit.default_timer() - start_time
        sys.stdout.write(
            "\r{1} FPS with {0} MS interpolation delay \t".format(int(elapsed*1000), int(1/elapsed)))
        sys.stdout.flush()
        if cv2.waitKey(1) & 0xFF == ord('0'):
            break

    # Clean up on exit
    signal_handler(0, 0)
