# For the Terminal

import cv2 
import pandas as pd
import numpy as np
import math
from ultralytics import YOLO


def bounding_box(img, bbox, l=30, t=5, rt=1, color=(255, 0, 255)):
    cv2.rectangle(img, bbox, color, rt)
    
def detect_light_color(cropped_img):
    """
    Detect the color of the traffic light in the cropped image
    """
    # Convert to HSV
    hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for red, yellow, and green
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    green_lower = np.array([40, 100, 100])
    green_upper = np.array([70, 255, 255])

    # Create masks for each color
    red_mask = cv2.inRange(hsv_img, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv_img, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv_img, green_lower, green_upper)

    # Count the number of pixels in each mask
    red_pixels = cv2.countNonZero(red_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    green_pixels = cv2.countNonZero(green_mask)

    # Determine the color with the most pixels
    if red_pixels > yellow_pixels and red_pixels > green_pixels:
        return "Red"
    elif yellow_pixels > red_pixels and yellow_pixels > green_pixels:
        return "Yellow"
    elif green_pixels > red_pixels and green_pixels > yellow_pixels:
        return "Green"
    else:
        return "Unknown"
    

# Load Model
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
classNames = model.names # get the all className from model

# Tag Parameter
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1 
thickness = 2 # Line thickness of 2 px
color = (255, 0, 0) # Blue color in BGR

while (cap.isOpened()):
    success, img = cap.read()
    if not success:
        print("fail to open the camera")
    
    results = model.track(img, stream=True, conf=0.3)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # confidence
            conf = math.ceil((box.conf[0] * 100))/100
            
            # className
            cls = int(box.cls[0])
            # print(classNames[cls])

            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)          
            w, h = x2 - x1, y2 - y1
            
            if classNames[cls] == "traffic light":
                
                bounding_box(img, (x1, y1, w, h))    
                # Crop the bounding box
                cropped_img = img[y1:y2, x1:x2]

                # Detect the color of the traffic light
                light_color = detect_light_color(cropped_img)
                img = cv2.putText(img, f'{classNames[cls]} Color: {light_color}', (max(0, x1), max(0, y1)), font, font_scale, color, thickness)
            
    # Display the video
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break        

cap.release()
cv2.destroyAllWindows()        