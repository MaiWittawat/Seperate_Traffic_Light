#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge



def process_image_thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    return thresh

def detect_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=50
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
    
    return image, circles


def check_traffic_light_pattern(circles):
    if circles is None or len(circles) < 3:
        return False

    circles = sorted(circles, key=lambda c: c[1]) 
    distances = [circles[i+1][1] - circles[i][1] for i in range(len(circles) - 1)]
    
    if all(abs(distances[i] - distances[0]) < 10 for i in range(len(distances))):
        return True
    
    return False


def image_callback(msg):
    bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(msg, "bgr8")
    
    thresh = process_image_thresholding(image)
    
    processed_image, circles = detect_circles(image)
    
    is_traffic_light = check_traffic_light_pattern(circles)
    rospy.loginfo(f"Traffic Light Detected: {is_traffic_light}")
    
    # แสดงผล
    cv2.imshow("Processed Image", processed_image)
    cv2.waitKey(1)

def main():
    rospy.init_node('traffic_light_detector')
    rospy.Subscriber('/usb_cam/image_raw.compressed', Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
