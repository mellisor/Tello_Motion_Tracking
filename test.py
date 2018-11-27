import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
import imutils

import math
from time import sleep
from threading import Thread

from std_msgs.msg import Empty
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

bridge = CvBridge()

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

tracker = OPENCV_OBJECT_TRACKERS['csrt']()

class Tello_Motion(object):
        
        def __init__(self):

                # Image bounding block
                self.imbb = None
                # Is the bounding block being set?
                self.set = False
                # Should the program be running?
                self.running = True

                # Initialize node, create subscribers and publishers
                self.tak = rospy.Publisher('/tello/takeoff',Empty,queue_size=10)
                self.lan = rospy.Publisher('/tello/land',Empty,queue_size=10)
                self.cmd = rospy.Publisher('/tello/cmd_vel',Twist,queue_size=10)
                rospy.init_node('motion_tracker',disable_signals=True,anonymous=True)
                
                # Initialize tracker, create globals so tracking thread can communicate with image thread
                self.tracker = OPENCV_OBJECT_TRACKERS['csrt']()
                self.frame = None
                self.rectangle = None

                # Middle of the image and dead zones
                self.mid_x = 250
                self.mid_y = 180
                self.dz = 20

                # Create threads for tracking and ros
                self.tracker_thread = Thread(target=self.proc_image)

                sleep(1)
                # Trying to take off
                self.tak.publish()
                # Wait for drone to take off
                sleep(6)
                self.vid = rospy.Subscriber('/tello/image_raw', Image, self.im_callback)
                # While the program is running
                while self.running:
                        sleep(.05)
                # Land after the program is done
                self.lan.publish()
                exit()

        def im_callback(self,msg):
                # If there is no roi and it isn't already being set, set it
                if not self.set:
                        # Tell the program you're setting the roi
                        self.set = True
                        # Convert from ros image to opencv format, convert from bgr to rgb, and resize
                        im_msg = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                        im_msg = cv2.cvtColor(im_msg, cv2.COLOR_BGR2RGB)
                        im_msg = imutils.resize(im_msg,width=500)
                        # Select region of interest and close window after
                        self.imbb = cv2.selectROI('select',im_msg, showCrosshair=True)
                        cv2.destroyWindow('select')
                        # Initialize tracker with selected roi
                        self.tracker.init(im_msg,self.imbb)
                        self.frame = im_msg
                        # Start tracker thread
                        self.tracker_thread.start()
                        # Get midpoints of frame
                        self.mid_x = np.size(self.frame,1) / 2
                        self.mid_y = np.size(self.frame,0) / 3
                # If a bounding border is specified and the program is running
                if self.imbb and self.running:
                        # Modify image
                        im_msg = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                        im_msg = imutils.resize(im_msg,width=500)
                        self.frame = cv2.cvtColor(im_msg, cv2.COLOR_BGR2RGB)
                        # If the tracker finds the region
                        if self.rectangle:
                                # Get rectangle coordinates and size
                                (x, y, w, h) = [int(v) for v in self.rectangle]
                                cv2.rectangle(self.frame, (x, y), (x + w, y + h),
                                        (0, 255, 0), 2)
                                # Get center of rectangle
                                pos_x = x + w/2
                                pos_y = y + h/2
                                # Move drone according to rectangle position
                                vel = Twist()
                                # If it's too far right, move left 
                                if pos_x > self.mid_x + self.dz:
                                        vel.linear.y = -.2
                                # If too far left, move right
                                elif pos_x < self.mid_x - self.dz:
                                        vel.linear.y = .2
                                # If too low, move up
                                if pos_y > self.mid_y + self.dz:
                                        vel.linear.z = -.4
                                        print("Moving down")
                                # If too hight, move down
                                elif pos_y < self.mid_y - self.dz:
                                        vel.linear.z = .4
                                        print("Moving up")
                                # Publish velocity command
                                self.cmd.publish(vel)
                        # Show image
                        cv2.imshow('frame',self.frame)
                        # Press Q on keyboard to  exit
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                                cv2.destroyAllWindows()
                                self.running = False
                # If the program is terminating, kill it
                elif not self.running:
                        self.lan.publish()
                        rospy.signal_shutdown("q pressed")

        # While the program is running, update tracker
        def proc_image(self):
                while self.running:
                        (success,self.rectangle) = self.tracker.update(self.frame)
                        if not success:
                                self.rectangle = None
                print("leaving")



t_node = Tello_Motion()


