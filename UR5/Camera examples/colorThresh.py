# 2.12 Lab 9: Image Processing
# Jacob Guggenheim 2019
# Jerry Ng 2019, 2020

import numpy as np
import cv2  # OpenCV module
import time
from tkinter import *
import threading
import math


global tk
tk = Tk()
global l_b, u_b, l_g, u_g, l_r, u_r
l_b = Scale(tk, from_ = 0, to = 255, label = 'Blue, lower', orient = HORIZONTAL)
l_b.pack()
u_b = Scale(tk, from_ = 0, to = 255, label = 'Blue, upper', orient = HORIZONTAL)
u_b.pack()
u_b.set(100)
l_g = Scale(tk, from_ = 0, to = 255, label = 'Green, lower', orient = HORIZONTAL)
l_g.pack()
u_g = Scale(tk, from_ = 0, to = 255, label = 'Green, upper', orient = HORIZONTAL)
u_g.pack()
u_g.set(255)
l_r = Scale(tk, from_ = 0, to = 255, label = 'Red, lower', orient = HORIZONTAL)
l_r.pack()
u_r = Scale(tk, from_ = 0, to = 255, label = 'Red, upper', orient = HORIZONTAL)
u_r.pack()
u_r.set(255)
global l_h, u_h, l_s, u_s, l_v, u_v
l_h = Scale(tk, from_ = 0, to = 180, label = 'Hue, lower', orient = HORIZONTAL)
l_h.pack()
u_h = Scale(tk, from_ = 0, to = 180, label = 'Hue, upper', orient = HORIZONTAL)
u_h.pack()
u_h.set(180)
l_s = Scale(tk, from_ = 0, to = 255, label = 'Saturation, lower', orient = HORIZONTAL)
l_s.pack()
u_s = Scale(tk, from_ = 0, to = 255, label = 'Saturation, upper', orient = HORIZONTAL)
u_s.pack()
u_s.set(255)
l_v = Scale(tk, from_ = 0, to = 255, label = 'Value, lower', orient = HORIZONTAL)
l_v.pack()
u_v = Scale(tk, from_ = 0, to = 255, label = 'Value, upper', orient = HORIZONTAL)
u_v.pack()
u_v.set(255)

class cvclass():
    def __init__(self):
        #threading.Thread.__init__(self)
        # Open Camera
        self.cap = cv2.VideoCapture(2)
        while True:
            tk.update()
            self.read_frame()

    def read_frame(self):
        # Read from the webcam, frame by frame
        ret, cv_image = self.cap.read()

        
        # visualize it in a cv window
        cv2.imshow("Original_Image", cv_image)
        cv2.waitKey(3)
        ################ RGB THRESHOLDING ####################
        #get threshold values
        
        lower_bound_RGB = np.array([l_b.get(), l_g.get(), l_r.get()])
        upper_bound_RGB = np.array([u_b.get(), u_g.get(), u_r.get()])

        # threshold
        mask_RGB = cv2.inRange(cv_image, lower_bound_RGB, upper_bound_RGB)

        # get display image
        disp_image_RGB = cv2.bitwise_and(cv_image,cv_image, mask= mask_RGB)
        cv2.imshow("RGB_Thresholding", disp_image_RGB)
        cv2.waitKey(3)


        ################ HSV THRESHOLDING ####################
        # conver to HSV
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # get threshold values
        lower_bound_HSV = np.array([l_h.get(), l_s.get(), l_v.get()])
        upper_bound_HSV = np.array([u_h.get(), u_s.get(), u_v.get()])

        # threshold
        mask_HSV = cv2.inRange(hsv_image, lower_bound_HSV, upper_bound_HSV)

        # get display image
        disp_image_HSV = cv2.bitwise_and(cv_image,cv_image, mask= mask_HSV)
        cv2.imshow("HSV_Thresholding", disp_image_HSV)
        cv2.waitKey(3)
        time.sleep(0.05)

        print(lower_bound_HSV)
        print(upper_bound_HSV)

if __name__=='__main__':

    cvapp = cvclass()


