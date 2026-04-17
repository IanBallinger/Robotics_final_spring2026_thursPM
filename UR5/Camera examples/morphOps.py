# 2.12 Lab 7 object detection: a node for observing erosion/dilation
# Jacob Guggenheim 2019
# Jerry Ng 2019, 2020

import numpy as np
import cv2  # OpenCV module
import time
from tkinter import *

import math

tk = Tk()
l_h = Scale(tk, from_ = 0, to = 255, label = 'Hue, lower', orient = HORIZONTAL)
l_h.pack()
u_h = Scale(tk, from_ = 0, to = 255, label = 'Hue, upper', orient = HORIZONTAL)
u_h.pack()
u_h.set(255)
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


def main():
    # Open up the webcam
    cap = cv2.VideoCapture(2)
    while True:
        tk.update()

        # Read from the camera frame by frame
        ret, cv_image = cap.read()
        # visualize it in a cv window
        cv2.imshow("Original_Image", cv_image)
        cv2.waitKey(3)

        ################ HSV THRESHOLDING ####################
        # convert to HSV
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # get threshold values
        #lower_bound_HSV = np.array([l_h.get(), l_s.get(), l_v.get()])
        #upper_bound_HSV = np.array([u_h.get(), u_s.get(), u_v.get()])
        # optional TODO: input your HSV threshold values here
        

        plc_hsv = np.array([156, 83, 0])
        puc_hsv = np.array([180, 176, 143])
        ylc_hsv = np.array([ 13, 255, 120])
        yuc_hsv = np.array([ 98, 255, 208])
        glb_hsv = np.array([ 53,  90, 128])
        gup_hsv = np.array([ 87, 180, 221])
        tl_hsv = np.array([ 40,  71, 139])
        tu_hsv = np.array([ 56, 195, 255])
        tl_hsv = np.array([ 40,  71, 139])
        tu_hsv = np.array([ 56, 195, 255])
        rbl_hsv = np.array([  1, 180, 131])
        rbu_hsv = np.array([  3, 255, 236])

        lower_bound_HSV = glb_hsv
        upper_bound_HSV = gup_hsv



        # threshold
        mask_HSV = cv2.inRange(hsv_image, lower_bound_HSV, upper_bound_HSV)

        # display image
        cv2.imshow("HSV_Thresholding", mask_HSV)
        cv2.waitKey(3)

        # kernel for all morphological operations
        #TODO: Change size of kernel
        # Also, try changing the shape of the kernel (places 1's in certain locations). Try making a circle/line/etc.
        kernel = np.ones((5,5),np.uint8)

        # EXAMPLE OF A VERTICAL LINE:
        # kernel = np.array([[1, 1, 1, 1],\
        #                    [1, 0, 0, 1],\
        #                    [1, 0, 0, 1],\
        #                    [1, 1, 1, 1]], dtype=np.uint8)
        
        #TODO: Change number of iterations to see the effect.
        num_iterations = 3
        ################ Erosion ####################
        # erode blobs
        erosion = cv2.erode(mask_HSV,kernel,iterations = num_iterations)

        # display image
        # cv2.imshow("Erosion", erosion)
        # cv2.waitKey(3)

        ################ Dilation ####################
        # dilate blobs
        dilation = cv2.dilate(mask_HSV,kernel,iterations = num_iterations)

        # display image
        # cv2.imshow("Dilation", dilation)
        # cv2.waitKey(3)

        ################ Opening ####################
        # good for removing noise. its an erosion (to get rid of noise) followed by a dilation (to get back the original blobs you wanted to keep)
        opening = cv2.morphologyEx(mask_HSV, cv2.MORPH_OPEN, kernel, iterations = num_iterations)

        # display image
        cv2.imshow("Opening - Get rid of noise", opening)
        cv2.waitKey(3)

        ################ Closing ####################
        # good for filling small holes in blobs. its a dilation (to fill the holes) followed by an erosion (to get the object back to the right size)
        closing = cv2.morphologyEx(mask_HSV, cv2.MORPH_CLOSE, kernel, iterations = num_iterations)

        # display image
        cv2.imshow("Closing - Fill in blobs", closing)
        cv2.waitKey(3)

        result = cv2.bitwise_and(cv_image, cv_image, mask=mask_HSV)
        cv2.imshow("Masked", result)
        cv2.waitKey(3)

        # 4. Find contours
        contours, _ = cv2.findContours(mask_HSV, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #CM_PIXEL = 51.0 / 640
        # CM_PIXEL = 39.0 / 480
        # CM_PIXEL = 90.9 / (561-113)
        CM_PIXEL = 54.0 / 275
    

        # 5. Draw bounding box
        for cnt in contours:
            if cv2.contourArea(cnt) > 500: # Filter small noise
                x, y, w, h = cv2.boundingRect(cnt)
                x_c = x + int(w/2)
                y_c = y + int(h/2)
                cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(cv_image, (x_c, y_c), 4, (0, 255, 0), 2)
                x_cm = (x_c-337.5) * CM_PIXEL
                y_cm = (337.5-y_c) * CM_PIXEL

                text = f"x: {x_cm: .2f}, y: {y_cm: .2f}"
                cv2.putText(cv_image, text, (x_c - 10, y_c - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                

        cv2.rectangle(cv_image, (335, 335), (340, 340), (255, 255, 0), 2)
        cv2.rectangle(cv_image, (113, 205), (561,205), (255, 255, 0), 2)

        cv2.imshow('Bounding Box', cv_image)

        


if __name__=='__main__':
    main()
