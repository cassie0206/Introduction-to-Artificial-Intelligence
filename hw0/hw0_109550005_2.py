# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 21:11:45 2022

@author: ｃａｓｓｉｅ
"""

import cv2
import numpy as np

cap = cv2.VideoCapture("video.mp4")




while(1):
    img1 = cap.read()[1]
    img2 = cap.read()[1]


    result = cv2.absdiff(img1, img2)

    b,g,r=cv2.split(result)
    zeros = np.zeros(result.shape[:2],dtype = "uint8")
    result=cv2.merge([zeros,g,zeros])
    
    
    
    finalResult = np.hstack((img2, result))
    
   
   
    cv2.imshow("diffShow", finalResult)
    
    
    if cv2.waitKey(30) == 27:
        cv2.imwrite("hw0_109550005_2.png",finalResult)
        break
    
   
   
    


cap.release()
cv2.destroyAllWindows()