# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 03:50:37 2022

@author: ｃａｓｓｉｅ
"""

import cv2
import numpy as np



img = cv2.imread('image.png')


red_color = (0, 0, 255) # BGR
cv2.rectangle(img, (608, 616), (721, 505), red_color, 3, cv2.LINE_AA)
cv2.rectangle(img, (836, 557), (916, 477), red_color, 3, cv2.LINE_AA)
cv2.rectangle(img, (1073, 726), (1180, 600), red_color, 3, cv2.LINE_AA)
cv2.rectangle(img, (985, 468), (1042, 417), red_color, 3, cv2.LINE_AA)
cv2.rectangle(img, (994, 383), (1042, 346), red_color, 3, cv2.LINE_AA)
cv2.rectangle(img, (1042, 314), (1088, 282), red_color, 3, cv2.LINE_AA)





cv2.imshow('image', img)

cv2.imwrite('hw0_109550005_1.png', img)
cv2.waitKey(0)






"""

red_color = (0, 0, 255) # BGR
cv2.rectangle(img, (608, 616), (721, 505), red_color, 3, cv2.LINE_AA)

cv2.imshow('Result', image)
cv2.waitKey(0)


plt.show()
"""


cv2.destroyAllWindows()

