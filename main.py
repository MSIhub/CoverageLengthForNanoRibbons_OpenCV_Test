# Author: Mohamed sadiq Ikbal
# Date: 16 October 2018
# Purpose: To measure the coverage length of a nano ribbon

# test version 1

# Imports
import cv2

# reading image
img = cv2.imread('im/in/fig1.png', cv2.IMREAD_GRAYSCALE)

# processing

edgeCanny = cv2.Canny(img,100,100)

# showing image in cv
##
cv2.imshow('image', img)
cv2.imshow('edgeCanny',edgeCanny)
cv2.waitKey(0)  # waits for any key
cv2.destroyallwindows()     # close the image
##





#
# import numpy as np
# # import matplotlib.pyplot as plt
#
# #opencv is BGR
#
# # reading image
# img = cv2.imread('im/in/watch.jpg', cv2.IMREAD_COLOR)
#
# #IMREAD_COLOR = 1
# #IMREAD_GRAYSCALE = 0
# #IMREAD_UNCHANGED = -1
#
# # showing image in cv
# ##
# # cv2.imshow('image', img)
# # cv2.waitkey(0) # waits for any key
# # cv2.destroyallwindows() # close the image
# ##
#
# # showing image through matlplotlib and drawing a line on it
# ##
# # plt.imshow(img, cmap='gray', interpolation='bicubic')
# # plt.plot([50,100],[80,100], 'c', linewidth=5)
# # plt.show()
# ##
#
# # saving an image
# ##
#     # cv2.imwrite('im/out/watch_grey.png',img)
# ##
#
# # Video capturing from web cam or loading a video from file
# ##
# # cap = cv2.VideoCapture(0) # 0 = primary camera, 1 - secondary camera, or the video file name
# # fourcc = cv2.VideoWriter_fourcc(*'XVID') # codec definition
# # out = cv2.VideoWriter('ouput.avi', fourcc, 20.0,(640,480)) # size
# #
# # while True:
# #     ret, frame = cap.read() # ret - true or false , capturing the frame
# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # converting color to grayscale
# #     out.write(frame) # saving the video file
# #     cv2.imshow('frame',frame) # showing the video by frame
# #     cv2.imshow('gray',gray) # showing the grayscaled video
# #     if cv2.waitKey(1) & 0xFF == ord('q'): # wait for 1 key and the key is q
# #         break # while loop breaks
# #
# # cap.release() # releases the camera capturing
# # out.release()
# # cv2.destroyAllWindows() # closes the opened windows
# ##
#
# ## drawing shapes in the image
# # cv2.line(img, (0,0), (150,150), (0,1,198), 15) #bgr # line
# # cv2.rectangle(img, (15,25), (200,150),(0,255,0),5) # rectangle
# # cv2.circle(img, (100,63), 55, (0,0,255), -1) # circle # linewidh = -1 fills the circle
# # pts = np.array([[10,50],[200,300],[700,200],[500,100], [300,200]],np.int32)
# # #pts = pts.reshape((-1,1,2))
# # cv2.polylines(img ,[pts], True ,(0,255,255),5) # polylines # True= to close the last points
# # # Writing texts on a image
# # font = cv2.FONT_HERSHEY_SIMPLEX
# # cv2.putText(img, 'OpenCV Tuts!', (0,130), font, 5, (200,255,255), 5 , cv2.LINE_AA )
# ## showing the image
# # cv2.imshow('image', img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# # Accesing a particular pixel of an image and modifying it
# img[55,55] = [255,255,255]
# px = img[55,55]
# print(px)
#
# # ROI - region of image
#
# roi = img[100:150, 100:150]
# print(roi)
# img[100:150, 100:150] = [255,255,255]
#
# # copy and paste a portion of an image
# watch_face = img[37:111,107:194] #
# img[0:74, 0:87] = watch_face # 74 = 111-37 and 87 = 194-107 SHOULD BE THE SAME SIZE FOR COPY PASTE
#
# # showing the image
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
