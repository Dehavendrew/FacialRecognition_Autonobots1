import cv2
import os
import numpy as np
import time

#Start the video Capture Feed
vid = cv2.VideoCapture(0)

photo_idx = 0

#Name of the folder where you want to store your photos.
className = "Person"


while(True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()


    #display photo to the screen
    print('Photos Taken: ' + str(photo_idx) )
    cv2.imshow('frame', frame)
    frame = cv2.resize(frame,(224,224))

    #save photo as a jpg
    fileName = className + "\img_person_3_" + str(photo_idx) + ".jpg"
    cv2.imwrite(fileName, frame)



    photo_idx += 1
    #add delay of 1 second between photos
    time.sleep(0.5)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
