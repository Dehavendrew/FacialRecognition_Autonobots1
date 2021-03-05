import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.applications import imagenet_utils
from PIL import Image

#Create Video Capture utility
vid = cv2.VideoCapture(0)

#load the mobileNet modile
model = tf.keras.applications.MobileNet()

while(True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()


    #Convert from CV2 format to Tensorflow format
    model_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    model_frame = Image.fromarray(model_frame)
    model_frame = model_frame.resize((224, 224))
    model_frame= image.img_to_array(model_frame)
    model_frame = np.expand_dims(model_frame, axis = 0)
    model_frame = tf.keras.applications.mobilenet.preprocess_input(model_frame)

    #Predict the objects in the image
    preds = model.predict(model_frame)

    #Print the predictions and display the image
    print('Predicted:', decode_predictions(preds, top=3)[0])
    cv2.imshow('frame', frame)
    
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
