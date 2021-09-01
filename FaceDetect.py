import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.applications import imagenet_utils
from PIL import Image

vid = cv2.VideoCapture(0)

model = tf.keras.models.load_model("faceDetector3_1.h5")

while(True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    model_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    # Display the resulting frame
    model_frame = Image.fromarray(model_frame)
    model_frame = model_frame.resize((224, 224))
    model_frame= image.img_to_array(model_frame)
    model_frame = np.expand_dims(model_frame, axis = 0)
    model_frame = tf.keras.applications.mobilenet.preprocess_input(model_frame)

    preds = model.predict(model_frame)
    print('Correct-Person: {0} Non-Person: {1} Incorrect-Person: {2}'.format(preds[0][0], preds[0][1], preds[0][2]))
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
