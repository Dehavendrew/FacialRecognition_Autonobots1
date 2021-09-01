import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.applications import imagenet_utils
from PIL import Image
from cryptography.fernet import Fernet

vid = cv2.VideoCapture(0)

model = tf.keras.models.load_model("faceDetector3_1.h5")

correct = 0
incorrect = 0
access_granted = True

key = ""
with open("key.txt", 'r') as file:
    key = file.readline()
cipher_suite = Fernet(key)


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
    #print('Correct-Person: {0} Non-Person: {1} Incorrect-Person: {2}'.format(preds[0][0], preds[0][1], preds[0][2]))


    if(preds[0][0] > 0.6):
        frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT,None,(0,255,0))
        correct = correct + 1
        incorrect = 0
    elif(preds[0][2] > 0.5):
        frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT,None,(0,0,255))
        incorrect = incorrect + 1
        correct = 0
    else:
        frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT,None,(255,0,0))
        correct = 0
        incorrect = 0

    cv2.imshow('frame', frame)

    if(correct == 20):
        access_granted = True
        break
    if(incorrect == 20):
        access_granted = False
        break

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

if(access_granted):
    print("ACCESS GRANTED")

    while(True):
        print("\nType ls to list all passwords")
        print("Type mkpwd <label> <username> <password> to make a new entry")
        print("Type exit to quit\n")
        cmd = input("$: ")
        if(cmd == "exit"):
            break
        if(cmd == "ls"):
            with open('vault.csv','r') as rfile:
                for line in rfile:
                    line = line.rstrip()
                    line = bytes(line, 'utf-8')
                    decoded_text = cipher_suite.decrypt(line)
                    print(str(decoded_text))
        if("mkpwd" in cmd):
            with open('vault.csv', 'a') as wfile:
                entry = cmd.split(' ')
                string =  entry[1] + ',' +  entry[2] + ',' + entry[3]
                string = bytes(string, 'utf-8')
                encoded_text = str(cipher_suite.encrypt(string))[1:]
                wfile.write(encoded_text + '\n')

else:
    print("ACCESS DENIED")
