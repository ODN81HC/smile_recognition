import cv2
import numpy as np
import tensorflow as tf
from main import CNNFig1

# Load model
model = CNNFig1(1)
dummy_x = tf.zeros((1, 64, 64, 3))
model._set_inputs(dummy_x)
model.build(input_shape=dummy_x.shape)
model.load_weights("my_model.h5")
print("Model has been loaded")

# Run the camera
# cap = cv2.VideoCapture('video_recorder.mp4')
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("The camera/video cannot be loaded. Check again!")
else:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        else:
            # Frame resize
            h, w = frame.shape[:2]
            hyper_parameter = 150
            frame_cropped = frame[int(h/2) - hyper_parameter: int(h/2)+ hyper_parameter, int(w/2)-hyper_parameter: int(w/2)+hyper_parameter]
            h, w = frame.shape[:2]
            resized = cv2.resize(frame_cropped, (64, 64))
            # Frame normalize
            resized = resized.astype('float32')/255.0
            resized.resize((1, 64, 64, 3))
            # Model predict
            pred = model.predict(resized)
            value = pred[0][0]
            if value > 0.5:
                cv2.putText(frame, "Smile", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Not Smile", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # Show frame
            cv2.imshow('Output window', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    cap.release()
    cv2.destroyAllWindows()