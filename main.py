import cv2
import numpy as np
from keras.models import load_model


model = load_model("theft_detection_model.h5")


labels = {0: 'Not theft', 1: 'Theft'}

while True:
    # Grab the current frame
    frame = cv2.imread(0)


    if frame is None:
        break

    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype("float32") / 255.0
    frame = np.expand_dims(frame, axis=0)


    prediction = model.predict(frame)


    label = labels[int(np.argmax(prediction[0]))]

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, label, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)


    cv2.imshow("Theft Detection", frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        cap.release( )


    cv2.destroyAllWindows( )