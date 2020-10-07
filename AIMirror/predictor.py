import os
import numpy as np
import cv2
import AIMirror.config as config
import AIMirror.__main__ as main


def predict_user(recog_set):
    capture = cv2.VideoCapture(0)
    capture.set(3, 640)  # set Width
    capture.set(4, 480)  # set Height
    recognizer = recog_set["recognizer"]
    recognizer.read(os.path.join(config.trainer_dir, recog_set["YML_name"]))
    face_cascade = cv2.CascadeClassifier(os.path.join(config.cascades_path, 'haarcascade_frontalface_default.xml'))
    font_face = cv2.FONT_HERSHEY_SIMPLEX

    while True:

        ret, frame = capture.read()
        roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(roi_gray,
                                              scaleFactor=1.2,
                                              minNeighbors=3,
                                              minSize=(round(capture.get(3) * 0.2), round(capture.get(4) * 0.2)))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            resized = cv2.resize(roi_gray[y:y + h, x:x + w], (180, 180))
            uid, confidence = recognizer.predict(resized, )

            if confidence < 50:
                u_name = main.user_dict[str(uid)]
            else:
                u_name = "Unknown"

            cv2.putText(frame, u_name + " " + str(round(100 - confidence)) + "%", (x, y - 10), font_face, 1,
                        (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Camera', frame)

        k = cv2.waitKey(30) & 0xff  # if cv2.waitKey(1) & 0xFF == ord('q'):
        if k == 27:  # press 'ESC' to quit
            break

    capture.release()
    cv2.destroyAllWindows()