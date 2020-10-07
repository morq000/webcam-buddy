import os
import numpy as np
import cv2
import AIMirror.config as config


def recognize(face_id, user_dict, get_eyes=False, get_smile=False):

    # Get video capture from webcam
    capture = cv2.VideoCapture(0)
    capture.set(3, 640)  # set Width
    capture.set(4, 480)  # set Height
    font_face = cv2.FONT_HERSHEY_SIMPLEX

    # Load face cascade and smile and eyes cascade if needed
    face_cascade = cv2.CascadeClassifier(os.path.join(config.cascades_path, 'haarcascade_frontalface_default.xml'))
    if get_eyes:
        eye_cascade = cv2.CascadeClassifier(os.path.join(config.cascades_path, 'haarcascade_eye.xml'))
    if get_smile:
        smile_cascade = cv2.CascadeClassifier(os.path.join(config.cascades_path, 'haarcascade_smile.xml'))

    count = 0  # frame counter, loop stops after N times

    # set infinite loop for webcam stream
    while count < 100:
        ret, frame = capture.read()

        # create a grayscale version
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # create detected faces list
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=10,
            minSize=(20, 20))

        # iterate over faces list
        for (x, y, w, h) in faces:
            count += 1

            # roi = area of an image with a face. create color and grayscale versions
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imwrite(os.path.join(config.dataset_dir, ("User." + str(face_id) + '.' + str(count) + ".jpg")), roi_gray)

            if get_eyes:
                # create eye cascade
                eyes = eye_cascade.detectMultiScale(roi_gray, minNeighbors=3)
                eye_count = 0
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    eye_count += 1
                    if eye_count == 2:
                        break

            if get_smile:
                smile = smile_cascade.detectMultiScale(roi_gray, minNeighbors=15)
                for (sx, sy, sw, sh) in smile:
                    if sy < (h / 2):
                        pass
                    else:
                        cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
                        break

        cv2.putText(frame, 'Picture ' + str(count) + ' of 100 for user ' + user_dict[face_id], (30, 50), font_face, 1.2,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        k = cv2.waitKey(200) & 0xff  # if cv2.waitKey(1) & 0xFF == ord('q'):
        if k == 27:  # press 'ESC' to quit
            break

    capture.release()
    cv2.destroyAllWindows()
