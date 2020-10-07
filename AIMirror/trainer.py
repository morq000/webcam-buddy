import os
import numpy as np
import cv2
# from PIL import Image
import AIMirror.config as config


def get_ids_and_images(path):
    img_paths = [os.path.join(path, file) for file in os.listdir(path)]
    images = []
    ids = []

    for img_path in img_paths:
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        # pil_img = Image.open(img_path).convert('L')  # convert to grayscale
        # img_np = np.array(img, 'uint8')
        uid = int(os.path.split(img_path)[-1].split('.')[1])
        # print(uid)
        images.append(img)
        ids.append(uid)

    return images, ids


def train_recognizer(recog_set):

    faces, ids = get_ids_and_images(config.dataset_dir)
    recognizer = recog_set["recognizer"]
    faces = [cv2.resize(face, (180, 180)) for face in faces]
    recognizer.train(faces, np.array(ids))
    recognizer.write(os.path.join(config.trainer_dir, recog_set["YML_name"]))
    print("Recognizer trained and written to file")
