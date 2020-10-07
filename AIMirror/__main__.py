import os
import cv2
import json
from AIMirror import config, learner, trainer, predictor


user_dict = dict()  # contains generated hashed user_id and user name entered via UI


# Load cascade classifiers
# face_cascade = cv2.CascadeClassifier(os.path.join(cascades_path, 'haarcascade_frontalface_default.xml'))
# eye_cascade = cv2.CascadeClassifier(os.path.join(cascades_path, 'haarcascade_eye.xml'))
# smile_cascade = cv2.CascadeClassifier(os.path.join(cascades_path, 'haarcascade_smile.xml'))

# Перемещаемся в рабочую папку
os.chdir(config.project_dir)
print("Working in directory: %s" % os.getcwd())


def save_db(file=config.users_list_name):
    print("Writing user database")
    with open(file, 'w') as f:
        f.write(json.dumps(user_dict))


def read_db():
    # check if file with user ids exists
    if os.path.exists(config.users_list_name):
        with open(config.users_list_name, 'r') as f:
            temp_dict = json.loads(f.read())
            if temp_dict == {}:
                print("Database file is empty")
            else:
                print("Loaded user database")
    else:
        # create new file
        with open(config.users_list_name, 'w') as f:
            f.write('')
            temp_dict = dict()
            print("Created new database file")

    return temp_dict


def check_db_and_pictures(dataset=config.dataset_dir):
    """Функция прочесывает папку с фотографиями и ищет те, ID которых отсутствует в базе пользователей
        если находит - выдает диалог Добавить пользователя или нет
        если да - запрашивает имя, заносит в БД и сохраняет ее
        Если нет - удаляет все фото с этим ID"""

    # global user_dict
    for image_name in os.listdir(dataset):
        uid = image_name.split('.')[1]

        # проверка, есть ли файл на месте, т.к. он может быть удален предыдущей итерацией, а имя остается в итераторе
        # также выполняется основная проверка - есть ли ID в базе пользователей
        if os.path.exists(os.path.join(dataset, image_name)) and (uid not in user_dict.keys()):

            print("Found picture of unknown user")
            image = cv2.imread(os.path.join(dataset, image_name))
            cv2.imshow('Unknown user', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            key = input('Save this user to DB? Y/N')
            while key not in ['Y', 'N']:
                key = input('Save this user to DB? Y/N')
            if key == 'Y':
                user_dict[uid] = input('Enter user name: ')
                save_db()
            else:
                for image_to_delete in os.listdir(dataset):
                    if image_to_delete.split('.')[1] == uid:
                        os.remove(os.path.join(dataset, image_to_delete))

    print("User database and picture dataset are consistent")


def new_user():
    name = input('\n Enter user name: ')
    user_id = str(abs(hash(name)) % (10**8))
    user_dict[user_id] = name
    input('Get ready for face recognition <Press any key>')
    learner.recognize(user_id, user_dict)
    print("User %s learned" % name)


def delete_user(name):
    # uid_to_delete = ''
    for uid in user_dict:
        # print(uid)
        if user_dict[uid] == name:
            uid_to_delete = uid
            # uid = int(os.path.split(img_path)[-1].split('.')[1])

            # удалить все фотки пользователя. получает id и сравнивает с обрезанным куском пути к файлу, перебор файлов
            for file_name in os.listdir(config.dataset_dir):
                if uid_to_delete == os.path.split(file_name)[-1].split('.')[1]:
                    os.remove(os.path.join(config.dataset_dir, file_name))
                    print("Deleted file: %s" % file_name)

            try:  # удалить пару из словаря
                del user_dict[uid_to_delete]
                print("Deleted user_dict for user " + user_dict[uid_to_delete])
                # записать обновленный словарь в файл
                save_db()
                print("<<<DB renewed>>>")
            except ValueError:
                print("User " + name + " not found if database")


user_dict = read_db()
LBPH_set = {"recognizer": cv2.face.LBPHFaceRecognizer_create(), "YML_name": "LBPHFaceRecognizer"}
Eigen_set = {"recognizer": cv2.face.EigenFaceRecognizer_create(), "YML_name": "EigenFaceRecognizer"}
Fisher_set = {"recognizer": cv2.face.FisherFaceRecognizer_create(), "YML_name": "FisherFaceRecognizer"}
# trainer.train_recognizer(Fisher_set)
# check_db_and_pictures()
predictor.predict_user(LBPH_set)
