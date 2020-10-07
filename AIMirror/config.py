import os

users_list_name = r'users_list_database.json'
project_dir = r'C:\Users\HP-888\Documents\__PROJECTS__\FaceRecognizerTest'
dataset_dir = r'WebcamTestv1'  # Folder with user pictures
trainer_dir = r'Trainer'
cascades_path = r'C:\Users\HP-888\anaconda3\pkgs\libopencv-4.4.0-py37_2\Library\etc\haarcascades'  # с этим придется помудиться

while not os.path.exists(cascades_path):
    print('Path to Haar cascades %s doesn\'t exist.' % cascades_path)
    cascades_path = input(' Please specify path to cascades:')
