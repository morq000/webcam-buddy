import os
from AIMirror import config, __main__

# Check if dataset path exists and, if not, create it
if not os.path.exists(config.dataset_dir):
    try:
        os.mkdir(config.dataset_dir, 0o755)
        print('Directory successfully created')
    except OSError:
        print('Creating dataset directory failed')


if not os.path.exists(config.trainer_dir):
    try:
        os.mkdir(config.trainer_dir, 0o755)
    except OSError:
        print('Creating trainer directory failed')


if __name__ == "__main__":
    __main__.__init__()
