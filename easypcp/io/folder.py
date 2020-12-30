import os
import time
from send2trash import send2trash

def make_dir(dir_path, clean=False):
    if os.path.exists(dir_path):
        if clean:
            # shutil.rmtree(dir_path)
            send2trash(dir_path)
            print(f'[pnt][I/O] has delete [{dir_path}] to recycle bin')
            time.sleep(1)  # ensure the folder is cleared
            os.mkdir(dir_path)
    else:
        os.makedirs(dir_path)
