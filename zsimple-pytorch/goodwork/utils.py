import os

def make_define_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path