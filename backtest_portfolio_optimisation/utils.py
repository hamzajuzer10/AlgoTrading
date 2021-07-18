import os


def save_file_path(folder_name:str, filename:str, wd=None):
    """Outpyts the full filepath and creates the folder if doesnt exist"""

    cwd = os.getcwd() if wd is None else wd
    model_dir = os.path.join(cwd, folder_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    file_dir = os.path.join(model_dir, filename)

    return file_dir