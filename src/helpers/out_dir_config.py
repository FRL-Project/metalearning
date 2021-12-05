import os

from datetime import datetime


def get_out_dir(calling_file='', postfix=None):
    # Out-dir from env var
    out_dir = os.getenv("OUT_DIR")
    exp_name = os.path.basename(calling_file)[:-3]
    date_string = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    folder_name = date_string + "_" + exp_name

    if postfix is not None:
        folder_name += "_" + postfix

    new_dir = os.path.join(out_dir, folder_name)

    return new_dir
