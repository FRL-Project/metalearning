import os

from datetime import datetime


def get_out_dir(calling_file='', create_dir=False):
    # Out-dir from env var
    out_dir = os.getenv("OUT_DIR")
    exp_name = os.path.basename(calling_file)[:-3]
    date_string = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    new_dir = os.path.join(out_dir, date_string + "_" + exp_name)
    if create_dir:
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
    return new_dir
