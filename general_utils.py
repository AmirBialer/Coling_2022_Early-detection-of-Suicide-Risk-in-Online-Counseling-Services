import datetime, random, torch, os, pickle, re
import numpy as np


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def create_dir_path(path, seed_val, tail_or_head, model_name):
    dir_path = path + str(seed_val)
    if not os.path.isdir(dir_path): os.makedirs(dir_path)
    dir_path += "/" + tail_or_head + "/"
    if not os.path.isdir(dir_path): os.makedirs(dir_path)
    dir_path += "/" + model_name + "/"
    if not os.path.isdir(dir_path): os.makedirs(dir_path)
    if not os.path.isdir(dir_path + "Best_Model"): os.makedirs(dir_path + "Best_Model")
    if not os.path.isdir(dir_path + "FT_Model"): os.makedirs(dir_path + "FT_Model")
    return dir_path


def save_pickle(file, filepath):
    p = re.findall(".*\/", filepath)[0]
    if not os.path.isdir(p): os.makedirs(p)
    pickle.dump(file, open(filepath, "wb"))
